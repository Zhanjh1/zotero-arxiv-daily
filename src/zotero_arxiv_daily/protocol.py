from dataclasses import dataclass
from typing import Optional, TypeVar, Any
from datetime import datetime
import re
import json

import tiktoken
from openai import OpenAI
from loguru import logger
from omegaconf import OmegaConf

RawPaperItem = TypeVar("RawPaperItem")


# =========================
# 通用工具函数
# =========================

def dedupe_keep_order(items: list[str]) -> list[str]:
    """
    去重但保留原始顺序。
    不使用 list(set(...))，避免输出顺序随机。
    """
    seen = set()
    result = []

    for item in items:
        item = str(item).strip()
        if not item:
            continue

        key = item.lower()
        if key in seen:
            continue

        seen.add(key)
        result.append(item)

    return result


def truncate_text_by_tokens(
    text: str,
    model_name: str = "gpt-4o",
    max_tokens: int = 4000,
) -> str:
    """
    按 token 截断文本，避免 prompt 过长。

    注意：
    这里的 model_name 只是用于 tiktoken 估算 token。
    即使用 MiniMax，也可以用 gpt-4o / cl100k_base 做近似截断。
    """
    if not text:
        return ""

    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    tokens = enc.encode(text)

    if len(tokens) <= max_tokens:
        return text

    return enc.decode(tokens[:max_tokens])


def strip_code_fence(text: str) -> str:
    """
    去掉 markdown 代码块包裹。
    """
    if not text:
        return ""

    text = text.strip()
    text = re.sub(r"^```(?:json|JSON)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_first_json_object(text: str) -> dict[str, Any]:
    """
    从模型输出中尽量解析出第一个 JSON object。

    适用于模型输出：
        {"motivation": "...", "method": "...", "results": "..."}

    也适用于模型混入少量废话：
        好的，结果如下：
        {"motivation": "...", "method": "...", "results": "..."}
    """
    if not text:
        return {}

    text = strip_code_fence(text)

    # 1. 直接解析
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # 2. 从每个 { 开始尝试 raw_decode
    decoder = json.JSONDecoder()

    for i, ch in enumerate(text):
        if ch != "{":
            continue

        try:
            data, _ = decoder.raw_decode(text[i:])
            if isinstance(data, dict):
                return data
        except Exception:
            continue

    return {}


def extract_json_string_field_fallback(text: str, field_name: str) -> str:
    """
    当 JSON 不完整时，尝试从 '"field": "value"' 片段中提取字符串。

    例如模型输出了：
        {"motivation": "xxx", "method": "yyy

    虽然整体不是合法 JSON，但仍可尽量提取 method。
    """
    if not text:
        return ""

    text = strip_code_fence(text)

    pattern = rf'"{re.escape(field_name)}"\s*:\s*"([^"]*)'
    match = re.search(pattern, text, flags=re.DOTALL)

    if not match:
        return ""

    value = match.group(1)
    value = value.replace("\\n", " ")
    value = value.replace('\\"', '"')
    value = re.sub(r"\s+", " ", value).strip()

    return value


def has_english_sentence(text: str) -> bool:
    """
    粗略判断文本是否像英文句子。
    用于防止 MiniMax 直接复制英文 abstract。
    """
    if not text:
        return False

    ascii_letters = len(re.findall(r"[A-Za-z]", text))
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))

    # 英文字母明显多于中文字符，基本可判断为英文句子
    return ascii_letters >= 20 and ascii_letters > chinese_chars * 2


def clean_summary_part(text: str) -> str:
    """
    清洗 motivation / method / results 单个字段。
    """
    if not text:
        return ""

    text = str(text).strip()
    text = strip_code_fence(text)

    # 去掉字段标签
    text = re.sub(
        r"^\s*(?:motivation|method|methods|results|result|动机|方法|结果|实验结果|研究动机|核心方法)\s*[:：]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # 去掉完整 TLDR 前缀
    text = re.sub(
        r"^\s*(?:TLDR|TL;DR|中文摘要|一句话摘要|一句话总结|摘要|总结)\s*[:：]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # 去掉常见套话
    text = re.sub(
        r"^\s*(?:该论文|本文|该研究|这篇论文|这项工作)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # 去掉模型自检废话
    bad_prefix_patterns = [
        r"^这太长了.*?[:：]\s*",
        r"^让我精简.*?[:：]\s*",
        r"^检查.*?[:：]\s*",
        r"^最终.*?[:：]\s*",
        r"^输出.*?[:：]\s*",
    ]

    for pattern in bad_prefix_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL).strip()

    # 去掉外层引号和多余空白
    text = text.strip().strip("\"'“”‘’").strip()

    # 压缩空白
    text = re.sub(r"\s+", " ", text)

    # 去掉末尾多余标点，后面统一拼接
    text = text.rstrip("。；;，, ")

    return text


def clean_tldr(text: str) -> str:
    """
    清洗完整 TLDR。
    主要用于 fallback。
    """
    if not text:
        return ""

    text = str(text).strip()
    text = strip_code_fence(text)

    text = re.sub(
        r"^\s*(?:TLDR|TL;DR|中文摘要|一句话摘要|一句话总结|摘要|总结)\s*[:：]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^\s*(?:该论文|本文|该研究|这篇论文|这项工作)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.strip().strip("\"'“”‘’").strip()
    text = re.sub(r"\s+", " ", text)

    return text


def format_three_part_tldr(
    motivation: str,
    method: str,
    results: str,
) -> str:
    """
    拼接成统一格式。

    返回：
        动机：xxx。方法：xxx。结果：xxx。

    注意：
        这里不加最前面的 "TLDR:"。
        如果你的外层展示代码是：
            print(f"TLDR: {paper.tldr}")

        那最终就是：
            TLDR: 动机：xxx。方法：xxx。结果：xxx。
    """
    motivation = clean_summary_part(motivation)
    method = clean_summary_part(method)
    results = clean_summary_part(results)

    if not motivation:
        motivation = "未提供明确研究动机"

    if not method:
        method = "未提供明确方法细节"

    if not results:
        results = "未提供明确实验结果"

    return f"动机：{motivation}。方法：{method}。结果：{results}。"


def build_safe_fallback_tldr(title: str = "") -> str:
    """
    LLM 调用失败或解析失败时的安全 fallback。
    不再把英文 abstract 硬塞进“方法”。
    """
    title = str(title).strip()

    if title:
        return (
            f"动机：围绕“{title}”相关问题展开研究。"
            f"方法：未能从模型输出中稳定提取方法信息。"
            f"结果：未能从模型输出中稳定提取实验结果。"
        )

    return (
        "动机：未能从模型输出中稳定提取研究动机。"
        "方法：未能从模型输出中稳定提取方法信息。"
        "结果：未能从模型输出中稳定提取实验结果。"
    )


# =========================
# 数据结构
# =========================

@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None

    def _generate_tldr_with_llm(
        self,
        openai_client: OpenAI,
        llm_params: dict,
    ) -> str:
        """
        使用 MiniMax-M2.7 生成三段式论文 TLDR。

        返回格式：
            动机：xxx。方法：xxx。结果：xxx。
        """
        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return (
                "动机：未提供明确研究动机。"
                "方法：未提供明确方法细节。"
                "结果：未提供明确实验结果。"
            )

        paper_info = []

        if self.title:
            paper_info.append(f"【论文标题】\n{self.title}")

        if self.abstract:
            paper_info.append(f"【论文摘要】\n{self.abstract}")

        if self.full_text:
            paper_info.append(f"【正文片段】\n{self.full_text}")

        user_content = "\n\n".join(paper_info)
        user_content = truncate_text_by_tokens(
            user_content,
            model_name="gpt-4o",
            max_tokens=4000,
        )

        system_content = """
你是一个严谨的论文信息抽取助手。你的任务是根据给定的论文标题、摘要和正文片段，从 motivation、method、results 三个角度总结论文。

你必须严格遵守以下规则：

1. 只基于给定文本总结，不要补充文本中没有的信息。
2. 所有字段必须使用简体中文输出，即使原文是英文，也必须翻译和概括为中文。
3. 禁止直接复制英文原文句子。
4. motivation 说明论文要解决的问题、背景痛点或研究动机。
5. method 说明论文提出的方法、模型、框架或核心技术路线。
6. results 说明论文报告的实验效果、性能提升、验证结论或应用结果。
7. 每个字段都必须是中文短句。
8. 每个字段控制在 20 到 80 个中文字符之间。
9. 不要输出“该论文”“本文”“这篇论文”“该研究”等开头套话。
10. 不要输出“Motivation:”“Method:”“Results:”“动机：”“方法：”“结果：”等标签。
11. 如果给定文本中没有明确结果，results 字段写“未提供明确实验结果”。
12. 如果给定文本中没有明确方法，method 字段写“未提供明确方法细节”。
13. 如果给定文本中没有明确动机，motivation 字段写“未提供明确研究动机”。
14. 禁止输出解释、检查过程、思考过程、markdown、代码块或额外文本。
15. 只能返回一个 JSON 对象，不能返回 JSON 数组。
16. JSON 对象必须包含且只包含三个字段：motivation、method、results。

返回格式必须严格如下：
{"motivation":"这里写中文动机","method":"这里写中文方法","results":"这里写中文结果"}
""".strip()

        gen_kwargs = OmegaConf.to_container(
        llm_params.get("generation_kwargs", {}),
        resolve=True,
        )
    
        if gen_kwargs is None:
            gen_kwargs = {}
        
        gen_kwargs = dict(gen_kwargs)
        gen_kwargs.pop("response_format", None)

        # 降低随机性，减少自言自语和格式漂移
        gen_kwargs.setdefault("temperature", 0.1)
        gen_kwargs.setdefault("max_tokens", 300)

        response = openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            **gen_kwargs,
        )

        message = response.choices[0].message
        raw_content = (message.content or "").strip()

        data = extract_first_json_object(raw_content)

        motivation = clean_summary_part(data.get("motivation", ""))
        method = clean_summary_part(data.get("method", ""))
        results = clean_summary_part(data.get("results", ""))

        # JSON 不完整时兜底抽取字段
        if not motivation:
            motivation = clean_summary_part(
                extract_json_string_field_fallback(raw_content, "motivation")
            )

        if not method:
            method = clean_summary_part(
                extract_json_string_field_fallback(raw_content, "method")
            )

        if not results:
            results = clean_summary_part(
                extract_json_string_field_fallback(raw_content, "results")
            )

        # 防止直接复制英文 abstract
        if has_english_sentence(motivation):
            logger.warning(f"Motivation seems to be English, dropping it: {motivation}")
            motivation = ""

        if has_english_sentence(method):
            logger.warning(f"Method seems to be English, dropping it: {method}")
            method = ""

        if has_english_sentence(results):
            logger.warning(f"Results seems to be English, dropping it: {results}")
            results = ""

        tldr = format_three_part_tldr(
            motivation=motivation,
            method=method,
            results=results,
        )

        return tldr

    def generate_tldr(
        self,
        openai_client: OpenAI,
        llm_params: dict,
    ) -> str:
        """
        对外调用入口：生成 TLDR，并写入 self.tldr。

        返回格式：
            动机：xxx。方法：xxx。结果：xxx。
        """
        try:
            tldr = self._generate_tldr_with_llm(openai_client, llm_params)
            self.tldr = tldr
            return tldr

        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {repr(e)}")

            fallback = build_safe_fallback_tldr(self.title)
            self.tldr = fallback
            return fallback

    def _generate_affiliations_with_llm(
        self,
        openai_client: OpenAI,
        llm_params: dict,
    ) -> Optional[list[str]]:
        """
        使用 MiniMax-M2.7 从 full_text 中提取作者机构。
        """
        if self.full_text is None:
            return None

        prompt = f"""
请从以下论文文本中提取作者所属机构。

你必须严格遵守以下规则：

1. 只提取文本中明确出现的机构名称，不要根据作者、邮箱、标题或常识猜测。
2. 提取最高层级机构名称。
3. 例如 "Department of Computer Science, Tsinghua University" 只提取 "Tsinghua University"。
4. 例如 "MIT CSAIL" 如果全文没有更完整机构名，则保留 "MIT CSAIL"。
5. 可以提取大学、公司、研究院、实验室所属的顶层组织，例如 "Google DeepMind"、"Microsoft Research"、"Stanford University"。
6. 不要提取国家、城市、邮箱域名、作者姓名、基金项目。
7. 不要提取 "Department of ..."、"School of ..."、"College of ..." 这类中间层级，除非没有更高层级机构。
8. 输出英文正式机构名，尽量保留原文写法。
9. 如果没有明确机构，返回空数组。
10. 禁止输出解释、检查过程、思考过程、markdown、代码块或额外文本。
11. 只能返回一个 JSON 对象，不能返回 JSON 数组。
12. JSON 对象必须包含且只包含一个字段：affiliations。

返回格式必须严格如下：
{{"affiliations":["Institution A","Institution B"]}}

【论文文本】
{self.full_text}
""".strip()

        prompt = truncate_text_by_tokens(
            prompt,
            model_name="gpt-4o",
            max_tokens=2500,
        )

        system_content = """
你是一个严谨的学术机构抽取助手。你只能从用户给定文本中抽取明确出现的机构名称，不能猜测、补全或扩写文本中没有的信息。

你只能返回 JSON 对象，不要在 JSON 外输出任何文字。

返回格式：
{"affiliations":["Institution A","Institution B"]}
""".strip()

        gen_kwargs = OmegaConf.to_container(
            llm_params.get("generation_kwargs", {}),
            resolve=True,
        )
        
        if gen_kwargs is None:
            gen_kwargs = {}
        
        gen_kwargs = dict(gen_kwargs)
        gen_kwargs.pop("response_format", None)

        gen_kwargs.setdefault("temperature", 0)
        gen_kwargs.setdefault("max_tokens", 256)

        response = openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
            **gen_kwargs,
        )

        message = response.choices[0].message
        raw_content = (message.content or "").strip()

        data = extract_first_json_object(raw_content)

        affiliations = data.get("affiliations", [])

        if not isinstance(affiliations, list):
            affiliations = []

        affiliations = [
            str(a).strip()
            for a in affiliations
            if str(a).strip()
        ]

        affiliations = dedupe_keep_order(affiliations)

        return affiliations

    def generate_affiliations(
        self,
        openai_client: OpenAI,
        llm_params: dict,
    ) -> Optional[list[str]]:
        """
        对外调用入口：生成 affiliations，并写入 self.affiliations。
        """
        try:
            affiliations = self._generate_affiliations_with_llm(
                openai_client,
                llm_params,
            )
            self.affiliations = affiliations
            return affiliations

        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {repr(e)}")
            self.affiliations = None
            return None


@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
