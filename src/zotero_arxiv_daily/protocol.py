from dataclasses import dataclass
from typing import Optional, TypeVar, Any
from datetime import datetime
import re
import json

import tiktoken
from openai import OpenAI
from loguru import logger


RawPaperItem = TypeVar("RawPaperItem")


def dedupe_keep_order(items: list[str]) -> list[str]:
    """
    去重但保留原始顺序。
    不使用 list(set(...))，避免每次输出顺序随机。
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


def extract_first_json_object(text: str) -> dict[str, Any]:
    """
    从模型输出中尽量解析出第一个 JSON object。

    比 re.search(r'\\{.*\\}') 更安全：
    1. 先尝试直接 json.loads；
    2. 如果失败，再从每一个 '{' 位置尝试 raw_decode；
    3. 全部失败则返回空 dict。
    """
    if not text:
        return {}

    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

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
    这是兜底逻辑，不作为主路径。

    例如：
        {"motivation": "xxx
    虽然不是合法 JSON，但可以尽量把 xxx 抽出来。
    """
    if not text:
        return ""

    pattern = rf'"{re.escape(field_name)}"\s*:\s*"([^"]*)'
    match = re.search(pattern, text, flags=re.DOTALL)

    if not match:
        return ""

    value = match.group(1)
    value = value.replace("\\n", " ").replace('\\"', '"')
    value = re.sub(r"\s+", " ", value).strip()

    return value


def clean_summary_part(text: str) -> str:
    """
    清洗 motivation / method / results 单个字段。

    注意：
    这里是保守清洗，避免误删有效摘要。
    """
    if not text:
        return ""

    text = str(text).strip()

    # 去掉 markdown code fence
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # 去掉字段标签
    text = re.sub(
        r"^\s*(?:motivation|method|results|result|动机|方法|结果|实验结果)\s*[:：]\s*",
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

    # 如果模型仍然输出了完整 TLDR 前缀，也清理掉
    text = re.sub(
        r"^\s*(?:TLDR|TL;DR|中文摘要|一句话摘要|一句话总结|摘要|总结)\s*[:：]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # 去掉外层引号和多余空白
    text = text.strip().strip("\"'“”‘’").strip()

    # 压缩空白
    text = re.sub(r"\s+", " ", text)

    # 去掉末尾多余句号，后面统一拼接
    text = text.rstrip("。；;，, ")

    return text


def clean_tldr(text: str) -> str:
    """
    清洗完整 TLDR。
    用于 fallback 或兼容旧逻辑。
    """
    if not text:
        return ""

    tldr = str(text).strip()

    # 去掉 markdown code fence
    tldr = re.sub(r"^```(?:json)?\s*", "", tldr, flags=re.IGNORECASE)
    tldr = re.sub(r"\s*```$", "", tldr)

    # 去掉常见标签前缀
    tldr = re.sub(
        r"^\s*(?:TLDR|TL;DR|中文摘要|一句话摘要|一句话总结|摘要|总结)\s*[:：]\s*",
        "",
        tldr,
        flags=re.IGNORECASE,
    )

    # 去掉不希望出现的开头套话
    tldr = re.sub(
        r"^\s*(?:该论文|本文|该研究|这篇论文|这项工作)\s*",
        "",
        tldr,
        flags=re.IGNORECASE,
    )

    # 去掉外层引号和多余空白
    tldr = tldr.strip().strip("\"'“”‘’").strip()

    # 压缩空白
    tldr = re.sub(r"\s+", " ", tldr)

    return tldr


def format_three_part_tldr(
    motivation: str,
    method: str,
    results: str,
) -> str:
    """
    将三段式结果拼成统一 TLDR 内容。

    返回格式：
        动机：xxx。方法：xxx。结果：xxx。

    注意：
        这里不加最前面的 "TLDR:"。
        因为你的展示层通常已经有：
            print(f"TLDR: {paper.tldr}")
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


def truncate_text_by_tokens(
    text: str,
    model_name: str = "gpt-4o",
    max_tokens: int = 4000,
) -> str:
    """
    按 token 截断文本，避免 prompt 过长。
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
        使用 LLM 生成三段式论文 TLDR：
            动机：xxx。方法：xxx。结果：xxx。
        """
        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "动机：未提供明确研究动机。方法：未提供明确方法细节。结果：未提供明确实验结果。"

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

要求：
1. 只基于给定文本总结，不要补充文本中没有的信息。
2. motivation 说明论文要解决的问题、背景痛点或研究动机。
3. method 说明论文提出的方法、模型、框架或核心技术路线。
4. results 说明论文报告的实验效果、性能提升、验证结论或应用结果。
5. 每个字段都必须是中文短句。
6. 每个字段控制在 20 到 70 个中文字符之间。
7. 不要输出“该论文”“本文”“这篇论文”“该研究”等开头套话。
8. 不要输出“Motivation:”“Method:”“Results:”“动机：”“方法：”“结果：”等标签，字段名由 JSON 提供即可。
9. 如果给定文本中没有明确结果，results 字段写“未提供明确实验结果”。
10. 如果给定文本中没有明确方法，method 字段写“未提供明确方法细节”。
11. 如果给定文本中没有明确动机，motivation 字段写“未提供明确研究动机”。
12. 禁止输出解释、检查过程、思考过程、markdown、代码块或额外文本。
13. 必须严格返回 JSON 对象，不要在 JSON 外输出任何文字。
""".strip()

        summary_schema = {
            "type": "object",
            "properties": {
                "motivation": {
                    "type": "string",
                    "description": "论文研究动机，中文短句，不含标签。",
                },
                "method": {
                    "type": "string",
                    "description": "论文核心方法，中文短句，不含标签。",
                },
                "results": {
                    "type": "string",
                    "description": "论文实验结果或验证结论，中文短句；若文本没有明确结果，写未提供明确实验结果。",
                },
            },
            "required": ["motivation", "method", "results"],
            "additionalProperties": False,
        }

        gen_kwargs = llm_params.get("generation_kwargs", {}).copy()
        gen_kwargs.setdefault("temperature", 0.1)
        gen_kwargs.setdefault("max_tokens", 300)

        # 如果调用方没有显式传入 response_format，则默认使用 JSON Schema。
        # 如果你的模型或服务端不支持 json_schema，可以删除这一段。
        gen_kwargs.setdefault(
            "response_format",
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "paper_structured_tldr",
                    "strict": True,
                    "schema": summary_schema,
                },
            },
        )

        response = openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            **gen_kwargs,
        )

        message = response.choices[0].message

        if getattr(message, "refusal", None):
            logger.warning(
                f"Model refused to generate tldr for {self.url}: {message.refusal}"
            )
            return "动机：模型拒绝生成摘要。方法：未提供明确方法细节。结果：未提供明确实验结果。"

        raw_content = (message.content or "").strip()

        data = extract_first_json_object(raw_content)

        motivation = clean_summary_part(data.get("motivation", ""))
        method = clean_summary_part(data.get("method", ""))
        results = clean_summary_part(data.get("results", ""))

        # JSON 不完整时的兜底提取
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

        tldr = format_three_part_tldr(
            motivation=motivation,
            method=method,
            results=results,
        )

        if not tldr:
            logger.warning(
                f"Failed to parse valid structured tldr from raw content: {raw_content}"
            )
            return "动机：未提供明确研究动机。方法：未提供明确方法细节。结果：未提供明确实验结果。"

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
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")

            if self.abstract:
                abstract = clean_tldr(self.abstract[:180])
                fallback = f"动机：未提供明确研究动机。方法：{abstract}。结果：未提供明确实验结果。"
            elif self.title:
                fallback = f"动机：围绕“{self.title}”展开研究。方法：未提供明确方法细节。结果：未提供明确实验结果。"
            else:
                fallback = "动机：未提供明确研究动机。方法：未提供明确方法细节。结果：未提供明确实验结果。"

            self.tldr = fallback
            return fallback

    def _generate_affiliations_with_llm(
        self,
        openai_client: OpenAI,
        llm_params: dict,
    ) -> Optional[list[str]]:
        """
        使用 LLM 从 full_text 中提取作者机构。
        """
        if self.full_text is None:
            return None

        prompt = f"""
请从以下论文文本中提取作者所属机构。

【提取规则】
1. 只提取文本中明确出现的机构名称，不要根据作者、邮箱、标题或常识猜测。
2. 提取最高层级机构名称，例如：
   - "Department of Computer Science, Tsinghua University" 只提取 "Tsinghua University"
   - "MIT CSAIL" 优先提取 "Massachusetts Institute of Technology"；如果全文只写了 "MIT CSAIL"，则保留 "MIT CSAIL"
3. 可以提取大学、公司、研究院、实验室所属的顶层组织，例如 "Google DeepMind"、"Microsoft Research"、"Stanford University"。
4. 不要提取国家、城市、邮箱域名、作者姓名、基金项目。
5. 不要提取 "Department of ..."、"School of ..."、"College of ..." 这类中间层级，除非没有更高层级机构。
6. 输出英文正式机构名，尽量保留原文写法。
7. 如果没有明确机构，返回空数组。

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
必须严格返回 JSON 对象，不要在 JSON 外输出任何文字。
""".strip()

        affiliation_schema = {
            "type": "object",
            "properties": {
                "affiliations": {
                    "type": "array",
                    "description": "论文作者机构列表，去重，保留原文中的最高层级机构名称。",
                    "items": {"type": "string"},
                }
            },
            "required": ["affiliations"],
            "additionalProperties": False,
        }

        gen_kwargs = llm_params.get("generation_kwargs", {}).copy()
        gen_kwargs.setdefault("temperature", 0)
        gen_kwargs.setdefault("max_tokens", 256)

        # 如果调用方没有显式传入 response_format，则默认使用 JSON Schema。
        # 如果你的模型或服务端不支持 json_schema，可以删除这一段。
        gen_kwargs.setdefault(
            "response_format",
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "paper_affiliations",
                    "strict": True,
                    "schema": affiliation_schema,
                },
            },
        )

        response = openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
            **gen_kwargs,
        )

        message = response.choices[0].message

        if getattr(message, "refusal", None):
            logger.warning(
                f"Model refused to extract affiliations for {self.url}: {message.refusal}"
            )
            return []

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
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = None
            return None


@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
