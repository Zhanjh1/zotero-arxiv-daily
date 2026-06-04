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


def clean_tldr(text: str) -> str:
    """
    保守清洗 TLDR。

    只清理明确无用的标签、markdown code fence、外层引号和少量套话。
    不做过度正则替换，避免误删正常摘要内容。
    """
    if not text:
        return ""

    tldr = str(text).strip()

    # 去掉 markdown code fence
    tldr = re.sub(r"^```(?:json)?\s*", "", tldr, flags=re.IGNORECASE)
    tldr = re.sub(r"\s*```$", "", tldr)

    # 如果模型仍然输出了标签前缀，去掉它
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
        使用 LLM 生成一句话中文 TLDR。
        """
        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "无法生成摘要：未提供足够文本。"

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
你是一个严谨的论文信息抽取助手。你的任务是根据给定的论文标题、摘要和正文片段，生成一个中文 TLDR。

要求：
1. 只基于给定文本总结，不要补充文本中没有的信息。
2. 输出必须是“一句话中文摘要”。
3. 摘要应直接描述论文的核心贡献、主要方法或核心问题。
4. 不要使用“该论文”“本文”“这篇论文”“该研究”等开头套话。
5. 不要输出“TLDR:”“摘要:”等标签。
6. 不要使用项目符号、编号、换行。
7. 尽量控制在 40 到 90 个中文字符之间。
8. 如果信息不足，只能概括已有信息，不要猜测实验结果、数据集、机构或应用场景。
""".strip()

        tldr_schema = {
            "type": "object",
            "properties": {
                "tldr": {
                    "type": "string",
                    "description": "一句话中文论文摘要，不含标签，不以“该论文/本文/这篇论文/该研究”开头。",
                }
            },
            "required": ["tldr"],
            "additionalProperties": False,
        }

        gen_kwargs = llm_params.get("generation_kwargs", {}).copy()
        gen_kwargs.setdefault("temperature", 0.2)
        gen_kwargs.setdefault("max_tokens", 256)

        # 如果调用方没有显式传入 response_format，则默认使用 JSON Schema
        gen_kwargs.setdefault(
            "response_format",
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "paper_tldr",
                    "strict": True,
                    "schema": tldr_schema,
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
            return "（模型拒绝生成摘要）"

        raw_content = (message.content or "").strip()
        data = extract_first_json_object(raw_content)
        tldr = clean_tldr(data.get("tldr", ""))

        if not tldr:
            logger.warning(f"Failed to parse valid tldr from raw content: {raw_content}")
            return "（未能提取到有效摘要）"

        return tldr

    def generate_tldr(
        self,
        openai_client: OpenAI,
        llm_params: dict,
    ) -> str:
        """
        对外调用入口：生成 TLDR，并写入 self.tldr。
        """
        try:
            tldr = self._generate_tldr_with_llm(openai_client, llm_params)
            self.tldr = tldr
            return tldr

        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")

            if self.abstract:
                fallback = clean_tldr(self.abstract[:200]) + "..."
            elif self.title:
                fallback = f"围绕“{self.title}”展开研究，但缺少摘要或正文，无法生成可靠 TLDR。"
            else:
                fallback = "（未能生成摘要：缺少标题、摘要和正文信息）"

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

        # 如果调用方没有显式传入 response_format，则默认使用 JSON Schema
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
