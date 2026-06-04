from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json

RawPaperItem = TypeVar('RawPaperItem')

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

    def _generate_tldr_with_llm(self, openai_client: OpenAI, llm_params: dict) -> str:
        # 构建完全中文的 User Prompt
        prompt = "请阅读以下论文信息，并为其生成准确的、仅限三句话的中文 TLDR 摘要（简明扼要地概括核心贡献与方法）：\n\n"
        if self.title:
            prompt += f"论文标题 (Title):\n {self.title}\n\n"

        if self.abstract:
            prompt += f"论文摘要 (Abstract):\n {self.abstract}\n\n"

        if self.full_text:
            prompt += f"正文片段 (Preview):\n {self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "无法生成摘要：未提供摘要或正文内容。"
        
        # 使用 gpt-4o 分词器进行长度估算并截断
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # 限制在 4000 词元内
        prompt = enc.decode(prompt_tokens)
        
        # 极度严厉的中文 System Prompt，彻底压制国内模型的“碎碎念”
        system_content = (
            "你是一个顶尖的科研助理。你的唯一任务是提供论文的核心一句话中文摘要（TLDR）。\n"
            "【核心红线规则（绝对必须遵守）】:\n"
            "1. 只能输出最终的【一句话中文摘要】，严禁包含任何前言、后记、寒暄或解释性文本。\n"
            "2. 绝对不能输出你的内心独白、思考过程、推理步骤，或对用户指令的复读（例如严禁出现 '让我来为你分析...'）。\n"
            "3. 摘要必须直接以核心事实或贡献开头，严禁带有诸如 'TLDR:', '摘要:', '中文摘要:', '本文指出:', '总结来说:' 等任何前缀标签或双引号。"
        )
        
        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_content,
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get('generation_kwargs', {})
        )
        tldr = response.choices[0].message.content.strip()
        
        # 后置强力清洗防御：防止部分模型强行加前缀
        if tldr:
            # 剥离可能生成的各类标签前缀
            markers = [r"生成中文摘要[：:]", r"中文摘要[：:]", r"摘要[：:]", r"TLDR[：:]", r"Summary[：:]"]
            for marker in markers:
                parts = re.split(marker, tldr, flags=re.IGNORECASE)
                if len(parts) > 1:
                    tldr = parts[-1].strip()
            
            # 清理句首常见的废话词组
            tldr = re.sub(r'^(该论文|本文|根据.*?信息|一句话总结)[:：\s]*', '', tldr, flags=re.IGNORECASE)
            # 剥离大模型习惯外包的双引号或单引号
            tldr = tldr.strip('"\' \n\r')
            
        return tldr
    
    def generate_tldr(self, openai_client: OpenAI, llm_params: dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client, llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            tldr = self.abstract
            self.tldr = tldr
            return tldr

    def _generate_affiliations_with_llm(self, openai_client: OpenAI, llm_params: dict) -> Optional[list[str]]:
        if self.full_text is not None:
            # 中文 User Prompt
            prompt = f"请阅读以下论文开篇文本，提取所有作者的最高层级机构名称（Affiliations），并严格以 Python 列表格式返回。如果没有找到任何机构，请直接返回空列表 '[]'：\n\n{self.full_text}"
            
            # 长度截断
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:2000]
            prompt = enc.decode(prompt_tokens)
            
            # 极度严厉的机构提取中文 System Prompt
            system_content = (
                "你是一个专门从学术论文中精准提取机构名称的自动化工具。\n"
                "【核心红线规则（绝对必须遵守）】:\n"
                "1. 你必须只返回一个合法的 Python 列表（或 JSON 数组）格式，例如 [\"Tsinghua University\", \"Peking University\"]。\n"
                "2. 严禁输出任何多余的解释、中间推理、前言导语（如 '好的，为您提取如下：'）或后记。\n"
                "3. 机构名称应按作者顺序排列。如果有多个作者属于同一机构，请去重保持唯一。\n"
                "4. 如果包含多级机构（如 'Department of Computer Science, Tsinghua University'），请只保留最高层级的独立法人机构名称（如 'Tsinghua University'）。\n"
                "5. 若文本中未找到任何有效机构，请直接返回包含四个字符的空列表：[]"
            )
            
            affiliations = openai_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_content,
                    },
                    {"role": "user", "content": prompt},
                ],
                **llm_params.get('generation_kwargs', {})
            )
            affiliations_content = affiliations.choices[0].message.content.strip()

            # 强力鲁棒性正则清洗，只捞取方括号中间的内容，防止模型在列表外包裹废话
            match = re.search(r'\[.*?\]', affiliations_content, flags=re.DOTALL)
            if not match:
                logger.warning(f"No list found in LLM affiliations response for {self.url}")
                return []
                
            affiliations_str = match.group(0)
            affiliations_list = json.loads(affiliations_str)
            affiliations_list = list(set(affiliations_list))  # 再次去重保底
            affiliations_list = [str(a).strip() for a in affiliations_list if str(a).strip()]

            return affiliations_list
        return None
    
    def generate_affiliations(self, openai_client: OpenAI, llm_params: dict) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client, llm_params)
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
