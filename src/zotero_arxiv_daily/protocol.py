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
        prompt = "请阅读以下论文信息：\n\n"
        if self.title:
            prompt += f"论文标题:\n {self.title}\n\n"
        if self.abstract:
            prompt += f"论文摘要:\n {self.abstract}\n\n"
        if self.full_text:
            prompt += f"正文片段:\n {self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "无法生成摘要：未提供足够文本。"
        
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]
        prompt = enc.decode(prompt_tokens)
        
        # 终极武器：只准输出标准 JSON
        system_content = (
            "你是一个自动化数据提取机器人。请将给定的论文总结为一句话中文摘要（TLDR）。\n"
            "【绝对红线规则】:\n"
            "你必须严格返回一个 JSON 对象，严禁包含任何前导语（如'好的'）或后续解释。整个输出必须能直接被 json.loads() 解析。\n"
            "返回的 JSON 格式必须精确如下：\n"
            '{"tldr": "这里写一句话中文摘要，直接以论文核心贡献开头，严禁包含前缀标签"}'
        )
        
        # 强制开启可能支持的 JSON Mode
        gen_kwargs = llm_params.get('generation_kwargs', {}).copy()
        
        response = openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
            **gen_kwargs
        )
        raw_content = response.choices[0].message.content.strip()
        
        # --- 后处理清洗逻辑 ---
        try:
            # 1. 尝试直接正则捞取 {}
            json_match = re.search(r'\{.*\}', raw_content, flags=re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                tldr = data.get("tldr", "").strip()
            else:
                tldr = raw_content
        except Exception as json_err:
            logger.warning(f"JSON parse failed, falling back to regex: {json_err}")
            tldr = raw_content

        # 2. 文本保底清洗（干掉残存的废话）
        if tldr:
            markers = [r"生成中文摘要[：:]", r"中文摘要[：:]", r"摘要[：:]", r"TLDR[：:]"]
            for marker in markers:
                parts = re.split(marker, tldr, flags=re.IGNORECASE)
                if len(parts) > 1:
                    tldr = parts[-1].strip()
            tldr = re.sub(r'^(该论文|本文|根据.*?信息|一句话总结)[:：\s]*', '', tldr, flags=re.IGNORECASE)
            tldr = tldr.strip('"\' \n\r{}')
            
        return tldr if tldr else "（未能提取到有效摘要）"
    
    def generate_tldr(self, openai_client: OpenAI, llm_params: dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client, llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            self.tldr = self.abstract[:200] + "..."
            return self.tldr

    def _generate_affiliations_with_llm(self, openai_client: OpenAI, llm_params: dict) -> Optional[list[str]]:
        if self.full_text is not None:
            prompt = f"请提取以下文本中所有作者的最高层级机构名称（如 Tsinghua University）：\n\n{self.full_text}"
            
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:2000]
            prompt = enc.decode(prompt_tokens)
            
            system_content = (
                "你是一个自动化数据提取机器人。请提取文本中的学术机构。\n"
                "【绝对红线规则】:\n"
                "你必须严格返回一个 JSON 对象，严禁包含任何多余文本。\n"
                "返回的 JSON 格式必须精确如下：\n"
                '{"affiliations": ["Institution A", "Institution B"]}\n'
                "如果未找到任何机构，返回：\n"
                '{"affiliations": []}'
            )
            
            gen_kwargs = llm_params.get('generation_kwargs', {}).copy()
            
            response = openai_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                **gen_kwargs
            )
            raw_content = response.choices[0].message.content.strip()

            try:
                json_match = re.search(r'\{.*\}', raw_content, flags=re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    affiliations_list = data.get("affiliations", [])
                else:
                    affiliations_list = []
            except Exception:
                affiliations_list = []

            affiliations_list = list(set([str(a).strip() for a in affiliations_list if str(a).strip()]))
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
