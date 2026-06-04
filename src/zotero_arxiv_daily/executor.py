import re
import random
from datetime import datetime
from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig
from openai import OpenAI
from tqdm import tqdm

from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper
from .reranker import get_reranker_cls
from .construct_email import render_email
from .utils import send_email


def normalize_path_patterns(patterns: list[str] | ListConfig | None, config_key: str) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.zotero.{config_key} must contain only glob pattern strings.")

    return list(patterns)


class Executor:
    def __init__(self, config: DictConfig):
        self.config = config
        self.include_path_patterns = normalize_path_patterns(config.zotero.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.zotero.ignore_path, "ignore_path")
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)

    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = zot.everything(zot.collections())
        collections = {c['key']: c for c in collections}
        corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
        corpus = [c for c in corpus if c['data']['abstractNote'] != '']

        def get_collection_path(col_key: str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']

        for c in corpus:
            paths = [get_collection_path(col) for col in c['data']['collections']]
            c['paths'] = paths
        logger.info(f"Fetched {len(corpus)} zotero papers")
        return [CorpusPaper(
            title=c['data']['title'],
            abstract=c['data']['abstractNote'],
            added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
            paths=c['paths']
        ) for c in corpus]
    
    def filter_corpus(self, corpus: list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(f"Selecting zotero papers matching include_path: {self.include_path_patterns}")
            corpus = [
                c for c in corpus
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]
        if self.ignore_path_patterns:
            logger.info(f"Excluding zotero papers matching ignore_path: {self.ignore_path_patterns}")
            corpus = [
                c for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]
        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(corpus, min(5, len(corpus)))
            samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths) for c in samples])
            logger.info(f"Selected {len(corpus)} zotero papers:\n{samples}\n...")
        return corpus

    def _boost_keywords(self, papers: list) -> list:
        """
        核心关键词保驾护航机制：防止高价值的 VLA 论文在 Top-K 截断时被泛 AI/通用大模型论文挤掉
        """
        keywords = []
        try:
            # 尝试从配置文件中动态读取 arxiv 关键词列表
            if "source" in self.config and "arxiv" in self.config.source and "keywords" in self.config.source.arxiv:
                keywords = list(self.config.source.arxiv.keywords)
        except Exception:
            pass
        
        # 保底机制：若配置未读取成功，则自动采用内置的具身智能核心高价值词表
        if not keywords:
            keywords = ["VLA", "Vision-Language-Action", "Vision Language Action", "Embodied AI", "Embodied Agent"]
            
        logger.info(f"Starting keyword boosting prioritization using terms: {keywords}")
        
        patterns = []
        for kw in keywords:
            kw_stripped = kw.strip()
            if not kw_stripped:
                continue
            # 对 VLA 这种容易混淆在单词内部的短缩写，强制加上 \b 单词边界正则，防止误伤（如 flavor）
            if kw_stripped.isalnum() and len(kw_stripped) <= 4:
                patterns.append(rf"\b{re.escape(kw_stripped)}\b")
            else:
                patterns.append(re.escape(kw_stripped))
                
        if not patterns:
            return papers
            
        regex = re.compile("|".join(patterns), re.IGNORECASE)
        
        matched_papers = []
        unmatched_papers = []
        
        for p in papers:
            # 综合检索标题和摘要
            search_text = f"{p.title} {p.abstract}"
            if regex.search(search_text):
                matched_papers.append(p)
            else:
                unmatched_papers.append(p)
                
        logger.info(f"Keyword prioritization complete: {len(matched_papers)} papers promoted, {len(unmatched_papers)} normal papers remaining.")
        
        # 将命中强核心词的论文排在最前（且完美继承它们原有的 Reranker 相对先后顺序），普通论文紧随其后
        return matched_papers + unmatched_papers

    def run(self):
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.error(f"No zotero papers found. Please check your zotero settings:\n{self.config.zotero}")
            return
        all_papers = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)
        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")
        
        reranked_papers = []
        if len(all_papers) > 0:
            logger.info("Reranking papers...")
            # 1. 走原始 Reranker 计算与 Zotero 文献库的关联度
            reranked_papers = self.reranker.rerank(all_papers, corpus)
            
            # 2. 调用核心词拦截保送逻辑，确保真 VLA 论文挪到队列最前端
            reranked_papers = self._boost_keywords(reranked_papers)
            
            # 3. 此时再进行 Top-K 数量切片，即便后面位置不够用，被丢弃的也必然是那些不相关的“噪音”论文
            reranked_papers = reranked_papers[:self.config.executor.max_paper_num]
            
            logger.info("Generating TLDR and affiliations...")
            for p in tqdm(reranked_papers):
                p.generate_tldr(self.openai_client, self.config.llm)
                p.generate_affiliations(self.openai_client, self.config.llm)
        elif not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return
            
        logger.info("Sending email...")
        email_content = render_email(reranked_papers)
        send_email(self.config, email_content)
        logger.info("Email sent successfully")
