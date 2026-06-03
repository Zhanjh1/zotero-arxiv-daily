from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper
import random
from datetime import datetime
from .reranker import get_reranker_cls
from .construct_email import render_email
from .utils import send_email
from openai import OpenAI
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
import re

def normalize_path_patterns(patterns: list[str] | ListConfig | None, config_key: str) -> list[str] | None:
    if patterns is None:
        return None
    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(f"config.zotero.{config_key} must be a list of glob patterns or null.")
    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.zotero.{config_key} must contain only strings.")
    return list(patterns)

class Executor:
    def __init__(self, config:DictConfig):
        self.config = config
        self.include_path_patterns = normalize_path_patterns(config.zotero.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.zotero.ignore_path, "ignore_path")
        self.retrievers = {source: get_retriever_cls(source)(config) for source in config.executor.source}
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.topic_keywords = getattr(self.config.source.arxiv, "keywords", []) or []
        if self.topic_keywords:
            self.topic_embeddings = self.embedding_model.encode(self.topic_keywords, convert_to_tensor=True)
        self.top_k = getattr(config.executor, "top_k", 10)

    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = {c['key']:c for c in zot.everything(zot.collections())}
        corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
        corpus = [c for c in corpus if c['data']['abstractNote'] != '']
        def get_collection_path(col_key:str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']
        for c in corpus:
            c['paths'] = [get_collection_path(col) for col in c['data']['collections']]
        return [CorpusPaper(
            title=c['data']['title'],
            abstract=c['data']['abstractNote'],
            added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
            paths=c['paths']
        ) for c in corpus]

    def filter_corpus(self, corpus:list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            corpus = [c for c in corpus if any(glob_match(path, pattern)
                                               for path in c.paths
                                               for pattern in self.include_path_patterns)]
        if self.ignore_path_patterns:
            corpus = [c for c in corpus if not any(glob_match(path, pattern)
                                                   for path in c.paths
                                                   for pattern in self.ignore_path_patterns)]
        return corpus

    def semantic_filter_topk(self, papers:list[CorpusPaper]):
        """语义过滤：保留 top-k 论文"""
        if not self.topic_keywords:
            return papers, [set() for _ in papers]
        abstracts = [p.title + " " + p.abstract for p in papers]
        paper_embeddings = self.embedding_model.encode(abstracts, convert_to_tensor=True)
        cosine_scores = util.cos_sim(paper_embeddings, self.topic_embeddings).cpu()
        # 每篇论文找到最大分数对应的关键词
        scores, indices = torch.max(cosine_scores, dim=1)
        paper_score_kw = list(zip(papers, scores.tolist(), indices.tolist()))
        # 按分数排序
        paper_score_kw.sort(key=lambda x: x[1], reverse=True)
        topk = paper_score_kw[:self.top_k]
        filtered, matched_keywords_list = [], []
        for p, score, idx in topk:
            filtered.append(p)
            matched_keywords_list.append({self.topic_keywords[idx]})
        logger.info(f"Top-{self.top_k} semantic filter: {len(papers)} -> {len(filtered)}")
        return filtered, matched_keywords_list

    def keyword_fallback_filter(self, papers:list[CorpusPaper]):
        """正则模糊匹配关键词"""
        if not self.topic_keywords:
            return papers, [set() for _ in papers]
        filtered, matched_keywords_list = [], []
        keywords_lower = [k.lower() for k in self.topic_keywords]
        for p in papers:
            text = (p.title + " " + p.abstract).lower()
            matched = set(k for k in keywords_lower if re.search(r'\b' + re.escape(k) + r'\b', text))
            if matched:
                filtered.append(p)
                matched_keywords_list.append(matched)
        logger.info(f"Keyword fallback filter: {len(papers)} -> {len(filtered)}")
        return filtered, matched_keywords_list

    def run(self):
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.error("No zotero papers found")
            return

        all_papers = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()
            all_papers.extend(papers)
        logger.info(f"Retrieved {len(all_papers)} papers from all sources")

        # 语义 Top-K 过滤
        filtered_papers, matched_keywords_list = self.semantic_filter_topk(all_papers)

        # fallback
        if len(filtered_papers) == 0:
            logger.warning("Semantic filter removed all papers, fallback to keyword regex filter")
            filtered_papers, matched_keywords_list = self.keyword_fallback_filter(all_papers)

        # 如果仍为空，就发原始全部
        if len(filtered_papers) == 0:
            logger.warning("No papers passed filters, sending all papers")
            filtered_papers = all_papers
            matched_keywords_list = [set() for _ in all_papers]

        reranked_papers = []
        if len(filtered_papers) > 0:
            reranked_papers = self.reranker.rerank(filtered_papers, corpus)
            reranked_papers = reranked_papers[:self.config.executor.max_paper_num]
            for p, kw in zip(reranked_papers, matched_keywords_list):
                p.match_keywords = kw
                p.generate_tldr(self.openai_client, self.config.llm)
                p.generate_affiliations(self.openai_client, self.config.llm)

        email_content = render_email(reranked_papers)
        send_email(self.config, email_content)
        logger.info("Email sent successfully")
