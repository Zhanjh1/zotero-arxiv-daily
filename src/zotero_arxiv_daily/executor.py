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

# ========== NEW ==========
from sentence_transformers import SentenceTransformer, util
import torch
# =========================


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
            source: get_retriever_cls(source)(config)
            for source in config.executor.source
        }

        self.reranker = get_reranker_cls(config.executor.reranker)(config)

        self.openai_client = OpenAI(
            api_key=config.llm.api.key,
            base_url=config.llm.api.base_url
        )

        # ========== NEW: embedding model ==========
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # topic keywords -> embedding
        self.topic_keywords = getattr(config.source.arxiv, "keywords", None)

        if self.topic_keywords:
            topic_text = " ; ".join(self.topic_keywords)
            self.topic_embedding = self.embedding_model.encode(
                topic_text,
                convert_to_tensor=True
            )
        else:
            self.topic_embedding = None

    # =========================
    # Zotero corpus
    # =========================
    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")

        zot = zotero.Zotero(
            self.config.zotero.user_id,
            'user',
            self.config.zotero.api_key
        )

        collections = zot.everything(zot.collections())
        collections = {c['key']: c for c in collections}

        corpus = zot.everything(
            zot.items(itemType='conferencePaper || journalArticle || preprint')
        )

        corpus = [c for c in corpus if c['data']['abstractNote'] != '']

        def get_collection_path(col_key: str) -> str:
            parent = collections[col_key]['data']['parentCollection']
            if parent:
                return get_collection_path(parent) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']

        for c in corpus:
            paths = [get_collection_path(col) for col in c['data']['collections']]
            c['paths'] = paths

        logger.info(f"Fetched {len(corpus)} zotero papers")

        return [
            CorpusPaper(
                title=c['data']['title'],
                abstract=c['data']['abstractNote'],
                added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
                paths=c['paths']
            )
            for c in corpus
        ]

    # =========================
    # path filter
    # =========================
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

    # =========================
    # NEW: semantic filter
    # =========================
    def semantic_filter(self, papers: list[CorpusPaper], threshold: float = 0.45):
        if self.topic_embedding is None:
            return papers

        texts = [p.title + " " + p.abstract for p in papers]

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_tensor=True
        )

        scores = util.cos_sim(embeddings, self.topic_embedding).cpu().numpy().flatten()

        filtered = []
        for p, s in zip(papers, scores):
            if s >= threshold:
                filtered.append(p)

        logger.info(f"Semantic filter: {len(papers)} -> {len(filtered)} (threshold={threshold})")
        return filtered

    # =========================
    # fallback keyword filter
    # =========================
    def filter_by_keywords(self, papers, keywords):
        if not keywords:
            return papers

        keywords = [k.lower() for k in keywords]

        filtered = []
        for p in papers:
            text = (p.title + " " + p.abstract).lower()
            if any(k in text for k in keywords):
                filtered.append(p)

        logger.info(f"Keyword filter: {len(papers)} -> {len(filtered)}")
        return filtered

    # =========================
    # main pipeline
    # =========================
    def run(self):
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)

        if len(corpus) == 0:
            logger.error(f"No zotero papers found:\n{self.config.zotero}")
            return

        all_papers = []

        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()

            if len(papers) == 0:
                continue

            logger.info(f"Retrieved {len(papers)} {source} papers")

            all_papers.extend(papers)

        logger.info(f"Total {len(all_papers)} papers retrieved")

        # =========================
        # NEW: semantic filtering
        # =========================
        if self.topic_keywords:
            all_papers = self.semantic_filter(all_papers, threshold=0.45)

        # fallback keyword filter
        keywords = getattr(self.config.source.arxiv, "keywords", None)
        if keywords:
            all_papers = self.filter_by_keywords(all_papers, keywords)

        # =========================
        # rerank
        # =========================
        reranked_papers = []

        if len(all_papers) > 0:
            logger.info("Reranking papers...")

            reranked_papers = self.reranker.rerank(all_papers, corpus)
            reranked_papers = reranked_papers[:self.config.executor.max_paper_num]

            logger.info("Generating TLDR...")

            for p in tqdm(reranked_papers):
                p.generate_tldr(self.openai_client, self.config.llm)
                p.generate_affiliations(self.openai_client, self.config.llm)

        elif not self.config.executor.send_empty:
            logger.info("No papers found")
            return

        # =========================
        # send email
        # =========================
        logger.info("Sending email...")

        email_content = render_email(reranked_papers)
        send_email(self.config, email_content)

        logger.info("Email sent successfully")
