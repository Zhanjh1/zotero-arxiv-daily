import urllib.parse
import urllib.request
import feedparser
from loguru import logger
from datetime import datetime, timedelta, timezone
from .base import BaseRetriever, register_retriever
from ..protocol import Paper

# 1. 使用原作者的装饰器进行精准注册
@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    """
    完美契合原版生命周期的高级 ArxivRetriever。
    利用标准 API 在源头组合检索分类与关键词，阻断无关噪音论文。
    """

    def _retrieve_raw_papers(self) -> list:
        # 动态提取配置参数，若不存在则使用保底高价值具身智能词表
        try:
            categories = list(self.config.source.arxiv.category)
        except Exception:
            categories = ["cs.RO", "cs.CV", "cs.AI", "cs.LG"]
            
        try:
            keywords = list(self.config.source.arxiv.keywords)
        except Exception:
            keywords = ["VLA", "Vision-Language-Action", "Vision Language Action", "Embodied AI", "Embodied Agent"]

        logger.info(f"Using advanced arXiv API with categories: {categories} and key terms: {keywords}")

        # 构造高级组合查询表达式
        cat_query = " OR ".join([f"cat:{c}" for c in categories])
        kw_query = " OR ".join([f"ti:\"{kw}\" OR abs:\"{kw}\"" for kw in keywords])
        full_query = f"({cat_query}) AND ({kw_query})"
        encoded_query = urllib.parse.quote(full_query)
        
        # 请求标准 arXiv export API，单次取 100 篇最新内容
        api_url = f"https://export.arxiv.org/api/query?search_query={encoded_query}&sortBy=submittedDate&sortOrder=descending&max_results=100"
        
        logger.info(f"Requesting arXiv API: {api_url}")
        
        try:
            response = urllib.request.urlopen(api_url, timeout=15)
            feed = feedparser.parse(response.read())
        except Exception as e:
            logger.error(f"Failed to fetch from arXiv API: {e}")
            return []

        if not feed.entries:
            logger.info("No entries returned from arXiv API.")
            return []

        # 时间窗口过滤（过去 3 天内更新）
        raw_papers = []
        now = datetime.now(timezone.utc)
        time_threshold = now - timedelta(days=3)

        for entry in feed.entries:
            try:
                published_time = datetime.strptime(entry.published, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
            except ValueError:
                published_time = time_threshold + timedelta(days=1)

            if published_time < time_threshold:
                continue
                
            # 这里的 entry 就是原作者设计里的 RawPaperItem
            raw_papers.append(entry)

        logger.info(f"Filtered {len(raw_papers)} fresh raw candidate entries from arXiv API.")
        return raw_papers

    def convert_to_paper(self, raw_paper) -> Paper | None:
        """
        实现父类要求的抽象方法，负责将原始的 feed entry 转化为标准 Paper 对象
        """
        try:
            title = raw_paper.title.replace('\n', ' ').strip()
            abstract = raw_paper.summary.replace('\n', ' ').strip()
            url = raw_paper.link
            pdf_url = raw_paper.link.replace('/abs/', '/pdf/') if '/abs/' in raw_paper.link else None
            authors = [a.name for a in raw_paper.authors] if 'authors' in raw_paper else []

            # 提取关键词列表用于最终确认
            try:
                keywords = list(self.config.source.arxiv.keywords)
            except Exception:
                keywords = ["VLA", "Vision-Language-Action", "Vision Language Action", "Embodied AI", "Embodied Agent"]

            # 二次强校验过滤，不包含具身智能核心关键词的直接丢弃
            search_str = f"{title} {abstract}".lower()
            if not any(k.lower() in search_str for k in keywords):
                return None

            return Paper(
                source="arxiv",
                title=title,
                authors=authors,
                abstract=abstract,
                url=url,
                pdf_url=pdf_url,
                full_text=abstract  # 先用摘要垫底，供后续提取机构使用
            )
        except Exception as e:
            logger.warning(f"Error parse raw paper entry: {e}")
            return None
