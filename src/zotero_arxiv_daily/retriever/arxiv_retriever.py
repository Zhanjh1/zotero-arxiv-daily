import urllib.parse
import urllib.request
import feedparser
from loguru import logger
from datetime import datetime, timedelta, timezone
from .base import BaseRetriever
from ..protocol import Paper

class ArxivRetriever(BaseRetriever):
    """
    完美兼容原版工厂模式的 ArxivRetriever。
    内部采用高级 arXiv API 进行关键词 + 分类复合检索，从源头阻断噪音。
    """
    def _retrieve_raw_papers(self) -> list:
        # 1. 动态安全提取配置参数
        try:
            categories = list(self.config.source.arxiv.category)
        except Exception:
            categories = ["cs.RO", "cs.CV", "cs.AI", "cs.LG"]
            
        try:
            keywords = list(self.config.source.arxiv.keywords)
        except Exception:
            keywords = ["VLA", "Vision-Language-Action", "Vision Language Action", "Embodied AI", "Embodied Agent"]

        logger.info(f"Using advanced arXiv API with categories: {categories} and key terms: {keywords}")

        # 2. 构造 arXiv API 检索表达式 (精确控制：类别交集 + 关键词并集)
        cat_query = " OR ".join([f"cat:{c}" for c in categories])
        kw_query = " OR ".join([f"ti:\"{kw}\" OR abs:\"{kw}\"" for kw in keywords])
        
        full_query = f"({cat_query}) AND ({kw_query})"
        encoded_query = urllib.parse.quote(full_query)
        
        # arXiv 标准 API 地址
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

        # 3. 时间窗口过滤
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

            title = entry.title.replace('\n', ' ').strip()
            abstract = entry.summary.replace('\n', ' ').strip()
            url = entry.link
            pdf_url = entry.link.replace('/abs/', '/pdf/') if '/abs/' in entry.link else None
            authors = [a.name for a in entry.authors] if 'authors' in entry else []

            # 强校验：确保真货
            search_str = f"{title} {abstract}".lower()
            if not any(k.lower() in search_str for k in keywords):
                continue

            paper = Paper(
                source="arxiv",
                title=title,
                authors=authors,
                abstract=abstract,
                url=url,
                pdf_url=pdf_url,
                full_text=abstract  # 映射为初阶段正文以供提取机构
            )
            raw_papers.append(paper)

        logger.info(f"Successfully processed {len(raw_papers)} fresh VLA/Embodied papers from arXiv.")
        return raw_papers
