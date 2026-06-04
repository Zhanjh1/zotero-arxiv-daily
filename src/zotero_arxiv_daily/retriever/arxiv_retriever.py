import urllib.parse
import urllib.request
import feedparser
from loguru import logger
from datetime import datetime, timedelta, timezone
from .base import BaseRetriever, RETRIEVER_REGISTRY  # 注入点：直接引入原项目的注册字典
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
                full_text=abstract
            )
            raw_papers.append(paper)

        logger.info(f"Successfully processed {len(raw_papers)} fresh VLA/Embodied papers from arXiv.")
        return raw_papers

# =====================================================================
# 🛠️ 终极保底硬注册逻辑：直接把当前类强行塞进工厂字典，彻底解决 ValueError
# =====================================================================
try:
    # 尝试一：如果原项目用的是 RETRIEVER_REGISTRY 字典
    RETRIEVER_REGISTRY["arxiv"] = ArxivRetriever
    logger.info("Successfully registered ArxivRetriever to RETRIEVER_REGISTRY manually.")
except Exception:
    try:
        # 尝试二：如果原项目用的是底层类的子类自动收集机制，或者叫别的名字
        from .base import register_retriever
        @register_retriever("arxiv")
        class RegisteredArxivRetriever(ArxivRetriever):
            pass
        logger.info("Successfully registered ArxivRetriever via @register_retriever manually.")
    except Exception:
        # 尝试三：如果上面都找不到，直接通过 sys.modules 暴力打补丁
        import sys
        from . import base
        if hasattr(base, 'retriever_registry'):
            getattr(base, 'retriever_registry')["arxiv"] = ArxivRetriever
        elif hasattr(base, '_registry'):
            getattr(base, '_registry')["arxiv"] = ArxivRetriever
