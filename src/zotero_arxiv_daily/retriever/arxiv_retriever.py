import time
import urllib.parse
import urllib.request
import urllib.error
import feedparser

from loguru import logger
from datetime import datetime, timedelta, timezone

from .base import BaseRetriever, register_retriever
from ..protocol import Paper


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    """
    高级 ArxivRetriever。

    功能：
    1. 使用 arXiv API 组合检索分类与关键词；
    2. 增加 User-Agent，降低被限流概率；
    3. 遇到 timeout / 429 / 临时网络错误时自动退避重试；
    4. 重试失败后返回空列表，不抛异常，不触发错误邮件；
    5. 失败时会打印日志，但主流程会表现为“没有新论文，不发邮件”。
    """

    def _get_config_list(self, key: str, default: list[str]) -> list[str]:
        """
        从 config.source.arxiv 中读取 list 配置。
        如果配置不存在或为空，则使用 default。
        """
        try:
            value = getattr(self.config.source.arxiv, key)
            value = list(value)
            return value if value else default
        except Exception:
            return default

    def _get_config_int(self, key: str, default: int) -> int:
        """
        从 config.source.arxiv 中读取 int 配置。
        如果配置不存在，则使用 default。
        """
        try:
            value = int(getattr(self.config.source.arxiv, key))
            return value
        except Exception:
            return default

    def _build_arxiv_api_url(
        self,
        categories: list[str],
        keywords: list[str],
        max_results: int,
    ) -> str:
        """
        构造 arXiv API URL。
        """
        cat_query = " OR ".join([f"cat:{c}" for c in categories])

        kw_parts = []
        for kw in keywords:
            kw_parts.append(f'ti:"{kw}"')
            kw_parts.append(f'abs:"{kw}"')

        kw_query = " OR ".join(kw_parts)

        full_query = f"({cat_query}) AND ({kw_query})"
        encoded_query = urllib.parse.quote(full_query)

        api_url = (
            "https://export.arxiv.org/api/query?"
            f"search_query={encoded_query}"
            "&sortBy=submittedDate"
            "&sortOrder=descending"
            f"&max_results={max_results}"
        )

        return api_url

    def _fetch_arxiv_feed_with_backoff(
        self,
        api_url: str,
        max_retries: int = 5,
        timeout: int = 60,
        base_sleep_seconds: int = 5,
    ):
        """
        请求 arXiv API。

        遇到以下情况会自动退避重试：
        1. HTTP 429：被限流；
        2. HTTP 5xx：服务端临时错误；
        3. Timeout / URL error / 网络异常；
        4. feed 解析异常但仍有内容时，尽量继续。

        如果最终仍失败，返回 None，不抛异常。
        """

        user_agent = "zotero-arxiv-daily/1.0 (mailto:your_email@example.com)"

        headers = {
            "User-Agent": user_agent,
            "Accept": "application/atom+xml,application/xml,text/xml;q=0.9,*/*;q=0.8",
        }

        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                if attempt == 1:
                    # 首次请求前也稍微等待，避免短时间多次启动任务导致 429
                    sleep_seconds = 3
                else:
                    # 指数退避：5, 10, 20, 40, 60...
                    sleep_seconds = min(base_sleep_seconds * (2 ** (attempt - 2)), 60)

                logger.info(
                    f"Waiting {sleep_seconds}s before arXiv API request "
                    f"({attempt}/{max_retries})..."
                )
                time.sleep(sleep_seconds)

                request = urllib.request.Request(
                    api_url,
                    headers=headers,
                    method="GET",
                )

                logger.info(f"Sending arXiv API request ({attempt}/{max_retries})")

                with urllib.request.urlopen(request, timeout=timeout) as response:
                    content = response.read()

                feed = feedparser.parse(content)

                if getattr(feed, "bozo", False):
                    logger.warning(
                        f"arXiv feed parse warning: "
                        f"{getattr(feed, 'bozo_exception', None)}"
                    )

                return feed

            except urllib.error.HTTPError as e:
                last_error = e

                if e.code == 429:
                    retry_after = e.headers.get("Retry-After")

                    if retry_after:
                        try:
                            sleep_seconds = int(retry_after)
                        except ValueError:
                            sleep_seconds = 60
                    else:
                        sleep_seconds = min(30 * attempt, 180)

                    logger.warning(
                        f"arXiv API returned HTTP 429 rate limit. "
                        f"Sleep {sleep_seconds}s before next retry "
                        f"({attempt}/{max_retries})."
                    )
                    time.sleep(sleep_seconds)
                    continue

                if 500 <= e.code < 600:
                    logger.warning(
                        f"arXiv API temporary server error HTTP {e.code}: {e.reason}. "
                        f"Will retry if attempts remain."
                    )
                    continue

                logger.error(
                    f"arXiv API non-retryable HTTP error: "
                    f"HTTP {e.code}, reason={e.reason}"
                )
                return None

            except urllib.error.URLError as e:
                last_error = e
                logger.warning(
                    f"arXiv API URL error: {repr(e)} "
                    f"({attempt}/{max_retries})"
                )
                continue

            except TimeoutError as e:
                last_error = e
                logger.warning(
                    f"arXiv API timeout: {repr(e)} "
                    f"({attempt}/{max_retries})"
                )
                continue

            except Exception as e:
                last_error = e
                logger.warning(
                    f"arXiv API request failed: {repr(e)} "
                    f"({attempt}/{max_retries})"
                )
                continue

        logger.error(
            f"Failed to fetch from arXiv API after {max_retries} retries. "
            f"Last error: {repr(last_error)}"
        )

        return None

    def _retrieve_raw_papers(self) -> list:
        """
        从 arXiv API 检索原始论文条目。
        """
        categories = self._get_config_list(
            key="category",
            default=["cs.RO", "cs.CV", "cs.AI", "cs.LG"],
        )

        keywords = self._get_config_list(
            key="keywords",
            default=[
                "VLA",
                "Vision-Language-Action",
                "Vision Language Action",
                "Embodied AI",
                "Embodied Agent",
            ],
        )

        max_results = self._get_config_int("max_results", 20)
        days = self._get_config_int("days", 3)
        max_retries = self._get_config_int("max_retries", 5)
        timeout = self._get_config_int("timeout", 60)

        logger.info(
            f"Using advanced arXiv API with categories: {categories} "
            f"and key terms: {keywords}"
        )

        api_url = self._build_arxiv_api_url(
            categories=categories,
            keywords=keywords,
            max_results=max_results,
        )

        logger.info(f"Requesting arXiv API: {api_url}")

        feed = self._fetch_arxiv_feed_with_backoff(
            api_url=api_url,
            max_retries=max_retries,
            timeout=timeout,
            base_sleep_seconds=5,
        )

        if feed is None:
            logger.warning(
                "arXiv retrieval failed after retries. "
                "Return empty list and skip email in the main pipeline."
            )
            return []

        if not getattr(feed, "entries", None):
            logger.info("No entries returned from arXiv API.")
            return []

        raw_papers = []
        now = datetime.now(timezone.utc)
        time_threshold = now - timedelta(days=days)

        for entry in feed.entries:
            try:
                published_time = datetime.strptime(
                    entry.published,
                    "%Y-%m-%dT%H:%M:%SZ",
                ).replace(tzinfo=timezone.utc)
            except Exception:
                logger.warning(
                    f"Failed to parse published time for entry: "
                    f"{getattr(entry, 'title', '')}"
                )
                published_time = now

            if published_time < time_threshold:
                continue

            raw_papers.append(entry)

        logger.info(
            f"Filtered {len(raw_papers)} fresh raw candidate entries from arXiv API."
        )

        return raw_papers

    def convert_to_paper(self, raw_paper) -> Paper | None:
        """
        将 feed entry 转化为标准 Paper 对象。
        """
        try:
            title = raw_paper.title.replace("\n", " ").strip()
            abstract = raw_paper.summary.replace("\n", " ").strip()
            url = raw_paper.link

            pdf_url = (
                raw_paper.link.replace("/abs/", "/pdf/")
                if "/abs/" in raw_paper.link
                else None
            )

            authors = (
                [a.name for a in raw_paper.authors]
                if "authors" in raw_paper
                else []
            )

            keywords = self._get_config_list(
                key="keywords",
                default=[
                    "VLA",
                    "Vision-Language-Action",
                    "Vision Language Action",
                    "Embodied AI",
                    "Embodied Agent",
                ],
            )

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
                full_text=abstract,
            )

        except Exception as e:
            logger.warning(f"Error parse raw paper entry: {repr(e)}")
            return None
