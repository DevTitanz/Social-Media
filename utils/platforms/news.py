import re
import logging
import requests
from typing import List, Dict, Any
from utils.platforms import PlatformAdapter, PlatformError
import config

logger = logging.getLogger(__name__)

INDIAN_NEWS_DOMAINS = [
    "indiatimes.com", "thehindu.com", "hindustantimes.com", "indiatoday.in",
    "ndtv.com", "indianexpress.com", "abpnews.com", "abplive.in",
    "zeenews.india.com", "aajtak.in", "republicworld.com", "news18.com",
    "cnbctv18.com", "financialexpress.in", "business-standard.com",
    "moneycontrol.com", "scroll.in", "thewire.in", "deccanchronicle.com",
    "jansatta.com", "dainikbhaskar.com", "dainikjagran.com", "navbharattimes.com"
]

GLOBAL_NEWS_DOMAINS = [
    "bbc.com", "bbc.co.uk", "cnn.com", "reuters.com", "apnews.com",
    "guardian.co.uk", "nytimes.com", "washingtonpost.com", "forbes.com",
    "bloomberg.com", "wsj.com", "economist.com"
]

ALL_NEWS_DOMAINS = INDIAN_NEWS_DOMAINS + GLOBAL_NEWS_DOMAINS


class NewsAdapter(PlatformAdapter):
    """Adapter for News API (newsapi.org)"""
    
    def __init__(self, api_key: str = None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.api_key = api_key or config.NEWS_API_KEY or None
        self.base_url = "https://newsapi.org/v2"
    
    def get_platform_name(self) -> str:
        return "news"
    
    def validate_credentials(self) -> bool:
        if not self.api_key:
            return False
        try:
            response = requests.get(
                f"{self.base_url}/top-headlines",
                params={"country": "in", "pageSize": 1},
                headers={"X-Api-Key": self.api_key},
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
    
    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def search_news(self, query: str, limit: int = 50, 
                   source_type: str = "indian") -> List[Dict[str, Any]]:
        """Search for news articles"""
        if not self.api_key:
            raise PlatformError(
                "News API key not configured.\n"
                "Please add NEWS_API_KEY to .env file.\n"
                "Get free key from: https://newsapi.org/register"
            )
        
        domains = self._get_domains(source_type)
        articles = []
        
        try:
            params = {
                "q": query,
                "language": "en",
                "sortBy": "relevance",
                "pageSize": min(limit, 100),
            }
            if domains:
                params["domains"] = domains
            
            response = requests.get(
                f"{self.base_url}/everything",
                params=params,
                headers={"X-Api-Key": self.api_key},
                timeout=30
            )
            
            if response.status_code == 401:
                raise PlatformError("Invalid News API key.")
            
            if response.status_code == 429:
                raise PlatformError("News API rate limit exceeded.")
            
            if response.status_code != 200:
                raise PlatformError(f"News API error: {response.status_code}")
            
            data = response.json()
            
            if "articles" in data:
                for article in data["articles"]:
                    title = self._clean_text(article.get("title", ""))
                    description = self._clean_text(article.get("description", ""))
                    
                    if title and title != "[Removed]":
                        combined_text = f"{title}. {description}" if description else title
                        articles.append({
                            "title": title,
                            "description": description,
                            "source": article.get("source", {}).get("name", "Unknown"),
                            "url": article.get("url", ""),
                            "published_at": article.get("publishedAt", "")[:10],
                            "text": combined_text
                        })
            
            logger.info(f"Fetched {len(articles)} news articles for: {query}")
            return articles
            
        except PlatformError:
            raise
        except Exception as e:
            logger.error(f"News API request failed: {e}")
            raise PlatformError(f"Failed to fetch news: {str(e)}")
    
    def _get_domains(self, source_type: str) -> str:
        if source_type == "indian":
            return ",".join(INDIAN_NEWS_DOMAINS)
        elif source_type == "global":
            return ",".join(GLOBAL_NEWS_DOMAINS)
        return ""
    
    def fetch_google_news_rss(self, query: str, limit: int = 50,
                               source_type: str = "all") -> List[Dict[str, Any]]:
        """Fetch real-time news articles from Google News RSS feed"""
        import xml.etree.ElementTree as ET
        import urllib.parse
        from datetime import datetime

        encoded_query = urllib.parse.quote(query)
        if source_type == "indian":
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
        else:
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

        articles = []
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                timeout=15
            )
            if resp.status_code != 200:
                return []

            root = ET.fromstring(resp.content)
            items = root.findall('./channel/item')

            for item in items[:limit]:
                raw_title = item.find('title').text if item.find('title') is not None else ""
                url_elem = item.find('link')
                article_url = url_elem.text if url_elem is not None else ""
                pub_elem = item.find('pubDate')
                pub_text = pub_elem.text if pub_elem is not None else ""
                
                # Format published date to YYYY-MM-DD
                pub_date = datetime.now().strftime("%Y-%m-%d")
                if pub_text:
                    try:
                        # RFC 822 / 2822 date parsing e.g. "Tue, 08 Sep 2026 12:00:00 GMT"
                        parsed_dt = datetime.strptime(pub_text[:16].strip(), "%a, %d %b %Y")
                        pub_date = parsed_dt.strftime("%Y-%m-%d")
                    except Exception:
                        pub_date = pub_text[:10]

                source_elem = item.find('source')
                source_name = source_elem.text if source_elem is not None else "Google News"

                # If source is in the title, e.g. "Headline - Source Name", extract source name
                clean_title = raw_title
                if " - " in raw_title:
                    parts = raw_title.rsplit(" - ", 1)
                    clean_title = parts[0].strip()
                    if source_name == "Google News" and len(parts) > 1:
                        source_name = parts[1].strip()

                clean_title = self._clean_text(clean_title)
                if clean_title:
                    articles.append({
                        "title": clean_title,
                        "description": "",
                        "source": source_name,
                        "url": article_url,
                        "published_at": pub_date,
                        "text": clean_title
                    })
            logger.info(f"Fetched {len(articles)} articles from Google News RSS for query: {query}")
        except Exception as e:
            logger.warning(f"Google News RSS fetch failed: {e}")

        return articles

    def fetch_topic_coverage(self, query: str, limit: int = 30,
                             source_type: str = "all") -> List[Dict[str, Any]]:
        """
        Unified multi-source coverage fetcher for trends & company topics.
        Combines NewsAPI and Google News RSS with deduplication.
        """
        seen_titles = set()
        combined = []

        # 1. Try NewsAPI if configured
        if self.api_key:
            try:
                newsapi_articles = self.search_news(query, limit=limit, source_type=source_type)
                for art in newsapi_articles:
                    norm = art['title'].lower()[:50]
                    if norm not in seen_titles:
                        seen_titles.add(norm)
                        combined.append(art)
            except Exception as e:
                logger.warning(f"NewsAPI error in topic coverage: {e}")

        # 2. Augment with Google News RSS to guarantee comprehensive real-time coverage
        if len(combined) < limit:
            needed = limit - len(combined)
            rss_articles = self.fetch_google_news_rss(query, limit=max(needed, 15), source_type=source_type)
            for art in rss_articles:
                norm = art['title'].lower()[:50]
                if norm not in seen_titles:
                    seen_titles.add(norm)
                    combined.append(art)
                if len(combined) >= limit:
                    break

        return combined[:limit]

    def fetch_posts(self, query: str, limit: int = 50, **kwargs) -> List[str]:
        source_type = kwargs.get("source_type", "indian")
        articles = self.fetch_topic_coverage(query, limit, source_type)
        return [a["text"] for a in articles]


def create_news_adapter() -> NewsAdapter:
    return NewsAdapter()

