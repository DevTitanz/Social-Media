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
                "sortBy": "publishedAt",
                "pageSize": min(limit, 100),
                "domains": domains
            }
            
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
        return ",".join(ALL_NEWS_DOMAINS)
    
    def fetch_posts(self, query: str, limit: int = 50, **kwargs) -> List[str]:
        source_type = kwargs.get("source_type", "indian")
        articles = self.search_news(query, limit, source_type)
        return [a["text"] for a in articles]


def create_news_adapter() -> NewsAdapter:
    return NewsAdapter()
