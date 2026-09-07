import re
import logging
import requests
from typing import List, Dict, Any
from utils.platforms import PlatformAdapter, PlatformError, RateLimitError
import config

logger = logging.getLogger(__name__)

class RedditAdapter(PlatformAdapter):
    """Adapter for Reddit API"""
    
    def __init__(self, client_id: str = None, client_secret: str = None,
                 user_agent: str = "SentimentAnalysisBot/1.0", **kwargs):
        super().__init__(client_id, client_secret, **kwargs)
        self.client_id = client_id or config.REDDIT_CLIENT_ID or None
        self.client_secret = client_secret or config.REDDIT_CLIENT_SECRET or None
        self.user_agent = user_agent
        self.base_url = "https://www.reddit.com"
        self.access_token = None
    
    def get_platform_name(self) -> str:
        return "reddit"
    
    def validate_credentials(self) -> bool:
        if self.client_id and self.client_secret:
            try:
                self._get_access_token()
                return True
            except:
                return False
        return False  # Reddit now requires auth
    
    def _get_access_token(self) -> str:
        if self.access_token:
            return self.access_token
        
        if not self.client_id or not self.client_secret:
            return None
        
        try:
            auth = requests.auth.HTTPBasicAuth(self.client_id, self.client_secret)
            data = {"grant_type": "client_credentials"}
            response = requests.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=auth,
                data=data,
                headers={"User-Agent": self.user_agent},
                timeout=10
            )
            if response.status_code == 200:
                self.access_token = response.json().get("access_token")
                return self.access_token
        except Exception as e:
            logger.warning(f"Failed to get Reddit access token: {e}")
        
        return None
    
    def fetch_posts(self, query: str, limit: int = 100, subreddit: str = None, 
                    sort: str = "hot", **kwargs) -> List[str]:
        """Fetch posts from Reddit"""
        
        if not self.client_id or not self.client_secret:
            raise PlatformError(
                "Reddit API requires authentication.\n"
                "Please add your Reddit API credentials to .env file:\n"
                "REDDIT_CLIENT_ID=your_client_id\n"
                "REDDIT_CLIENT_SECRET=your_client_secret\n\n"
                "Get credentials from: https://www.reddit.com/prefs/apps"
            )
        
        posts = []
        
        if subreddit:
            url = f"{self.base_url}/r/{subreddit}/{sort}.json"
        else:
            url = f"{self.base_url}/search.json"
        
        headers = {"User-Agent": self.user_agent}
        
        access_token = self._get_access_token()
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        
        try:
            params = {
                "limit": min(limit, 100),
                "q": query,
                "sort": sort if not subreddit else "relevance"
            }
            
            if subreddit:
                del params["q"]
            
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 401:
                raise PlatformError("Reddit authentication failed. Check your API credentials.")
            
            if response.status_code == 429:
                raise RateLimitError("Reddit API rate limit exceeded. Please try again later.")
            
            if response.status_code == 403:
                raise PlatformError("Reddit access forbidden. Your API may lack required permissions.")
            
            if response.status_code != 200:
                raise PlatformError(f"Reddit API error: HTTP {response.status_code}")
            
            data = response.json()
            
            if "data" in data:
                children = data["data"].get("children", [])
                
                for child in children:
                    post = child.get("data", {})
                    
                    title = post.get("title", "")
                    selftext = post.get("selftext", "")
                    text = f"{title}. {selftext}" if selftext else title
                    
                    cleaned_text = self._clean_post(text)
                    if cleaned_text:
                        posts.append(cleaned_text)
            
            logger.info(f"Fetched {len(posts)} Reddit posts for query: {query}")
            return posts
            
        except PlatformError:
            raise
        except requests.exceptions.Timeout:
            raise PlatformError("Reddit request timed out. Please try again.")
        except requests.exceptions.ConnectionError as e:
            raise PlatformError(f"Cannot connect to Reddit. Check your internet connection.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Reddit API request failed: {e}")
            raise PlatformError(f"Reddit API request failed: {str(e)}")
    
    def _clean_post(self, text: str) -> str:
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\[deleted\]', '', text)
        text = re.sub(r'\[removed\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text if len(text) > 2 else None
    
    def fetch_subreddit_posts(self, subreddit: str, limit: int = 100, 
                              sort: str = "hot") -> List[str]:
        """Fetch posts from a specific subreddit"""
        return self.fetch_posts("", limit, subreddit=subreddit, sort=sort)


def create_reddit_adapter() -> RedditAdapter:
    """Factory function to create Reddit adapter"""
    return RedditAdapter()
