import re
import logging
import requests
from typing import List, Dict, Any
from utils.platforms import PlatformAdapter, PlatformError
import config

logger = logging.getLogger(__name__)

class FacebookAdapter(PlatformAdapter):
    """Adapter for Facebook Pages API"""
    
    def __init__(self, access_token: str = None, **kwargs):
        super().__init__(**kwargs)
        self.access_token = access_token or config.FACEBOOK_ACCESS_TOKEN or None
        self.base_url = "https://graph.facebook.com/v18.0"
    
    def get_platform_name(self) -> str:
        return "facebook"
    
    def validate_credentials(self) -> bool:
        if not self.access_token:
            return False
        try:
            response = requests.get(
                f"{self.base_url}/me",
                params={"access_token": self.access_token},
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
    
    def fetch_posts(self, query: str, limit: int = 100, page_id: str = None, **kwargs) -> List[str]:
        """Fetch posts from a Facebook page"""
        
        if not self.access_token:
            raise PlatformError(
                "Facebook API access token required. "
                "Please set FACEBOOK_ACCESS_TOKEN in .env file. "
                "Note: Facebook API requires app review for most endpoints."
            )
        
        if not page_id:
            raise PlatformError("Facebook page ID required. Provide page_id parameter.")
        
        try:
            response = requests.get(
                f"{self.base_url_url}/{page_id}/posts",
                params={
                    "limit": min(limit, 100),
                    "fields": "message,story,description",
                    "access_token": self.access_token
                },
                timeout=30
            )
            
            if response.status_code != 200:
                error = response.json()
                raise PlatformError(f"Facebook API error: {error.get('error', {}).get('message', 'Unknown')}")
            
            data = response.json()
            posts = []
            
            if "data" in data:
                for post in data["data"]:
                    message = post.get("message") or post.get("story") or post.get("description")
                    if message:
                        cleaned = self._clean_post(message)
                        if cleaned:
                            posts.append(cleaned)
            
            logger.info(f"Fetched {len(posts)} Facebook posts")
            return posts
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Facebook API request failed: {e}")
            raise PlatformError(f"Failed to connect to Facebook API: {str(e)}")
    
    def fetch_page_posts(self, page_id: str, limit: int = 100) -> List[str]:
        """Fetch public posts from a Facebook page"""
        return self.fetch_posts("", limit, page_id=page_id)
    
    def search_posts(self, query: str, limit: int = 100) -> List[str]:
        """Search for public posts (requires additional permissions)"""
        
        if not self.access_token:
            raise PlatformError(
                "Facebook API access token required for search. "
                "Please set FACEBOOK_ACCESS_TOKEN in .env file."
            )
        
        try:
            response = requests.get(
                f"{self.base_url}/search",
                params={
                    "q": query,
                    "type": "post",
                    "limit": min(limit, 100),
                    "access_token": self.access_token
                },
                timeout=30
            )
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            posts = []
            
            if "data" in data:
                for post in data["data"]:
                    message = post.get("message", "")
                    if message:
                        cleaned = self._clean_post(message)
                        if cleaned:
                            posts.append(cleaned)
            
            return posts
            
        except Exception as e:
            logger.error(f"Facebook search failed: {e}")
            return []
    
    def _clean_post(self, text: str) -> str:
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text if len(text) > 2 else None


def create_facebook_adapter() -> FacebookAdapter:
    """Factory function to create Facebook adapter"""
    return FacebookAdapter()
