from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PlatformAdapter(ABC):
    """Base class for all social media platform adapters"""
    
    def __init__(self, api_key: str = None, api_secret: str = None, 
                 access_token: str = None, **kwargs):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.kwargs = kwargs
    
    @abstractmethod
    def fetch_posts(self, query: str, limit: int = 100, **kwargs) -> List[str]:
        """Fetch posts/tweets/comments from the platform"""
        pass
    
    @abstractmethod
    def validate_credentials(self) -> bool:
        """Validate API credentials"""
        pass
    
    @abstractmethod
    def get_platform_name(self) -> str:
        """Return platform name"""
        pass
    
    def get_metadata(self, post: str) -> Dict[str, Any]:
        """Extract metadata from a post (optional override)"""
        return {}
    
    def format_response(self, posts: List[str], metadata: List[Dict] = None) -> Dict:
        """Format posts into standardized response"""
        if metadata is None:
            metadata = [{} for _ in posts]
        
        return {
            'platform': self.get_platform_name(),
            'posts': posts,
            'metadata': metadata,
            'count': len(posts)
        }


class InvalidCredentialsError(Exception):
    """Raised when platform credentials are invalid"""
    pass

class RateLimitError(Exception):
    """Raised when API rate limit is exceeded"""
    pass

class PlatformError(Exception):
    """General platform error"""
    pass
