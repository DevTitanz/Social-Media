import re
import logging
import requests
from typing import List, Optional
from utils.platforms import PlatformAdapter, PlatformError
import config

logger = logging.getLogger(__name__)


class InstagramAdapter(PlatformAdapter):
    """
    Instagram adapter — RapidAPI 'instagram-best-experience' only.
    Only supports Instagram post/reel/tv URLs.
    """

    def __init__(self, access_token: str = None, **kwargs):
        super().__init__(**kwargs)

        self.api_key = getattr(config, 'RAPIDAPI_KEY', '').strip()
        self.rapidapi_base = "https://instagram-best-experience.p.rapidapi.com"
        self.rapidapi_headers = {
            "Content-Type": "application/json",
            "x-rapidapi-host": "instagram-best-experience.p.rapidapi.com",
            "x-rapidapi-key": self.api_key,
        }
        self.graph_base = "https://graph.instagram.com"

    # ------------------------------------------------------------------
    # PlatformAdapter interface
    # ------------------------------------------------------------------

    def get_platform_name(self) -> str:
        return "instagram"

    def validate_credentials(self) -> bool:
        if self.api_key:
            return True
        logger.error(
            "[Instagram] No credentials found. "
            "Set RAPIDAPI_KEY in your .env file."
        )
        return False

    def fetch_posts(self, query: str, limit: int = 50, **kwargs) -> List[str]:
        if not query or not query.strip():
            raise PlatformError("An Instagram post URL or reel URL is required.")

        # Strip query string (?utm_source=...) before any processing
        query = query.strip().split('?')[0].strip()
        logger.info(f"[Instagram] DEBUG query after strip: {repr(query)}")
        if self._is_instagram_url(query):
            return self._fetch_comments_by_url(query, limit)

        raise PlatformError(
            "Only Instagram post/reel URLs are supported.\n"
            "Accepted formats:\n"
            "  https://www.instagram.com/p/ABC123/\n"
            "  https://www.instagram.com/reel/ABC123/\n"
            "  https://www.instagram.com/reels/ABC123/\n"
            "  https://instagram.com/tv/ABC123/"
        )

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    _INSTAGRAM_URL_RE = re.compile(
        r'(?:https?://)?(?:www\.)?instagram\.com/(?:p|reel|reels|tv)/([A-Za-z0-9_-]+)',
        re.IGNORECASE
    )

    def _is_instagram_url(self, text: str) -> bool:
        clean = text.split('?')[0].strip()
        return bool(self._INSTAGRAM_URL_RE.search(clean))

    def extract_shortcode(self, url: str) -> str:
        clean = url.split('?')[0].strip()
        match = self._INSTAGRAM_URL_RE.search(clean)
        if not match:
            raise PlatformError(
                "Invalid Instagram URL. Accepted formats:\n"
                "  https://www.instagram.com/p/ABC123/\n"
                "  https://www.instagram.com/reel/ABC123/\n"
                "  https://www.instagram.com/reels/ABC123/\n"
                "  https://instagram.com/tv/ABC123/"
            )
        return match.group(1)

    # ------------------------------------------------------------------
    # RapidAPI backend — comment scraping
    # ------------------------------------------------------------------

    def get_media_id(self, shortcode: str) -> str:
        """Resolve a shortcode → numeric media ID via RapidAPI /post."""
        if not self.api_key:
            raise PlatformError(
                "RAPIDAPI_KEY is not set. Please add it to your .env file."
            )
        try:
            resp = requests.get(
                f"{self.rapidapi_base}/post",
                headers=self.rapidapi_headers,
                params={"shortcode": shortcode},
                timeout=30,
            )

            if resp.status_code == 401:
                raise PlatformError(
                    "RapidAPI key invalid or inactive. "
                    "Check RAPIDAPI_KEY in your .env file."
                )
            if resp.status_code == 403:
                raise PlatformError(
                    "RapidAPI plan does not include this endpoint. "
                    "Verify your subscription at rapidapi.com."
                )
            if resp.status_code == 429:
                raise PlatformError(
                    "RapidAPI rate limit reached. Please wait and try again."
                )
            if resp.status_code != 200:
                raise PlatformError(
                    f"RapidAPI /post returned HTTP {resp.status_code}: "
                    f"{resp.text[:200]}"
                )

            data = resp.json()
            media_id = (
                data.get("id")
                or data.get("pk")
                or (data.get("data") or {}).get("id")
                or (data.get("data") or {}).get("pk")
            )
            if not media_id:
                raise PlatformError(
                    f"Media ID missing from RapidAPI response. "
                    f"Keys returned: {list(data.keys())}"
                )
            return str(media_id)

        except requests.exceptions.ConnectionError:
            raise PlatformError(
                "Cannot reach RapidAPI. Check your internet connection."
            )
        except requests.exceptions.Timeout:
            raise PlatformError("RapidAPI /post timed out. Please try again.")
        except PlatformError:
            raise
        except requests.exceptions.RequestException as e:
            raise PlatformError(f"Network error resolving media ID: {e}")

    def _fetch_comments_by_url(self, url: str, limit: int) -> List[str]:
        """Fetch comments for a post / reel URL via RapidAPI."""
        if not self.api_key:
            raise PlatformError(
                "RAPIDAPI_KEY is not set. Please add it to your .env file."
            )

        shortcode = self.extract_shortcode(url)
        logger.info(f"[Instagram/RapidAPI] Shortcode: {shortcode}")

        media_id = self.get_media_id(shortcode)
        logger.info(f"[Instagram/RapidAPI] Media ID: {media_id}")

        try:
            resp = requests.get(
                f"{self.rapidapi_base}/comments",
                headers=self.rapidapi_headers,
                params={"id": media_id},
                timeout=30,
            )

            if resp.status_code == 401:
                raise PlatformError(
                    "RapidAPI key rejected on /comments. "
                    "Verify RAPIDAPI_KEY in your .env file."
                )
            if resp.status_code == 429:
                raise PlatformError(
                    "RapidAPI rate limit reached. Please wait and try again."
                )
            if resp.status_code != 200:
                raise PlatformError(
                    f"RapidAPI /comments returned HTTP {resp.status_code}: "
                    f"{resp.text[:200]}"
                )

            data = resp.json()
            items = (
                data.get("comments")
                or data.get("data")
                or data.get("items")
                or []
            )

            comments = []
            for item in items:
                if len(comments) >= limit:
                    break
                text = (item.get("text") or item.get("content") or "").strip()
                if text:
                    comments.append(text)

            if not comments:
                raise PlatformError(
                    "No comments found. "
                    "The post may be private or comments may be disabled."
                )

            logger.info(f"[Instagram/RapidAPI] Fetched {len(comments)} comments.")
            return comments

        except requests.exceptions.Timeout:
            raise PlatformError("RapidAPI /comments timed out. Please try again.")
        except PlatformError:
            raise
        except requests.exceptions.RequestException as e:
            raise PlatformError(f"Failed to fetch comments: {e}")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _clean_caption(self, text: str) -> Optional[str]:
        """Remove hashtags, mentions, and URLs from caption text."""
        if not text:
            return None
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'@[\w.]+', '', text)
        text = re.sub(r'http\S+', '', text)
        text = text.strip()
        return text or None


def create_instagram_adapter() -> InstagramAdapter:
    """Factory function — returns a ready-to-use InstagramAdapter."""
    return InstagramAdapter()