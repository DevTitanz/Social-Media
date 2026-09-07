import os
import re
import logging
import requests
from typing import List, Dict, Any, Optional
from utils.platforms import PlatformAdapter, InvalidCredentialsError, RateLimitError, PlatformError

import config

logger = logging.getLogger(__name__)


class TwitterAdapter(PlatformAdapter):
    """
    Twitter/X adapter with dual-backend support:
      - Primary  : RapidAPI 'twitter241' (comment scraping, search, no OAuth needed)
      - Fallback : Twitter API v2 (official Bearer token, search/recent tweets)

    Routing logic in fetch_posts():
      - Tweet URL or numeric ID  → _fetch_comments()  [RapidAPI]
      - Keyword search           → _fetch_search()     [RapidAPI if key present,
                                                         else Twitter API v2]
    """

    def __init__(self, api_key: str = None, api_secret: str = None,
                 bearer_token: str = None, **kwargs):
        super().__init__(api_key, api_secret, **kwargs)

        # RapidAPI backend (v1)
        self.api_key = config.TWITTER_BEARER_TOKEN or os.getenv("TWITTER_BEARER_TOKEN")
        self.rapidapi_base = "https://twitter241.p.rapidapi.com"
        self.rapidapi_headers = {
            "x-rapidapi-host": "twitter241.p.rapidapi.com",
            "x-rapidapi-key": self.api_key or "",
        }

        # Official Twitter API v2 backend (v2)
        self.bearer_token = bearer_token or config.TWITTER_BEARER_TOKEN or os.getenv("TWITTER_BEARER_TOKEN")
        self.v2_base = "https://api.twitter.com/2"

    # ------------------------------------------------------------------
    # PlatformAdapter interface
    # ------------------------------------------------------------------

    def get_platform_name(self) -> str:
        return "twitter"

    def validate_credentials(self) -> bool:
        """
        Returns True if at least one backend is usable.
        RapidAPI key check is instant (no network call).
        Falls back to a live Twitter API v2 probe.
        """
        if self.api_key:
            return True
        if self.bearer_token:
            try:
                response = requests.get(
                    f"{self.v2_base}/tweets/counts/recent",
                    headers=self._v2_headers(),
                    timeout=10,
                )
                return response.status_code == 200
            except Exception as e:
                logger.error(f"Twitter v2 credentials validation failed: {e}")
                return False
        return False

    def fetch_posts(self, query: str, limit: int = 100, **kwargs) -> List[str]:
        """
        Dispatch to the appropriate fetch strategy:
          - Tweet URL / numeric ID → comment scraping via RapidAPI
          - Keyword query          → search via RapidAPI (preferred) or Twitter v2
        """
        if not self.api_key and not self.bearer_token:
            raise InvalidCredentialsError(
                "Twitter API key not configured. "
                "Please set TWITTER_BEARER_TOKEN in your .env file."
            )

        if self._is_tweet_url_or_id(query):
            return self._fetch_comments(query, limit)

        # Prefer RapidAPI for search; fall back to official v2
        if self.api_key:
            return self._fetch_search_rapidapi(query, limit)
        return self._fetch_search_v2(query, limit)

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _is_tweet_url_or_id(self, query: str) -> bool:
        query = query.strip()
        if query.isdigit():
            return True
        return bool(re.search(r"(?:twitter\.com|x\.com)/[^/]+/status/(\d+)", query))

    def extract_tweet_id(self, query: str) -> str:
        query = query.strip()
        if query.isdigit():
            return query
        match = re.search(r"(?:twitter\.com|x\.com)/[^/]+/status/(\d+)", query)
        if match:
            return match.group(1)
        raise PlatformError(
            "Invalid Tweet input. Please provide a valid Twitter/X post URL or Tweet ID."
        )

    # ------------------------------------------------------------------
    # RapidAPI backend
    # ------------------------------------------------------------------

    def _fetch_search_rapidapi(self, query: str, limit: int) -> List[str]:
        """Keyword search via RapidAPI twitter241 /search endpoint."""
        max_results = min(limit, 100)
        tweets = []

        try:
            response = requests.get(
                f"{self.rapidapi_base}/search",
                headers=self.rapidapi_headers,
                params={"query": query, "count": max_results, "type": "Latest"},
                timeout=30,
            )

            if response.status_code == 429:
                raise RateLimitError("Rate limit exceeded for RapidAPI Twitter service.")
            if response.status_code != 200:
                raise PlatformError(
                    f"Twitter API error: {response.status_code} — {response.text[:150]}"
                )

            data = response.json()
            tweets = self._parse_timeline_instructions(data, max_results)

            # Flat fallback
            if not tweets:
                for item in (data.get("data") or data.get("tweets") or []):
                    text = (
                        item.get("full_text")
                        or item.get("text")
                        or item.get("legacy", {}).get("full_text")
                    )
                    if text:
                        cleaned = self._clean_tweet(text)
                        if cleaned:
                            tweets.append(cleaned)
                            if len(tweets) >= max_results:
                                break

            logger.info(f"[Twitter/RapidAPI] Fetched {len(tweets)} tweets for '{query}'.")
            return tweets

        except requests.exceptions.RequestException as e:
            logger.error(f"Twitter RapidAPI search failed: {e}")
            raise PlatformError(f"Failed to connect to Twitter API: {str(e)}")

    def _fetch_comments(self, query: str, limit: int) -> List[str]:
        """Fetch replies/comments for a specific tweet via RapidAPI twitter241 /comments."""
        if not self.api_key:
            raise InvalidCredentialsError(
                "RapidAPI key required for comment fetching. "
                "Please set TWITTER_BEARER_TOKEN (RapidAPI key) in your .env file."
            )

        tweet_id = self.extract_tweet_id(query)
        logger.info(f"[Twitter/RapidAPI] Fetching comments for Tweet ID: {tweet_id}")
        max_results = min(limit, 100)
        comments = []

        try:
            response = requests.get(
                f"{self.rapidapi_base}/comments",
                headers=self.rapidapi_headers,
                params={"pid": tweet_id, "count": max_results},
                timeout=30,
            )

            if response.status_code == 429:
                raise RateLimitError("Rate limit exceeded for RapidAPI Twitter service.")
            if response.status_code != 200:
                raise PlatformError(
                    f"Twitter API error: {response.status_code} — {response.text[:150]}"
                )

            data = response.json()
            comments = self._parse_timeline_instructions(data, max_results)

            # Flat fallback
            if not comments:
                for item in (data.get("comments") or data.get("data") or []):
                    text = item.get("text") or item.get("full_text")
                    if text:
                        cleaned = self._clean_tweet(text)
                        if cleaned:
                            comments.append(cleaned)
                            if len(comments) >= max_results:
                                break

            logger.info(f"[Twitter/RapidAPI] Fetched {len(comments)} comments for tweet {tweet_id}.")
            return comments

        except requests.exceptions.RequestException as e:
            logger.error(f"Twitter RapidAPI comments request failed: {e}")
            raise PlatformError(f"Failed to connect to Twitter API: {str(e)}")

    def _parse_timeline_instructions(self, data: dict, limit: int) -> List[str]:
        """
        Extract tweet texts from twitter241's nested timeline instructions structure.
        Shared by both search and comments responses.
        """
        texts = []
        instructions = []

        if "result" in data:
            res = data["result"]
            tl = res.get("timeline") or {}
            instructions = tl.get("instructions") or res.get("instructions", [])
        elif "timeline" in data:
            instructions = data["timeline"].get("instructions", [])
        elif "instructions" in data:
            instructions = data["instructions"]

        for instruction in instructions:
            if instruction.get("type") == "TimelineAddEntries":
                for entry in instruction.get("entries", []):
                    text = self._extract_text_from_entry(entry)
                    if text:
                        texts.append(text)
                        if len(texts) >= limit:
                            return texts

        return texts

    def _extract_text_from_entry(self, entry: dict) -> Optional[str]:
        """Safely extract tweet text from a timeline entry (single item or threaded)."""
        try:
            content = entry.get("content", {})

            # Single-item entry
            item = content.get("itemContent", {})
            if item:
                tweet_res = item.get("tweet_results", {}).get("result", {})
                if tweet_res.get("__typename") != "TweetTombstone":
                    text = tweet_res.get("legacy", {}).get("full_text")
                    if text:
                        return self._clean_tweet(text)

            # Multi-item / threaded entry
            for sub in content.get("items", []):
                sub_item = sub.get("item", {}).get("itemContent", {})
                tweet_res = sub_item.get("tweet_results", {}).get("result", {})
                if tweet_res.get("__typename") == "TweetTombstone":
                    continue
                text = tweet_res.get("legacy", {}).get("full_text")
                if text:
                    return self._clean_tweet(text)

        except Exception as e:
            logger.debug(f"Entry parse error: {e}")

        return None

    # ------------------------------------------------------------------
    # Official Twitter API v2 backend
    # ------------------------------------------------------------------

    def _v2_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json",
        }

    def _fetch_search_v2(self, query: str, limit: int) -> List[str]:
        """Keyword search via official Twitter API v2 /tweets/search/recent."""
        if not self.bearer_token:
            raise InvalidCredentialsError(
                "Twitter Bearer token not configured. "
                "Please set TWITTER_BEARER_TOKEN in your .env file."
            )

        max_results = min(limit, 100)
        tweets = []

        try:
            response = requests.get(
                f"{self.v2_base}/tweets/search/recent",
                headers=self._v2_headers(),
                params={
                    "query": query,
                    "max_results": max_results,
                    "tweet.fields": "created_at,public_metrics,author_id",
                    "expansions": "author_id",
                    "user.fields": "username,name",
                },
                timeout=30,
            )

            if response.status_code == 429:
                raise RateLimitError("Twitter API v2 rate limit exceeded. Please try again later.")
            if response.status_code != 200:
                error = response.json()
                raise PlatformError(
                    f"Twitter API error: {error.get('detail', 'Unknown error')}"
                )

            data = response.json()
            for tweet in data.get("data", []):
                cleaned = self._clean_tweet(tweet.get("text", ""))
                if cleaned:
                    tweets.append(cleaned)

            logger.info(f"[Twitter/v2] Fetched {len(tweets)} tweets for '{query}'.")
            return tweets

        except requests.exceptions.RequestException as e:
            logger.error(f"Twitter v2 request failed: {e}")
            raise PlatformError(f"Failed to connect to Twitter API: {str(e)}")

    # ------------------------------------------------------------------
    # Shared text cleaner
    # ------------------------------------------------------------------

    def _clean_tweet(self, text: str) -> Optional[str]:
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'RT\s*:', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text if len(text) > 3 else None


class TwitterSearchAdapter(TwitterAdapter):
    """
    Thin alias kept for backward compatibility.
    Delegates entirely to TwitterAdapter.fetch_posts().
    """
    def __init__(self, bearer_token: str = None, **kwargs):
        super().__init__(bearer_token=bearer_token, **kwargs)


def create_twitter_adapter() -> TwitterAdapter:
    """Factory function to create a default TwitterAdapter."""
    return TwitterAdapter()