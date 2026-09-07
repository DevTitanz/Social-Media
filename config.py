import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).parent.resolve()

MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"

DATABASE_PATH = BASE_DIR / "sentiment_analysis.db"

MAX_TEXT_LENGTH = 5000
MAX_POSTS_PER_QUERY = 100
DEFAULT_POSTS_COUNT = 20
MAX_YOUTUBE_COMMENTS = MAX_POSTS_PER_QUERY
DEFAULT_YOUTUBE_COMMENTS = DEFAULT_POSTS_COUNT

NEUTRAL_THRESHOLD_LOW = 0.35
NEUTRAL_THRESHOLD_HIGH = 0.45

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN", "")
FACEBOOK_ACCESS_TOKEN = os.getenv("FACEBOOK_ACCESS_TOKEN", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

ALLOWED_CSV_EXTENSIONS = {".csv"}

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

AVAILABLE_PLATFORMS = {
    "youtube": True,
    "twitter": bool(TWITTER_BEARER_TOKEN),
    "reddit": True,
    "instagram": bool(INSTAGRAM_ACCESS_TOKEN),
    "facebook": bool(FACEBOOK_ACCESS_TOKEN),
    "news": bool(NEWS_API_KEY),
}

WORDCLOUD_WIDTH = 800
WORDCLOUD_HEIGHT = 400
WORDCLOUD_MAX_WORDS = 100
