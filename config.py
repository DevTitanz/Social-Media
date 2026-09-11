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
ROBERTA_MODEL_NAME = os.getenv("ROBERTA_MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment-latest")
DEFAULT_SENTIMENT_MODEL = os.getenv("DEFAULT_SENTIMENT_MODEL", "logistic_regression")

DATABASE_PATH = BASE_DIR / "sentiment_analysis.db"

MAX_TEXT_LENGTH = 5000
MAX_POSTS_PER_QUERY = 100
DEFAULT_POSTS_COUNT = 20
MAX_YOUTUBE_COMMENTS = MAX_POSTS_PER_QUERY
DEFAULT_YOUTUBE_COMMENTS = DEFAULT_POSTS_COUNT

NEUTRAL_THRESHOLD_LOW = 0.35
NEUTRAL_THRESHOLD_HIGH = 0.45

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "").strip()
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "").strip()
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN", "").strip()
FACEBOOK_ACCESS_TOKEN = os.getenv("FACEBOOK_ACCESS_TOKEN", "").strip()
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "").strip()
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "").strip() or INSTAGRAM_ACCESS_TOKEN or TWITTER_BEARER_TOKEN

ALLOWED_CSV_EXTENSIONS = {".csv"}

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

AVAILABLE_PLATFORMS = {
    "text": True,
    "youtube": bool(YOUTUBE_API_KEY),
    "twitter": bool(TWITTER_BEARER_TOKEN or RAPIDAPI_KEY),
    "reddit": bool(REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET),
    "instagram": bool(INSTAGRAM_ACCESS_TOKEN or RAPIDAPI_KEY),
    "facebook": bool(FACEBOOK_ACCESS_TOKEN),
    "news": bool(NEWS_API_KEY),
    "csv": True,
}

WORDCLOUD_WIDTH = 800
WORDCLOUD_HEIGHT = 400
WORDCLOUD_MAX_WORDS = 100
