"""
Word Cloud Generator for Sentiment Analysis
Generates word clouds for positive, negative, and neutral texts
"""
import re
import logging
from typing import List, Dict, Optional
from collections import Counter

logger = logging.getLogger(__name__)

WORDCLOUD_AVAILABLE = False

try:
    from wordcloud import WordCloud
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    logger.warning("WordCloud not available. Install: pip install wordcloud matplotlib")
    WordCloud = None


import config
import os

STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
    'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
    've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
    'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
    "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
    "weren't", 'won', "won't", 'wouldn', "wouldn't", 'rt', 'via', 'https', 'http',
    'co', 'amp', 'like', 'just', 'got', 'get', 'gets', 'getting', 'go', 'going',
    'went', 'gone', 'know', 'knows', 'knew', 'come', 'comes', 'came', 'make',
    'makes', 'made', 'want', 'wants', 'wanted', 'say', 'says', 'said', 'thing',
    'things', 'think', 'thinks', 'thought', 'see', 'sees', 'saw', 'look', 'looks',
    'looking', 'would', 'could', 'still', 'really', 'one', 'two', 'also', 'even',
    'back', 'way', 'well', 'much', 'say', 'said', 'people', 'time', 'times',
    'day', 'days', 'good', 'great', 'new', 'first', 'last', 'long', 'little',
    'use', 'used', 'using', 'us', 'need', 'needs', 'take', 'takes', 'took', 'taking',
    'give', 'gives', 'gave', 'find', 'finds', 'found', 'tell', 'tells', 'told'
}


def clean_text(text: str) -> str:
    """Clean text for word cloud generation"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    
    return ' '.join(words)


def generate_wordcloud(texts: List[str], sentiment: str = "all", 
                       width: int = None, height: int = None,
                       max_words: int = None) -> Optional[str]:
    """
    Generate a word cloud image from texts.
    
    Args:
        texts: List of text strings
        sentiment: Filter by sentiment ("positive", "negative", "neutral", "all")
        width: Image width
        height: Image height
        max_words: Maximum words to include
    
    Returns:
        Base64 encoded image string or None if failed
    """
    if not WORDCLOUD_AVAILABLE:
        logger.warning("WordCloud library not available")
        return None
    
    if not texts:
        return None
    
    width = width or config.WORDCLOUD_WIDTH
    height = height or config.WORDCLOUD_HEIGHT
    max_words = max_words or config.WORDCLOUD_MAX_WORDS
    
    cleaned_texts = [clean_text(t) for t in texts]
    combined_text = ' '.join(cleaned_texts)
    
    if not combined_text.strip():
        return None
    
    color_map = {
        "positive": "Greens",
        "negative": "Reds",
        "neutral": "Blues",
        "all": "viridis"
    }
    
    colormap = color_map.get(sentiment.lower(), "viridis")
    
    try:
        wordcloud = WordCloud(
            width=width,
            height=height,
            max_words=max_words,
            background_color='white',
            colormap=colormap,
            min_font_size=10,
            max_font_size=100,
            random_state=42
        ).generate(combined_text)
        
        return wordcloud
        
    except Exception as e:
        logger.error(f"Failed to generate word cloud: {e}")
        return None


def generate_wordcloud_by_sentiment(results: List[tuple], 
                                     output_dir: str = "static") -> Dict[str, str]:
    """
    Generate separate word clouds for each sentiment category.
    
    Args:
        results: List of (text, sentiment) tuples
        output_dir: Directory to save images
    
    Returns:
        Dict mapping sentiment to image filename
    """
    if not WORDCLOUD_AVAILABLE:
        return {}
    
    os.makedirs(output_dir, exist_ok=True)
    
    sentiments = {
        "positive": [],
        "negative": [],
        "neutral": []
    }
    
    for text, sentiment in results:
        if sentiment in sentiments:
            sentiments[sentiment].append(text)
    
    output_files = {}
    
    for sentiment, texts in sentiments.items():
        if texts:
            wc = generate_wordcloud(texts, sentiment=sentiment)
            if wc:
                filename = f"wordcloud_{sentiment}.png"
                filepath = os.path.join(output_dir, filename)
                wc.to_file(filepath)
                output_files[sentiment] = filename
    
    return output_files


def get_top_words(texts: List[str], n: int = 20, sentiment: str = None) -> List[tuple]:
    """
    Get top N words from texts, optionally filtered by sentiment.
    
    Args:
        texts: List of text strings or List of (text, sentiment) tuples
        n: Number of top words to return
        sentiment: Filter by sentiment (optional)
    
    Returns:
        List of (word, count) tuples
    """
    if not texts:
        return []
    
    if isinstance(texts[0], tuple):
        if sentiment:
            texts = [t for t, s in texts if s == sentiment]
        else:
            texts = [t for t, _ in texts]
    
    cleaned_texts = [clean_text(t) for t in texts]
    combined_text = ' '.join(cleaned_texts)
    
    words = combined_text.split()
    counter = Counter(words)
    
    return counter.most_common(n)


def get_word_stats(results: List[tuple]) -> Dict:
    """Get word statistics for each sentiment category"""
    positive_texts = [t for t, s in results if s == "positive"]
    negative_texts = [t for t, s in results if s == "negative"]
    neutral_texts = [t for t, s in results if s == "neutral"]
    
    return {
        "positive": {
            "count": len(positive_texts),
            "top_words": get_top_words(positive_texts, 10)
        },
        "negative": {
            "count": len(negative_texts),
            "top_words": get_top_words(negative_texts, 10)
        },
        "neutral": {
            "count": len(neutral_texts),
            "top_words": get_top_words(neutral_texts, 10)
        }
    }
