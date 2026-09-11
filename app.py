
import os
import logging
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, make_response
from werkzeug.utils import secure_filename
import pandas as pd
import json

import config
from utils.sentiment import predict_sentiment, analyze_sentiment, predict_sentiment_batch
from utils.youtube_fetch import fetch_youtube_comments, YouTubeAPIError, InvalidVideoURLError
from utils.platforms import PlatformError
from utils.platforms.twitter import TwitterAdapter
from utils.platforms.reddit import RedditAdapter
from utils.platforms.instagram import InstagramAdapter
from utils.platforms.facebook import FacebookAdapter
from utils.platforms.news import NewsAdapter
from utils.sectors import get_company_sectors
from utils.wordcloud import generate_wordcloud_by_sentiment, get_word_stats, get_top_words, WORDCLOUD_AVAILABLE
from utils.database import init_database, save_analysis, get_analysis_history, get_statistics
from utils.export import create_response_file

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'csv'}
ALLOWED_PLATFORMS = ['youtube', 'twitter', 'reddit', 'instagram', 'facebook']

try:
    init_database()
    logger.info("Database initialized")
except Exception as e:
    logger.warning(f"Database initialization failed: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_text_input(text):
    if not text or not isinstance(text, str):
        return False, "Text input is required"
    text = text.strip()
    if not text:
        return False, "Text cannot be empty"
    if len(text) > config.MAX_TEXT_LENGTH:
        return False, f"Text exceeds maximum length of {config.MAX_TEXT_LENGTH} characters"
    return True, None

def validate_num_posts(num):
    try:
        num = int(num)
        if not (1 <= num <= config.MAX_POSTS_PER_QUERY):
            return False, f"Number must be between 1 and {config.MAX_POSTS_PER_QUERY}"
        return True, num
    except ValueError:
        return False, "Invalid number"

def calculate_sentiment_stats(results):
    pos = 0
    neg = 0
    neu = 0
    for r in results:
        if len(r) >= 2:
            s = r[1]
        else:
            s = r[0]
        if s == "positive":
            pos += 1
        elif s == "negative":
            neg += 1
        else:
            neu += 1
    total = len(results)
    return {
        'positive': pos,
        'negative': neg,
        'neutral': neu,
        'total': total,
        'positive_pct': round(pos / total * 100, 1) if total > 0 else 0,
        'negative_pct': round(neg / total * 100, 1) if total > 0 else 0,
        'neutral_pct': round(neu / total * 100, 1) if total > 0 else 0,
    }

def analyze_posts(posts, platform, model_name="logistic_regression"):
    if not posts:
        return []
    sentiments = predict_sentiment_batch(posts, model_name=model_name)
    return list(zip(posts, sentiments))

def get_platform_adapter(platform):
    platform = platform.lower()
    if platform == 'youtube':
        return None
    elif platform == 'twitter':
        return TwitterAdapter()
    elif platform == 'reddit':
        return RedditAdapter()
    elif platform == 'instagram':
        return InstagramAdapter()
    elif platform == 'facebook':
        return FacebookAdapter()
    return None


@app.route('/')
def home():
    platforms = config.AVAILABLE_PLATFORMS
    return render_template("index.html", 
                          platforms=platforms)


@app.route('/api/status')
def api_status():
    from utils.sentiment import get_available_models
    return jsonify({
        'status': 'running',
        'models': get_available_models(),
        'platforms': config.AVAILABLE_PLATFORMS,
        'database': os.path.exists(config.DATABASE_PATH)
    })


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """REST API endpoint for sentiment analysis"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    text = data.get('text')
    platform = data.get('platform', 'text')
    query = data.get('query')
    model_name = data.get('model_name', 'logistic_regression')
    num_posts = data.get('num_posts', config.DEFAULT_POSTS_COUNT)
    
    start_time = time.time()
    
    try:
        is_valid, error_msg = validate_text_input(text)
        
        if text:
            if not is_valid:
                return jsonify({'error': error_msg}), 400
            sentiment = predict_sentiment(text, model_name=model_name)
            result = {
                'text': text[:500],
                'sentiment': sentiment,
                'analysis': analyze_sentiment(text, model_name=model_name)
            }
            return jsonify({
                'success': True,
                'data': result,
                'meta': {'processing_time_ms': int((time.time() - start_time) * 1000)}
            })
        
        if platform == 'text':
            return jsonify({'error': 'Text input is required'}), 400
        
        if not query:
            return jsonify({'error': 'Query required for platform analysis'}), 400
        
        posts = []
        
        if platform == 'youtube':
            try:
                is_valid_num, num_posts = validate_num_posts(num_posts)
                if not is_valid_num:
                    return jsonify({'error': num_posts}), 400
                posts = fetch_youtube_comments(query, num_posts)
            except (InvalidVideoURLError, YouTubeAPIError) as e:
                return jsonify({'error': str(e)}), 400
        
        elif platform == 'twitter':
            try:
                is_valid_num, num_posts = validate_num_posts(num_posts)
                if not is_valid_num:
                    return jsonify({'error': num_posts}), 400
                adapter = get_platform_adapter('twitter')
                posts = adapter.fetch_posts(query, num_posts)
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        
        elif platform == 'reddit':
            try:
                is_valid_num, num_posts = validate_num_posts(num_posts)
                if not is_valid_num:
                    return jsonify({'error': num_posts}), 400
                adapter = get_platform_adapter('reddit')
                posts = adapter.fetch_posts(query, num_posts)
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        
        elif platform == 'instagram':
            try:
                is_valid_num, num_posts = validate_num_posts(num_posts)
                if not is_valid_num:
                    return jsonify({'error': num_posts}), 400
                adapter = get_platform_adapter('instagram')
                posts = adapter.fetch_posts(query, num_posts)
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        
        elif platform == 'facebook':
            try:
                is_valid_num, num_posts = validate_num_posts(num_posts)
                if not is_valid_num:
                    return jsonify({'error': num_posts}), 400
                adapter = get_platform_adapter('facebook')
                posts = adapter.fetch_page_posts(query, num_posts)
            except Exception as e:
                return jsonify({'error': str(e)}), 400
        
        else:
            return jsonify({'error': f'Unknown platform: {platform}'}), 400
        
        results = analyze_posts(posts, platform, model_name=model_name)
        stats = calculate_sentiment_stats(results)
        
        try:
            save_analysis(platform, query, stats['total'], stats['positive'],
                         stats['negative'], stats['neutral'], model_name, results)
        except Exception as e:
            logger.warning(f"Failed to save analysis: {e}")
        
        return jsonify({
            'success': True,
            'data': {
                'results': [{'text': t[:200], 'sentiment': s} for t, s in results],
                'stats': stats
            },
            'meta': {
                'platform': platform,
                'query': query,
                'count': len(posts),
                'model': model_name,
                'processing_time_ms': int((time.time() - start_time) * 1000)
            }
        })
        
    except Exception as e:
        logger.exception("API error")
        return jsonify({'error': str(e)}), 500


@app.route('/api/history')
def api_history():
    """Get analysis history"""
    platform = request.args.get('platform')
    limit = request.args.get('limit', 50, type=int)
    
    try:
        history = get_analysis_history(platform, limit)
        return jsonify({'success': True, 'data': history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def api_stats():
    """Get overall statistics"""
    try:
        stats = get_statistics()
        return jsonify({'success': True, 'data': stats})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    try:
        text = request.form.get('text', '')
        model_name = request.form.get('model_name', 'logistic_regression')
        
        is_valid, error_msg = validate_text_input(text)
        if not is_valid:
            return render_template("error.html", error=error_msg), 400
        
        sentiment = predict_sentiment(text, model_name=model_name)
        
        results = [(text, sentiment)]
        stats = calculate_sentiment_stats(results)
        
        try:
            save_analysis('text', text[:100], stats['total'], stats['positive'],
                        stats['negative'], stats['neutral'], model_name, results)
        except Exception as e:
            logger.warning(f"Failed to save analysis: {e}")
        
        chart_data = json.dumps({
            'labels': ['Positive', 'Negative', 'Neutral'],
            'values': [stats['positive'], stats['negative'], stats['neutral']]
        })
        detailed_analysis = analyze_sentiment(text, model_name=model_name)
        
        return render_template(
            "result.html",
            results=results,
            stats=stats,
            chart_data=chart_data,
            analysis_type="text",
            detailed_analysis=detailed_analysis,
            model_name=model_name
        )
        
    except Exception as e:
        logger.exception("Error in analyze_text")
        return render_template("error.html", error=str(e)), 500


@app.route('/analyze_youtube', methods=['POST'])
def analyze_youtube():
    try:
        video_url = request.form.get('video_url', '')
        num_comments = request.form.get('num_comments', config.DEFAULT_POSTS_COUNT)
        model_name = request.form.get('model_name', 'logistic_regression')
        
        is_valid_num, num_comments = validate_num_posts(num_comments)
        if not is_valid_num:
            return render_template("error.html", error=num_comments), 400
        
        try:
            comments = fetch_youtube_comments(video_url, num_comments)
        except (InvalidVideoURLError, YouTubeAPIError, ValueError) as e:
            return render_template("error.html", error=str(e)), 400
        
        if not comments:
            return render_template("error.html", error="No comments found"), 404
        
        results = analyze_posts(comments, 'youtube', model_name=model_name)
        stats = calculate_sentiment_stats(results)
        
        chart_data = json.dumps({
            'labels': ['Positive', 'Negative', 'Neutral'],
            'values': [stats['positive'], stats['negative'], stats['neutral']]
        })
        
        wordcloud_files = {}
        if WORDCLOUD_AVAILABLE and len(results) >= 10:
            try:
                wordcloud_files = generate_wordcloud_by_sentiment(results)
            except Exception as e:
                logger.warning(f"Word cloud generation failed: {e}")
        
        try:
            save_analysis('youtube', video_url, stats['total'], stats['positive'],
                        stats['negative'], stats['neutral'], model_name, results)
        except Exception as e:
            logger.warning(f"Failed to save analysis: {e}")
        
        return render_template(
            "result.html",
            results=results,
            stats=stats,
            chart_data=chart_data,
            analysis_type="youtube",
            wordclouds=wordcloud_files,
            model_name=model_name
        )
        
    except Exception as e:
        logger.exception("Error in analyze_youtube")
        return render_template("error.html", error=str(e)), 500


@app.route('/analyze_twitter', methods=['POST'])
def analyze_twitter():
    try:
        query = request.form.get('query', '')
        num_tweets = request.form.get('num_tweets', config.DEFAULT_POSTS_COUNT)
        model_name = request.form.get('model_name', 'logistic_regression')
        
        if not query.strip():
            return render_template("error.html", error="Query is required"), 400
        
        is_valid_num, num_tweets = validate_num_posts(num_tweets)
        if not is_valid_num:
            return render_template("error.html", error=num_tweets), 400
        
        adapter = TwitterAdapter()
        
        if not adapter.validate_credentials():
            return render_template("error.html", 
                error="Twitter API not configured. Set TWITTER_BEARER_TOKEN in .env"), 400
        
        tweets = adapter.fetch_posts(query, num_tweets)
        
        if not tweets:
            return render_template("error.html", error="No tweets found"), 404
        
        results = analyze_posts(tweets, 'twitter', model_name=model_name)
        stats = calculate_sentiment_stats(results)
        
        chart_data = json.dumps({
            'labels': ['Positive', 'Negative', 'Neutral'],
            'values': [stats['positive'], stats['negative'], stats['neutral']]
        })
        
        try:
            save_analysis('twitter', query, stats['total'], stats['positive'],
                        stats['negative'], stats['neutral'], model_name, results)
        except Exception as e:
            logger.warning(f"Failed to save analysis: {e}")
        
        return render_template(
            "result.html",
            results=results,
            stats=stats,
            chart_data=chart_data,
            analysis_type="twitter",
            model_name=model_name
        )
        
    except Exception as e:
        logger.exception("Error in analyze_twitter")
        return render_template("error.html", error=str(e)), 500


@app.route('/analyze_reddit', methods=['POST'])
def analyze_reddit():
    try:
        query = request.form.get('query', '')
        subreddit = request.form.get('subreddit', '')
        num_posts = request.form.get('num_posts', config.DEFAULT_POSTS_COUNT)
        model_name = request.form.get('model_name', 'logistic_regression')
        
        if not query.strip() and not subreddit.strip():
            return render_template("error.html", 
                error="Query or subreddit is required"), 400
        
        is_valid_num, num_posts = validate_num_posts(num_posts)
        if not is_valid_num:
            return render_template("error.html", error=num_posts), 400
        
        adapter = RedditAdapter()
        posts = adapter.fetch_posts(query, num_posts, subreddit=subreddit if subreddit else None)
        
        if not posts:
            return render_template("error.html", error="No posts found"), 404
        
        results = analyze_posts(posts, 'reddit', model_name=model_name)
        stats = calculate_sentiment_stats(results)
        
        chart_data = json.dumps({
            'labels': ['Positive', 'Negative', 'Neutral'],
            'values': [stats['positive'], stats['negative'], stats['neutral']]
        })
        
        try:
            save_analysis('reddit', query or f"r/{subreddit}", stats['total'], 
                        stats['positive'], stats['negative'], stats['neutral'], 
                        model_name, results)
        except Exception as e:
            logger.warning(f"Failed to save analysis: {e}")
        
        return render_template(
            "result.html",
            results=results,
            stats=stats,
            chart_data=chart_data,
            analysis_type="reddit",
            model_name=model_name
        )
        
    except Exception as e:
        logger.exception("Error in analyze_reddit")
        return render_template("error.html", error=str(e)), 500

@app.route('/analyze_instagram', methods=['POST'])
def analyze_instagram():
    try:
        query = request.form.get('query', 'Recent Posts')
        num_posts = request.form.get('num_posts', config.DEFAULT_POSTS_COUNT)
        model_name = request.form.get('model_name', 'logistic_regression')
        
        is_valid_num, num_posts = validate_num_posts(num_posts)
        if not is_valid_num:
            return render_template("error.html", error=num_posts), 400
        
        adapter = InstagramAdapter()
        
        if not adapter.validate_credentials():
            return render_template("error.html", 
                error="Instagram API not configured. Set INSTAGRAM_ACCESS_TOKEN in .env"), 400
        
        posts = adapter.fetch_posts(query, num_posts)
        
        if not posts:
            return render_template("error.html", error="No posts found"), 404
        
        results = analyze_posts(posts, 'instagram', model_name=model_name)
        stats = calculate_sentiment_stats(results)
        
        chart_data = json.dumps({
            'labels': ['Positive', 'Negative', 'Neutral'],
            'values': [stats['positive'], stats['negative'], stats['neutral']]
        })
        
        try:
            save_analysis('instagram', query, stats['total'], stats['positive'],
                        stats['negative'], stats['neutral'], model_name, results)
        except Exception as e:
            logger.warning(f"Failed to save analysis: {e}")
        
        return render_template(
            "result.html",
            results=results,
            stats=stats,
            chart_data=chart_data,
            analysis_type="instagram",
            model_name=model_name
        )
        
    except Exception as e:
        logger.exception("Error in analyze_instagram")
        return render_template("error.html", error=str(e)), 500


@app.route('/analyze_facebook', methods=['POST'])
def analyze_facebook():
    try:
        page_id = request.form.get('page_id', '').strip()
        num_posts = request.form.get('num_posts', config.DEFAULT_POSTS_COUNT)
        model_name = request.form.get('model_name', 'logistic_regression')
        
        if not page_id:
            return render_template("error.html", error="Facebook Page ID or Username is required"), 400
        
        is_valid_num, num_posts = validate_num_posts(num_posts)
        if not is_valid_num:
            return render_template("error.html", error=num_posts), 400
        
        adapter = FacebookAdapter()
        if not adapter.validate_credentials():
            return render_template("error.html",
                error="Facebook API not configured. Set FACEBOOK_ACCESS_TOKEN in .env"), 400
        
        posts = adapter.fetch_page_posts(page_id, num_posts)
        if not posts:
            return render_template("error.html", error=f"No public posts found for Facebook page '{page_id}'"), 404
        
        results = analyze_posts(posts, 'facebook', model_name=model_name)
        stats = calculate_sentiment_stats(results)
        
        chart_data = json.dumps({
            'labels': ['Positive', 'Negative', 'Neutral'],
            'values': [stats['positive'], stats['negative'], stats['neutral']]
        })
        
        try:
            save_analysis('facebook', page_id, stats['total'], stats['positive'],
                        stats['negative'], stats['neutral'], model_name, results)
        except Exception as e:
            logger.warning(f"Failed to save analysis: {e}")
        
        return render_template(
            "result.html",
            results=results,
            stats=stats,
            chart_data=chart_data,
            analysis_type="facebook",
            model_name=model_name
        )
        
    except Exception as e:
        logger.exception("Error in analyze_facebook")
        return render_template("error.html", error=str(e)), 500


@app.route('/analyze_topic', methods=['POST'])
def analyze_topic():
    """Analyze sentiment trends for any search topic, brand, or company"""
    TOPIC_DIMENSIONS = {
        "Innovation & Tech": ["ai", "chip", "tech", "software", "hardware", "product", "launch", "feature", "update", "patent", "model", "cloud", "digital"],
        "Financials & Growth": ["revenue", "profit", "growth", "earnings", "valuation", "shares", "stock", "margin", "funding", "invest", "capital", "sales"],
        "Leadership & Strategy": ["ceo", "executive", "founder", "hire", "leader", "board", "partner", "strategy", "vision", "appoint", "expansion"],
        "Policy & Market Risk": ["sec", "probe", "investigation", "lawsuit", "regulation", "court", "antitrust", "ban", "scrutiny", "risk", "delay"],
        "Industry & Ecosystem": ["market", "industry", "customer", "user", "competitor", "global", "sector", "adoption", "ecosystem", "trend"]
    }

    EMOTIONS_MAP = {
        "Excitement": ["breakthrough", "record", "soaring", "surge", "blowout", "innovative", "revolution", "unveil", "huge"],
        "Fear": ["crash", "plunge", "lawsuit", "probe", "scandal", "selloff", "warning", "threat", "danger", "crisis"],
        "Uncertainty": ["delay", "wait", "unclear", "volatile", "mixed", "guidance cut", "caution", "scrutiny", "questions"],
        "Optimism": ["upgrade", "growth", "expansion", "promising", "bullish", "opportunity", "partnership", "soar", "gain"]
    }

    TIER_1_SOURCES = ["bloomberg", "reuters", "wsj", "cnbc", "financial times", "economist", "moneycontrol", "livemint", "economic times", "wired", "the verge", "techcrunch", "forbes", "guardian", "bbc", "nytimes"]
    FUTURE_WORDS = ["expected", "projected", "upcoming", "guidance", "pipeline", "will", "future", "estimate", "forecast", "outlook", "anticipate", "roadmap"]

    def extract_bottom_line(text, sentiment):
        sents = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        if not sents:
            return text[:100] + "..."
        if sentiment == "positive":
            for s in sents:
                if any(w in s.lower() for w in ["increase", "growth", "success", "profit", "launch", "high", "beat", "positive", "soar", "deal", "win"]):
                    return s + "."
        elif sentiment == "negative":
            for s in sents:
                if any(w in s.lower() for w in ["decline", "loss", "drop", "fail", "probe", "lawsuit", "low", "miss", "negative", "plunge", "risk"]):
                    return s + "."
        return sents[0] + "."

    try:
        query = request.form.get('query', '').strip()
        source_type = request.form.get('source_type', 'all')
        num_articles = request.form.get('num_articles', 25)
        model_name = request.form.get('model_name', 'logistic_regression')

        if not query:
            return render_template("error.html", error="Please enter a topic or company name to analyze."), 400

        is_valid_num, num_articles_int = validate_num_posts(num_articles)
        if not is_valid_num:
            num_articles_int = 25

        adapter = NewsAdapter()
        raw_articles = adapter.fetch_topic_coverage(query, limit=num_articles_int, source_type=source_type)

        if not raw_articles:
            return render_template("error.html",
                error=f"No recent coverage found for topic '{query}'.\n\nTry a broader search term (e.g. OpenAI, Nvidia, Tesla, Artificial Intelligence, Apple)."), 404

        all_results = []
        all_articles_metadata = []
        source_metrics = {}
        dimension_data = {dim: [] for dim in TOPIC_DIMENSIONS}
        dimension_data["General Outlook"] = []

        for art in raw_articles:
            analysis = analyze_sentiment(art['text'], model_name=model_name)
            sentiment = analysis['sentiment']
            confidence = analysis.get('confidence', 0.5)

            all_results.append((art['text'], sentiment, confidence))

            src = art.get('source', 'Unknown')
            if src not in source_metrics:
                source_metrics[src] = {'total': 0, 'positive': 0, 'negative': 0, 'neutral': 0}
            source_metrics[src]['total'] += 1
            source_metrics[src][sentiment] += 1

            confidence_pct = int(confidence * 100) if confidence else 50
            text_lower = art['text'].lower()
            bottom_line = extract_bottom_line(art['text'], sentiment)

            # Detect dimension
            matched_dim = "General Outlook"
            for dim, words in TOPIC_DIMENSIONS.items():
                if any(w in text_lower for w in words):
                    matched_dim = dim
                    break

            # Tier 1 Source check
            src_lower = src.lower()
            is_tier1 = any(t in src_lower for t in TIER_1_SOURCES)
            if is_tier1:
                confidence_pct = min(100, confidence_pct + 10)

            # Emotions
            emotions = []
            for emo, words in EMOTIONS_MAP.items():
                if any(w in text_lower for w in words):
                    emotions.append(emo)
            top_emotion = emotions[0] if emotions else ("Optimism" if sentiment == "positive" else "Fear" if sentiment == "negative" else "Neutral")

            is_forward = any(w in text_lower for w in FUTURE_WORDS)

            meta = {
                'title': art['title'],
                'source': art['source'],
                'url': art['url'],
                'published_at': art.get('published_at', ''),
                'sentiment': sentiment,
                'confidence': confidence_pct,
                'text': art['text'],
                'bottom_line': bottom_line,
                'tags': [matched_dim],
                'is_tier1': is_tier1,
                'emotion': top_emotion,
                'peers': [],
                'forward_looking': is_forward
            }
            all_articles_metadata.append(meta)
            dimension_data[matched_dim].append(meta)

        stats = calculate_sentiment_stats([(t, s) for t, s, c in all_results])

        chart_data = json.dumps({
            'labels': ['Positive', 'Negative', 'Neutral'],
            'values': [stats['positive'], stats['negative'], stats['neutral']]
        })

        # Build sector/dimension dictionary for radar & cards
        sector_data_dict = {}
        for dim, articles_in_dim in dimension_data.items():
            if not articles_in_dim:
                continue
            dim_total = len(articles_in_dim)
            dim_pos = sum(1 for a in articles_in_dim if a['sentiment'] == 'positive')
            dim_neg = sum(1 for a in articles_in_dim if a['sentiment'] == 'negative')
            dim_neu = sum(1 for a in articles_in_dim if a['sentiment'] == 'neutral')

            sorted_in_dim = sorted(articles_in_dim, key=lambda x: x['confidence'], reverse=True)
            top_pos = [a for a in sorted_in_dim if a['sentiment'] == 'positive'][:2]
            top_neg = [a for a in sorted_in_dim if a['sentiment'] == 'negative'][:1]
            top_kw = [kw[0] for kw in get_top_words([a['text'] for a in articles_in_dim], n=4)]

            sector_data_dict[dim] = {
                'stats': {
                    'total': dim_total,
                    'positive': dim_pos,
                    'negative': dim_neg,
                    'neutral': dim_neu,
                    'positive_pct': round(dim_pos / dim_total * 100, 1) if dim_total > 0 else 0,
                    'negative_pct': round(dim_neg / dim_total * 100, 1) if dim_total > 0 else 0,
                    'neutral_pct': round(dim_neu / dim_total * 100, 1) if dim_total > 0 else 0,
                },
                'top_positive': top_pos,
                'top_negative': top_neg,
                'top_keywords': top_kw
            }

        # Key positive achievements / highlights
        WINNING_WORDS = ['deal', 'growth', 'leader', 'profit', 'record', 'innovat', 'milestone', 'success', 'soar', 'jump', 'rally', 'partner', 'lead', 'surpass']
        key_achievements = []
        for a in sorted(all_articles_metadata, key=lambda x: x['confidence'], reverse=True):
            if a['sentiment'] == 'positive':
                title_lower = a['title'].lower()
                if any(w in title_lower for w in WINNING_WORDS) or len(key_achievements) < 2:
                    if not any(k['title'] == a['title'] for k in key_achievements):
                        key_achievements.append(a)
                        if len(key_achievements) >= 3:
                            break

        # Media verdict
        media_verdict = []
        for a in sorted(all_articles_metadata, key=lambda x: x['confidence'], reverse=True):
            if a['sentiment'] == 'positive' and a['is_tier1']:
                if not any(v['title'] == a['title'] for v in media_verdict):
                    media_verdict.append(a)
                    if len(media_verdict) >= 5:
                        break
        if len(media_verdict) < 5:
            for a in sorted(all_articles_metadata, key=lambda x: x['confidence'], reverse=True):
                if a['sentiment'] == 'positive' and not any(v['title'] == a['title'] for v in media_verdict):
                    media_verdict.append(a)
                    if len(media_verdict) >= 5:
                        break

        # Radar Data
        radar_labels = list(sector_data_dict.keys())
        radar_positivity = [sector_data_dict[k]['stats']['positive_pct'] for k in radar_labels]
        radar_volume = [sector_data_dict[k]['stats']['total'] for k in radar_labels]

        sector_radar_data = json.dumps({
            'labels': radar_labels,
            'positivity': radar_positivity,
            'volume': radar_volume
        })

        # Timeline Data
        timeline_data = {}
        for a in all_articles_metadata:
            date = a['published_at'] or datetime.now().strftime("%Y-%m-%d")
            if date not in timeline_data:
                timeline_data[date] = {'positive': 0, 'negative': 0, 'neutral': 0, 'events': []}
            timeline_data[date][a['sentiment']] += 1
            if len(timeline_data[date]['events']) < 2 and a['confidence'] >= 60:
                timeline_data[date]['events'].append({'title': a['title'], 'sentiment': a['sentiment'], 'url': a['url']})

        sorted_dates = sorted(timeline_data.keys())
        timeline_chart_data = json.dumps({
            'labels': sorted_dates,
            'positive': [timeline_data[d]['positive'] for d in sorted_dates],
            'negative': [timeline_data[d]['negative'] for d in sorted_dates],
            'neutral': [timeline_data[d]['neutral'] for d in sorted_dates],
            'events': [timeline_data[d]['events'] for d in sorted_dates]
        })

        pos_conf = [a['confidence'] for a in all_articles_metadata if a['sentiment'] == 'positive']
        neg_conf = [a['confidence'] for a in all_articles_metadata if a['sentiment'] == 'negative']
        neu_conf = [a['confidence'] for a in all_articles_metadata if a['sentiment'] == 'neutral']

        avg_confidence = {
            'positive': int(sum(pos_conf) / len(pos_conf)) if pos_conf else 70,
            'negative': int(sum(neg_conf) / len(neg_conf)) if neg_conf else 60,
            'neutral': int(sum(neu_conf) / len(neu_conf)) if neu_conf else 50,
        }

        brand_health = int(((stats['positive'] + stats['neutral']) / stats['total']) * 100) if stats['total'] > 0 else 0

        advanced_stats = {
            'polarization': int(abs(stats['positive_pct'] - stats['negative_pct'])),
            'avg_confidence': avg_confidence,
            'sector_radar_data': sector_radar_data,
            'timeline_chart_data': timeline_chart_data,
            'brand_health': brand_health,
            'media_verdict': media_verdict
        }

        template_results = [(t, s) for t, s, c in all_results]
        top_sources = sorted(source_metrics.items(), key=lambda x: x[1]['total'], reverse=True)[:6]

        wordcloud_files = {}
        if WORDCLOUD_AVAILABLE and len(template_results) >= 5:
            try:
                wordcloud_files = generate_wordcloud_by_sentiment(template_results)
            except Exception as e:
                logger.warning(f"Wordcloud generation failed: {e}")

        try:
            save_analysis('topic', query, stats['total'], stats['positive'],
                         stats['negative'], stats['neutral'], model_name, template_results)
        except Exception as e:
            logger.warning(f"Failed to save topic analysis: {e}")

        return render_template(
            "result.html",
            results=template_results,
            stats=stats,
            chart_data=chart_data,
            analysis_type="topic",
            topic_query=query,
            sector_data=sector_data_dict,
            top_sources=top_sources,
            wordcloud_files=wordcloud_files,
            key_achievements=key_achievements,
            advanced_stats=advanced_stats,
            model_name=model_name
        )

    except Exception as e:
        logger.exception("Error in analyze_topic")
        return render_template("error.html", error=str(e)), 500


@app.route('/analyze_news', methods=['POST'])
def analyze_news():
    CATEGORIES_DICT = {
        "Earnings & Financials": ["earnings", "revenue", "profit", "loss", "q1", "q2", "q3", "q4", "dividend", "margin", "fiscal", "sales"],
        "Leadership & Mgt": ["ceo", "cfo", "executive", "board", "management", "director", "founder", "resigns", "appoints"],
        "Regulatory & Legal": ["sec", "lawsuit", "court", "probe", "investigation", "fined", "compliant", "sue", "legal", "antitrust"],
        "Product & Innovation": ["launch", "release", "update", "feature", "beta", "new product", "patent", "innovation", "ai"],
        "Macro & Market": ["inflation", "recession", "economy", "fed", "interest rate", "market", "dow", "nasdaq"]
    }

    EMOTIONS_DICT = {
        "Excitement": ["launch", "breakthrough", "record", "soaring", "surge", "blowout", "innovative"],
        "Fear": ["panic", "crash", "plunge", "lawsuit", "probe", "scandal", "selloff", "warning"],
        "Uncertainty": ["delay", "wait", "unclear", "volatile", "mixed", "guidance cut", "caution"],
        "Optimism": ["upgrade", "growth", "expansion", "promising", "bullish", "opportunity"]
    }

    PEERS_LIST = ["apple", "microsoft", "google", "alphabet", "amazon", "meta", "facebook", "tesla", "nvidia", "reliance", "tata", "infosys", "hdfc", "wipro", "adani", "itc"]
    FUTURE_WORDS = ["expected", "projected", "upcoming", "guidance", "pipeline", "will", "future", "estimate", "forecast", "outlook", "anticipate"]
    TIER_1_SOURCES = ["bloomberg", "reuters", "wsj", "cnbc", "financial times", "economist", "moneycontrol", "livemint", "economic times", "hindu", "wall street"]

    def get_bottom_line(text, sentiment):
        sents = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        if not sents:
            return ""
        if sentiment == "positive":
            for s in sents:
                if any(w in s.lower() for w in ["increase", "growth", "success", "profit", "launch", "high", "beat", "positive", "soar"]):
                    return s + "."
        elif sentiment == "negative":
            for s in sents:
                if any(w in s.lower() for w in ["decline", "loss", "drop", "fail", "probe", "lawsuit", "low", "miss", "negative", "plunge"]):
                    return s + "."
        return sents[0] + "."

    try:
        query = request.form.get('query', '')
        source_type = request.form.get('source_type', 'indian')
        num_articles = request.form.get('num_articles', config.DEFAULT_POSTS_COUNT)
        model_name = request.form.get('model_name', 'logistic_regression')
        
        if not query.strip():
            return render_template("error.html", error="Company name is required"), 400
        
        is_valid_num, num_articles_int = validate_num_posts(num_articles)
        if not is_valid_num:
            return render_template("error.html", error=num_articles_int), 400
        
        adapter = NewsAdapter()
        
        if not adapter.validate_credentials():
            return render_template("error.html", 
                error="News API not configured. Set NEWS_API_KEY in .env file.\n\nGet free key from: https://newsapi.org/register"), 400
                
        sectors = get_company_sectors(query)
        num_per_sector = max(1, num_articles_int // len(sectors))
        
        sector_data_dict = {}
        all_results = []
        all_articles_metadata = []
        source_metrics = {}
        
        for sector in sectors:
            # Append the sector to query to get targeted news insight
            sector_query = f"{query} AND {sector}"
            
            try:
                articles = adapter.search_news(sector_query, num_per_sector, source_type)
            except Exception as e:
                logger.warning(f"Failed to fetch news for sector {sector}: {e}")
                articles = []
                
            sector_results = []
            sector_articles_sentiment = []
            
            for article in articles:
                analysis = analyze_sentiment(article['text'], model_name=model_name)
                sentiment = analysis['sentiment']
                confidence = analysis.get('confidence', 0.5)
                
                sector_results.append((article['text'], sentiment, confidence))
                all_results.append((article['text'], sentiment, confidence))
                
                src = article.get('source', 'Unknown')
                if src not in source_metrics:
                    source_metrics[src] = {'total': 0, 'positive': 0, 'negative': 0, 'neutral': 0}
                source_metrics[src]['total'] += 1
                source_metrics[src][sentiment] += 1
                
                confidence_pct = int(confidence * 100) if confidence else 0
                
                text_lower = article['text'].lower()
                
                # Bottom Line
                bottom_line = get_bottom_line(article['text'], sentiment)
                
                # Top Categories
                tags = []
                for cat, words in CATEGORIES_DICT.items():
                    if any(w in text_lower for w in words):
                        tags.append(cat)
                        
                # Tier 1 Source
                src_lower = article.get('source', '').lower()
                is_tier1 = any(t in src_lower for t in TIER_1_SOURCES)
                if is_tier1:
                    confidence_pct = min(100, confidence_pct + 10)
                    
                # Emotions
                emotions = []
                for emo, words in EMOTIONS_DICT.items():
                    if any(w in text_lower for w in words):
                        emotions.append(emo)
                top_emotion = emotions[0] if emotions else ("Optimism" if sentiment == "positive" else "Fear" if sentiment == "negative" else "Neutral")
                
                # Peers
                q_lower = query.lower()
                peers = [p for p in PEERS_LIST if p in text_lower and p not in q_lower]
                
                # Forward Looking
                is_forward = any(w in text_lower for w in FUTURE_WORDS)

                article_meta = {
                    'title': article['title'],
                    'source': article['source'],
                    'url': article['url'],
                    'published_at': article['published_at'],
                    'sentiment': sentiment,
                    'confidence': confidence_pct,
                    'text': article['text'],
                    'bottom_line': bottom_line,
                    'tags': tags[:2],
                    'is_tier1': is_tier1,
                    'emotion': top_emotion,
                    'peers': peers[:2],
                    'forward_looking': is_forward
                }
                sector_articles_sentiment.append(article_meta)
                all_articles_metadata.append(article_meta)
            
            # Sub-stats for the sector
            s_stats = {
                'positive': sum(1 for _, s, _ in sector_results if s == "positive"),
                'negative': sum(1 for _, s, _ in sector_results if s == "negative"),
                'neutral': sum(1 for _, s, _ in sector_results if s == "neutral"),
                'total': len(sector_results),
            }
            s_total = s_stats['total']
            s_stats['positive_pct'] = round(s_stats['positive'] / s_total * 100, 1) if s_total > 0 else 0
            s_stats['negative_pct'] = round(s_stats['negative'] / s_total * 100, 1) if s_total > 0 else 0
            s_stats['neutral_pct'] = round(s_stats['neutral'] / s_total * 100, 1) if s_total > 0 else 0
            
            # Sort by confidence to get the most important insights
            sector_articles_sentiment.sort(key=lambda x: x['confidence'], reverse=True)
            top_positive = [a for a in sector_articles_sentiment if a['sentiment'] == 'positive'][:2]
            top_negative = [a for a in sector_articles_sentiment if a['sentiment'] == 'negative'][:1]
            
            sector_texts = [a['text'] for a in articles]
            top_keywords = []
            if sector_texts:
                kw_raw = get_top_words(sector_texts, n=5)
                top_keywords = [kw[0] for kw in kw_raw]
            
            sector_data_dict[sector] = {
                'stats': s_stats,
                'top_positive': top_positive,
                'top_negative': top_negative,
                'top_keywords': top_keywords
            }

        if len(all_results) == 0:
            # Fallback: search company query directly without strict AND sector restriction
            fallback_articles = []
            try:
                fallback_articles = adapter.search_news(query, num_articles_int, source_type)
                if not fallback_articles and source_type != "all":
                    fallback_articles = adapter.search_news(query, num_articles_int, "all")
            except Exception as e:
                logger.warning(f"Direct news search fallback error: {e}")
                fallback_articles = []
            
            if fallback_articles:
                primary_sector = sectors[0] if sectors else "Overview"
                sector_results = []
                sector_articles_sentiment = []
                for article in fallback_articles:
                    analysis = analyze_sentiment(article['text'], model_name=model_name)
                    sentiment = analysis['sentiment']
                    confidence = analysis.get('confidence', 0.5)
                    sector_results.append((article['text'], sentiment, confidence))
                    all_results.append((article['text'], sentiment, confidence))
                    
                    src = article.get('source', 'Unknown')
                    if src not in source_metrics:
                        source_metrics[src] = {'total': 0, 'positive': 0, 'negative': 0, 'neutral': 0}
                    source_metrics[src]['total'] += 1
                    source_metrics[src][sentiment] += 1
                    
                    confidence_pct = int(confidence * 100) if confidence else 0
                    text_lower = article['text'].lower()
                    bottom_line = get_bottom_line(article['text'], sentiment)
                    tags = []
                    for cat, words in CATEGORIES_DICT.items():
                        if any(w in text_lower for w in words):
                            tags.append(cat)
                    src_lower = article.get('source', '').lower()
                    is_tier1 = any(t in src_lower for t in TIER_1_SOURCES)
                    if is_tier1:
                        confidence_pct = min(100, confidence_pct + 10)
                    emotions = []
                    for emo, words in EMOTIONS_DICT.items():
                        if any(w in text_lower for w in words):
                            emotions.append(emo)
                    top_emotion = emotions[0] if emotions else ("Optimism" if sentiment == "positive" else "Fear" if sentiment == "negative" else "Neutral")
                    q_lower = query.lower()
                    peers = [p for p in PEERS_LIST if p in text_lower and p not in q_lower]
                    is_forward = any(w in text_lower for w in FUTURE_WORDS)
                    
                    article_meta = {
                        'title': article['title'],
                        'source': article['source'],
                        'url': article['url'],
                        'published_at': article['published_at'],
                        'sentiment': sentiment,
                        'confidence': confidence_pct,
                        'text': article['text'],
                        'bottom_line': bottom_line,
                        'tags': tags[:2],
                        'is_tier1': is_tier1,
                        'emotion': top_emotion,
                        'peers': peers[:2],
                        'forward_looking': is_forward
                    }
                    sector_articles_sentiment.append(article_meta)
                    all_articles_metadata.append(article_meta)
                
                s_stats = {
                    'positive': sum(1 for _, s, _ in sector_results if s == "positive"),
                    'negative': sum(1 for _, s, _ in sector_results if s == "negative"),
                    'neutral': sum(1 for _, s, _ in sector_results if s == "neutral"),
                    'total': len(sector_results),
                }
                s_total = s_stats['total']
                s_stats['positive_pct'] = round(s_stats['positive'] / s_total * 100, 1) if s_total > 0 else 0
                s_stats['negative_pct'] = round(s_stats['negative'] / s_total * 100, 1) if s_total > 0 else 0
                s_stats['neutral_pct'] = round(s_stats['neutral'] / s_total * 100, 1) if s_total > 0 else 0
                sector_articles_sentiment.sort(key=lambda x: x['confidence'], reverse=True)
                sector_data_dict[primary_sector] = {
                    'stats': s_stats,
                    'top_positive': [a for a in sector_articles_sentiment if a['sentiment'] == 'positive'][:2],
                    'top_negative': [a for a in sector_articles_sentiment if a['sentiment'] == 'negative'][:1],
                    'top_keywords': [kw[0] for kw in get_top_words([a['text'] for a in fallback_articles], n=5)] if fallback_articles else []
                }

        if len(all_results) == 0:
            return render_template("error.html", 
                error=f"No news articles found for '{query}'.\n\nTry a broader search term (e.g., Apple, Tesla, Reliance, Google)."), 404
        
        stats = {
            'positive': sum(1 for _, s, _ in all_results if s == "positive"),
            'negative': sum(1 for _, s, _ in all_results if s == "negative"),
            'neutral': sum(1 for _, s, _ in all_results if s == "neutral"),
            'total': len(all_results),
        }
        total = stats['total']
        stats['positive_pct'] = round(stats['positive'] / total * 100, 1) if total > 0 else 0
        stats['negative_pct'] = round(stats['negative'] / total * 100, 1) if total > 0 else 0
        stats['neutral_pct'] = round(stats['neutral'] / total * 100, 1) if total > 0 else 0
        
        chart_data = json.dumps({
            'labels': ['Positive', 'Negative', 'Neutral'],
            'values': [stats['positive'], stats['negative'], stats['neutral']]
        })
        
        try:
            save_analysis('news', query, stats['total'], stats['positive'],
                        stats['negative'], stats['neutral'], model_name, [(t, s) for t, s, c in all_results])
        except Exception as e:
            logger.warning(f"Failed to save analysis: {e}")
        
        template_results = [(text, sentiment) for text, sentiment, _ in all_results]
        
        wordcloud_files = {}
        if WORDCLOUD_AVAILABLE and len(all_results) >= 5:
            try:
                wordcloud_files = generate_wordcloud_by_sentiment([(t, s) for t, s, c in all_results])
            except Exception as e:
                logger.warning(f"Word cloud generation failed for news: {e}")
                
        top_sources = sorted(source_metrics.items(), key=lambda x: x[1]['total'], reverse=True)[:6]
        
        # Extract Key Achievements for the "Investor Pitch" view
        WINNING_WORDS = ['award', 'leader', 'top', 'best', 'growth', 'profit', 'record', 'innovat', 'leading', 'milestone', 'success', 'win', 'won', 'first', 'surpass', 'upside', 'target', 'soar', 'jump', 'rally', 'buy', 'outperform', 'upgrade', 'dividend']
        key_achievements = []
        for a in sorted(all_articles_metadata, key=lambda x: x['confidence'], reverse=True):
            if a['sentiment'] == 'positive':
                title_lower = a['title'].lower()
                if any(word in title_lower for word in WINNING_WORDS):
                    if not any(k['title'] == a['title'] for k in key_achievements):
                        key_achievements.append(a)
                        if len(key_achievements) >= 3:
                            break
        
        # Mathematical Polarization Logic
        if (stats['positive'] + stats['negative']) > 0:
            total_active = stats['positive'] + stats['negative']
            p_ratio = stats['positive'] / total_active
            n_ratio = stats['negative'] / total_active
            polarization_score = int((1 - abs(p_ratio - n_ratio)) * 100)
        else:
            polarization_score = 0
            
        # Confidence Averages
        pos_conf = [a['confidence'] for a in all_articles_metadata if a['sentiment'] == 'positive']
        neg_conf = [a['confidence'] for a in all_articles_metadata if a['sentiment'] == 'negative']
        neu_conf = [a['confidence'] for a in all_articles_metadata if a['sentiment'] == 'neutral']
        
        avg_confidence = {
            'positive': int(sum(pos_conf) / len(pos_conf)) if pos_conf else 0,
            'negative': int(sum(neg_conf) / len(neg_conf)) if neg_conf else 0,
            'neutral': int(sum(neu_conf) / len(neu_conf)) if neu_conf else 0,
        }
        
        # Sector Radar Chart
        radar_labels = []
        radar_positivity = []
        radar_volume = []
        
        for sct, sdat in sector_data_dict.items():
            radar_labels.append(sct)
            radar_positivity.append(sdat['stats']['positive_pct'])
            radar_volume.append(sdat['stats']['total'])
            
        sector_radar_data = json.dumps({
            'labels': radar_labels,
            'positivity': radar_positivity,
            'volume': radar_volume
        })
        
        # Timeline Data
        timeline_data = {}
        for a in all_articles_metadata:
            date = a['published_at']
            if date not in timeline_data:
                timeline_data[date] = {'positive': 0, 'negative': 0, 'neutral': 0, 'events': []}
            timeline_data[date][a['sentiment']] += 1
            if len(timeline_data[date]['events']) < 2 and a['confidence'] >= 60:
                timeline_data[date]['events'].append({'title': a['title'], 'sentiment': a['sentiment'], 'url': a['url']})
                
        sorted_dates = sorted(timeline_data.keys())
        timeline_chart_data = json.dumps({
            'labels': sorted_dates,
            'positive': [timeline_data[d]['positive'] for d in sorted_dates],
            'negative': [timeline_data[d]['negative'] for d in sorted_dates],
            'neutral': [timeline_data[d]['neutral'] for d in sorted_dates],
            'events': [timeline_data[d]['events'] for d in sorted_dates]
        })
        
        # The Media Verdict (Top 5 Global Positives)
        media_verdict = []
        for a in sorted(all_articles_metadata, key=lambda x: x['confidence'], reverse=True):
            if a['sentiment'] == 'positive' and a['is_tier1']:
                if not any(v['title'] == a['title'] for v in media_verdict):
                    media_verdict.append(a)
                    if len(media_verdict) >= 5:
                        break
        # If not enough tier 1 positive, fill with regulars
        if len(media_verdict) < 5:
            for a in sorted(all_articles_metadata, key=lambda x: x['confidence'], reverse=True):
                if a['sentiment'] == 'positive' and a not in media_verdict:
                    if not any(v['title'] == a['title'] for v in media_verdict):
                        media_verdict.append(a)
                        if len(media_verdict) >= 5:
                            break

        advanced_stats = {
            'polarization': polarization_score,
            'avg_confidence': avg_confidence,
            'sector_radar_data': sector_radar_data,
            'timeline_chart_data': timeline_chart_data,
            'brand_health': int(((stats['positive'] + stats['neutral']) / stats['total']) * 100) if stats['total'] > 0 else 0,
            'media_verdict': media_verdict
        }

        return render_template(
            "result.html",
            results=template_results,
            stats=stats,
            chart_data=chart_data,
            analysis_type="news",
            sector_data=sector_data_dict,
            news_articles=None,
            top_sources=top_sources,
            wordcloud_files=wordcloud_files,
            key_achievements=key_achievements,
            advanced_stats=advanced_stats,
            model_name=model_name
        )
        
    except Exception as e:
        logger.exception("Error in analyze_news")
        return render_template("error.html", error=str(e)), 500

@app.route('/analyze_csv', methods=['POST'])
def analyze_csv():
    try:
        model_name = request.form.get('model_name', 'logistic_regression')
        file = request.files.get('file')
        
        if not file:
            return render_template("error.html", error="No file uploaded"), 400
        
        if file.filename == '':
            return render_template("error.html", error="No file selected"), 400
        
        if not allowed_file(file.filename):
            return render_template("error.html", error="Only CSV files are allowed"), 400
        
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return render_template("error.html", error=f"Failed to parse CSV: {str(e)}"), 400
        
        if df.empty:
            return render_template("error.html", error="CSV file is empty"), 400
        
        if len(df.columns) < 1:
            return render_template("error.html", error="CSV must have at least one column"), 400
        
        text_column = df.columns[0]
        valid_texts = [str(text) for text in df[text_column] if pd.notna(text)]
        
        if not valid_texts:
            return render_template("error.html", error="No valid text in CSV"), 400
        
        sentiments = predict_sentiment_batch(valid_texts, model_name=model_name)
        results = list(zip(valid_texts, sentiments))
        
        stats = calculate_sentiment_stats(results)
        chart_data = json.dumps({
            'labels': ['Positive', 'Negative', 'Neutral'],
            'values': [stats['positive'], stats['negative'], stats['neutral']]
        })
        
        try:
            save_analysis('csv', file.filename, stats['total'], stats['positive'],
                        stats['negative'], stats['neutral'], model_name, results)
        except Exception as e:
            logger.warning(f"Failed to save analysis: {e}")
        
        return render_template(
            "result.html",
            results=results,
            stats=stats,
            chart_data=chart_data,
            analysis_type="csv",
            model_name=model_name
        )
        
    except Exception as e:
        logger.exception("Error in analyze_csv")
        return render_template("error.html", error=str(e)), 500


@app.route('/export', methods=['POST'])
def export_results():
    """Export results in various formats"""
    results_json = request.form.get('results')
    format_type = request.form.get('format', 'csv')
    
    if not results_json:
        return render_template("error.html", error="No results to export"), 400
    
    try:
        results = json.loads(results_json)
        results = [(r['text'], r['sentiment']) for r in results]
    except:
        return render_template("error.html", error="Invalid results format"), 400
    
    stats = calculate_sentiment_stats(results)
    
    try:
        filename, content, mimetype = create_response_file(results, stats, format_type)
        response = make_response(content)
        response.headers['Content-Disposition'] = f'attachment; filename={filename}'
        response.headers['Content-Type'] = mimetype
        return response
    except Exception as e:
        return render_template("error.html", error=str(e)), 500


@app.route('/history')
def history_page():
    """View analysis history"""
    try:
        history = get_analysis_history(limit=20)
        stats = get_statistics()
        return render_template("history.html", history=history, stats=stats)
    except Exception as e:
        return render_template("error.html", error=str(e)), 500


if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
    