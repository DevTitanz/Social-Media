# 📊 Multi-Platform Social Media Sentiment Analyzer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-black?style=for-the-badge&logo=flask&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-RoBERTa-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<p align="center">
  <b>A comprehensive, enterprise-ready sentiment analysis platform supporting multi-platform social harvesting (YouTube, Twitter/X, Reddit, Instagram, Facebook, News, and CSV), dual-engine ML classification (Logistic Regression + RoBERTa Transformer), word clouds, sector analysis, and PDF/CSV reporting.</b>
</p>

[Features](#-key-features) •
[Architecture](#-system-architecture) •
[Models Supported](#-ai--ml-models) •
[Installation & Setup](#-installation--setup) •
[API Keys Configuration](#-api-keys-setup) •
[Usage Guide](#-usage-guide) •
[Project Structure](#-project-structure)

</div>

---

## 🌟 Key Features

- **🌐 Multi-Platform Data Ingestion**:
  - **YouTube**: Fetch and analyze top and recent video comments using YouTube Data API v3.
  - **Twitter / X**: Scrape and classify tweets in real time via Twitter API or RapidAPI.
  - **Reddit**: Ingest discussions and submissions across targeted subreddits.
  - **News & Media**: Real-time headline and article sentiment tracking via NewsAPI.
  - **Instagram & Facebook**: Post and caption sentiment monitoring.
  - **Custom Text & Bulk CSV**: Instant single-text evaluation or bulk CSV dataset processing.

- **🧠 Dual-Engine Sentiment Analysis**:
  - **Logistic Regression + TF-IDF**: Ultra-fast, lightweight inference trained on the 1.6M Sentiment140 dataset.
  - **Twitter-RoBERTa Transformer** (`cardiffnlp/twitter-roberta-base-sentiment-latest`): State-of-the-art contextual deep learning model fine-tuned on social media corpora.
  - **Resilient Fallback Engine**: Rule-based heuristic and lexicon fallback ensures zero downtime even if pre-trained weights are missing.

- **📈 Visual Analytics & Insights**:
  - **Sentiment Distribution**: Clear visualization of Positive, Neutral, and Negative sentiments with confidence scores.
  - **Interactive Word Clouds**: Dedicated word cloud generators for Positive, Neutral, and Negative sentiment clusters.
  - **Sector Classification**: Automatic industry and domain tagging (Finance, Tech, Healthcare, Retail, etc.).
  - **Audit History**: SQLite database persistence tracking past queries, platforms, sample sizes, and timestamps.
  - **Report Export**: 1-click export of analysis results to **CSV** or styled **PDF** reports.

---

## 🏗️ System Architecture

```
                                  USER INTERACTION
                        (Web UI / Single Text / Bulk CSV)
                                       │
                                       ▼
                       FLASK CONTROLLER & DISPATCHER (app.py)
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
┌──────────────────┐         ┌──────────────────┐          ┌──────────────────┐
│ Platform Fetcher │         │ Sentiment Engine │          │ Visual & Reports │
│ (utils/platforms)│         │(utils/sentiment) │          │  (utils/export)  │
├──────────────────┤         ├──────────────────┤          ├──────────────────┤
│ • YouTube API    │         │ • RoBERTa Model   │          │ • Matplotlib     │
│ • Twitter API    │         │ • Logistic + TFIDF│          │ • WordCloud Gen  │
│ • Reddit API     │         │                   │          │ • CSV / PDF      │
│ • NewsAPI        │         │                   │          │ • SQLite DB      │
└──────────────────┘         └──────────────────┘          └──────────────────┘
```

---

## 🤖 AI & ML Models

| Model | Architecture | Best For | Latency |
|---|---|---|---|
| **Logistic Regression** | TF-IDF (100k features) + Scikit-Learn | High-throughput batch processing & bulk CSVs | **< 5 ms** |
| **Twitter-RoBERTa** | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Nuanced slang, sarcasm, and contextual social sentiment | **~ 50–150 ms (GPU/CPU)** |
| **Heuristic Lexicon** | VADER / Rule-Based Keyword Scoring | Offline fallback & instant sanity checks | **< 1 ms** |

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/DevTitanz/Social-Media.git
cd Social-Media
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Train the Logistic Regression Model
To retrain or generate your own `model/sentiment_model.pkl` and `model/vectorizer.pkl`:
1. Download the [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) (`training.1600000.processed.noemoticon.csv`).
2. Place it inside the `data/` directory.
3. Run the training script:
```bash
python train_model.py
```
*(Pre-trained weights are already included in the `model/` directory for immediate out-of-the-box usage).*

### 5. Run the Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🔑 API Keys Setup

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Open `.env` and fill in your developer credentials:
```ini
# YouTube Data API v3 (https://console.cloud.google.com/)
YOUTUBE_API_KEY="your_youtube_api_key"

# Twitter / X Bearer Token (https://developer.twitter.com/)
TWITTER_BEARER_TOKEN="your_twitter_bearer_token"

# Reddit Developer App (https://www.reddit.com/prefs/apps)
REDDIT_CLIENT_ID="your_reddit_client_id"
REDDIT_CLIENT_SECRET="your_reddit_client_secret"

# NewsAPI (https://newsapi.org/)
NEWS_API_KEY="your_news_api_key"

# Optional RapidAPI Key (for fallback Twitter / Instagram scrapers)
RAPIDAPI_KEY="your_rapidapi_key"

# Default Model Selection: "logistic_regression" or "roberta"
DEFAULT_SENTIMENT_MODEL="logistic_regression"

# Logging Level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL="INFO"
```

> **Note:** Platforms without configured API keys will gracefully disable in the UI without affecting the rest of the application (Direct Text and CSV analysis always remain active).

---

## 💻 Usage Guide

### 1. Single Text Analysis
Enter any sentence, tweet, or statement to receive instant confidence scores, sentiment category (Positive / Neutral / Negative), and polarities.

### 2. YouTube Comment Harvesting
Paste any standard YouTube video URL (e.g., `https://www.youtube.com/watch?v=...`) to harvest up to 100 comments, calculate overall community consensus, and generate sentiment word clouds.

### 3. News & Media Monitoring
Track sentiment trends for global brands, stock tickers, or geopolitical topics across real-time news publications.

### 4. Bulk CSV Upload
Upload customer review datasets, survey feedback, or custom social exports to automatically classify each record and download comprehensive aggregate reports.

---

## 📁 Project Structure

```
Social-Media/
├── model/                          # Pre-trained ML weights
│   ├── sentiment_model.pkl         # Logistic regression classifier
│   └── vectorizer.pkl              # TF-IDF vectorizer
├── static/                         # Frontend assets
│   ├── style.css                   # Custom responsive styling & themes
│   └── *.png                       # Generated graphs & word clouds
├── templates/                      # Jinja2 HTML templates
│   ├── index.html                  # Main dashboard & input forms
│   ├── result.html                 # Comprehensive analysis output
│   ├── history.html                # Past queries & audit logs
│   └── error.html                  # Graceful error handling
├── utils/                          # Core application utilities
│   ├── platforms/                  # Social platform integrations
│   │   ├── youtube_fetch.py        # YouTube comments scraper
│   │   ├── twitter.py              # Twitter / X API client
│   │   ├── reddit.py               # Reddit API integration
│   │   ├── news.py                 # NewsAPI aggregator
│   │   ├── instagram.py            # Instagram scraping handler
│   │   └── facebook.py             # Facebook graph integration
│   ├── database.py                 # SQLite query & history manager
│   ├── export.py                   # CSV and PDF report exporter
│   ├── sectors.py                  # Industry / Sector classification
│   ├── sentiment.py                # Dual Logistic & RoBERTa inference
│   └── wordcloud.py                # Word cloud generator
├── .env.example                    # Template environment variables
├── .gitignore                      # Git ignore rules (keys, databases, cache)
├── app.py                          # Main Flask application entrypoint
├── config.py                       # Global configuration & environment loader
├── requirements.txt                # Python package dependencies
├── train_model.py                  # Model training script
└── README.md                       # Documentation
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
