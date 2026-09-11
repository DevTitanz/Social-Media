"""
Sentiment Analysis Module
Supports:
1. Logistic Regression with TF-IDF Vectorizer (Fast, Lightweight)
2. RoBERTa Transformer Model (cardiffnlp/twitter-roberta-base-sentiment-latest)
"""
import pickle
import os
import logging
import threading
from typing import Dict, List, Tuple
import config

logger = logging.getLogger(__name__)

class ModelNotFoundError(Exception):
    pass

# =====================================================================
# Logistic Regression Model Loading & Fallback
# =====================================================================

def load_logistic_model():
    if not os.path.exists(config.MODEL_PATH):
        raise ModelNotFoundError(
            f"Model file not found at {config.MODEL_PATH}. "
            "Please run train_model.py first."
        )
    if not os.path.exists(config.VECTORIZER_PATH):
        raise ModelNotFoundError(
            f"Vectorizer file not found at {config.VECTORIZER_PATH}. "
            "Please run train_model.py first."
        )
    
    lr_model = pickle.load(open(config.MODEL_PATH, "rb"))
    lr_vectorizer = pickle.load(open(config.VECTORIZER_PATH, "rb"))
    return lr_model, lr_vectorizer

try:
    model, vectorizer = load_logistic_model()
except ModelNotFoundError as e:
    logger.warning(str(e))
    model = None
    vectorizer = None

# Simple sentiment lexicon for resilient fallback if model is unavailable
_POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'amazing', 'love', 'best', 'wonderful', 'fantastic',
    'awesome', 'happy', 'positive', 'win', 'winning', 'growth', 'profit', 'gain',
    'success', 'successful', 'up', 'soar', 'surge', 'leader', 'beat', 'bullish',
    'strong', 'high', 'top', 'innovative', 'breakthrough', 'record', 'dividend'
}
_NEGATIVE_WORDS = {
    'bad', 'terrible', 'worst', 'poor', 'hate', 'awful', 'horrible', 'negative',
    'loss', 'drop', 'fall', 'plunge', 'crash', 'down', 'fail', 'failure', 'probe',
    'lawsuit', 'fraud', 'scandal', 'investigation', 'decline', 'miss', 'bearish',
    'weak', 'low', 'crisis', 'danger', 'risk', 'warning', 'concern', 'delay'
}

def _heuristic_sentiment(text: str) -> Tuple[str, float]:
    words = set(text.lower().split())
    pos_count = len(words & _POSITIVE_WORDS)
    neg_count = len(words & _NEGATIVE_WORDS)
    if pos_count > neg_count:
        return "positive", min(0.95, 0.6 + (pos_count - neg_count) * 0.1)
    elif neg_count > pos_count:
        return "negative", min(0.95, 0.6 + (neg_count - pos_count) * 0.1)
    return "neutral", 0.5


# =====================================================================
# RoBERTa Transformer Model Loading & Pipeline
# =====================================================================

_roberta_model = None
_roberta_tokenizer = None
_roberta_id2label = None
_roberta_device = None
_roberta_lock = threading.Lock()
_roberta_load_failed = False

def load_roberta_model():
    """
    Lazy load the RoBERTa sequence classification model and tokenizer.
    Thread-safe and cached in memory.
    """
    global _roberta_model, _roberta_tokenizer, _roberta_id2label, _roberta_device, _roberta_load_failed
    if _roberta_model is not None:
        return _roberta_model, _roberta_tokenizer, _roberta_id2label, _roberta_device

    if _roberta_load_failed:
        raise RuntimeError("RoBERTa model loading previously failed")

    with _roberta_lock:
        if _roberta_model is not None:
            return _roberta_model, _roberta_tokenizer, _roberta_id2label, _roberta_device
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_name = getattr(config, "ROBERTA_MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment-latest")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            roberta = AutoModelForSequenceClassification.from_pretrained(model_name)
            roberta.to(device)
            roberta.eval()

            # Mapping for cardiffnlp/twitter-roberta-base-sentiment-latest:
            # 0: negative, 1: neutral, 2: positive
            raw_labels = getattr(roberta.config, "id2label", {0: 'negative', 1: 'neutral', 2: 'positive'})
            id2label = {int(k): str(v).lower() for k, v in raw_labels.items()}

            _roberta_model = roberta
            _roberta_tokenizer = tokenizer
            _roberta_id2label = id2label
            _roberta_device = device
            logger.info(f"RoBERTa model loaded successfully on {device} ({model_name})")
            return _roberta_model, _roberta_tokenizer, _roberta_id2label, _roberta_device
        except Exception as e:
            _roberta_load_failed = True
            logger.error(f"Failed to load RoBERTa model: {e}")
            raise


def _is_roberta_available() -> bool:
    try:
        import torch
        import transformers
        return True
    except ImportError:
        return False


def _is_roberta(model_name: str) -> bool:
    if not model_name:
        return False
    name = str(model_name).strip().lower()
    return "roberta" in name


# =====================================================================
# Logistic Regression Predictions
# =====================================================================

def predict_sentiment_logistic(text: str) -> str:
    global model, vectorizer
    if not text or not isinstance(text, str) or not text.strip():
        return "neutral"
    
    text = text.strip()
    if model is None or vectorizer is None:
        try:
            model, vectorizer = load_logistic_model()
        except Exception as e:
            logger.warning(f"Error loading logistic model: {e}")
            s, _ = _heuristic_sentiment(text)
            return s
    
    try:
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)[0]
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(vec)[0]
            classes = list(model.classes_)
            
            positive_idx = list(classes).index('positive') if 'positive' in classes else None
            if positive_idx is not None:
                positive_proba = proba[positive_idx]
                if positive_proba < config.NEUTRAL_THRESHOLD_LOW:
                    return "negative"
                elif positive_proba > config.NEUTRAL_THRESHOLD_HIGH:
                    return "positive"
                else:
                    return "neutral"
        return prediction
    except Exception as e:
        logger.warning(f"Logistic prediction failed, using heuristic: {e}")
        s, _ = _heuristic_sentiment(text)
        return s


def analyze_sentiment_logistic(text: str) -> Dict:
    global model, vectorizer
    if not text or not isinstance(text, str) or not text.strip():
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
            "model": "logistic_regression"
        }
    
    if model is None or vectorizer is None:
        try:
            model, vectorizer = load_logistic_model()
        except Exception:
            s, c = _heuristic_sentiment(text)
            pos_score = c if s == "positive" else (1 - c)/2
            neg_score = c if s == "negative" else (1 - c)/2
            neu_score = c if s == "neutral" else max(0.0, 1.0 - pos_score - neg_score)
            return {
                "sentiment": s,
                "confidence": round(c, 4),
                "scores": {"positive": round(pos_score, 4), "negative": round(neg_score, 4), "neutral": round(neu_score, 4)},
                "model": "heuristic_fallback"
            }
    
    try:
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)[0]
        confidence = 0.5
        scores = {}
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(vec)[0]
            classes = list(model.classes_)
            
            for i, cls in enumerate(classes):
                scores[cls] = float(proba[i])
            
            if 'positive' in classes and 'negative' in classes:
                pos_idx = classes.index('positive')
                neg_idx = classes.index('negative')
                pos_val = float(proba[pos_idx])
                if pos_val < config.NEUTRAL_THRESHOLD_LOW:
                    prediction = "negative"
                    confidence = float(proba[neg_idx])
                elif pos_val > config.NEUTRAL_THRESHOLD_HIGH:
                    prediction = "positive"
                    confidence = pos_val
                else:
                    prediction = "neutral"
                    confidence = 1.0 - abs(pos_val - 0.5) * 2
            else:
                confidence = float(max(proba))
        
        return {
            "sentiment": prediction,
            "confidence": round(confidence, 4),
            "scores": scores,
            "model": "logistic_regression"
        }
    except Exception as e:
        logger.warning(f"Logistic analysis failed, using heuristic: {e}")
        s, c = _heuristic_sentiment(text)
        return {
            "sentiment": s,
            "confidence": round(c, 4),
            "scores": {"positive": c if s == "positive" else 0.2, "negative": c if s == "negative" else 0.2, "neutral": 0.5},
            "model": "heuristic_fallback"
        }


# =====================================================================
# RoBERTa Predictions
# =====================================================================

def predict_sentiment_roberta(text: str) -> str:
    if not text or not isinstance(text, str) or not text.strip():
        return "neutral"
    
    try:
        import torch
        roberta, tokenizer, id2label, device = load_roberta_model()
        
        inputs = tokenizer(text[:1000], return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = roberta(**inputs)
            pred_idx = torch.argmax(outputs.logits, dim=-1).item()
            return id2label.get(pred_idx, "neutral")
    except Exception as e:
        logger.warning(f"RoBERTa prediction failed, falling back to logistic: {e}")
        return predict_sentiment_logistic(text)


def analyze_sentiment_roberta(text: str) -> Dict:
    if not text or not isinstance(text, str) or not text.strip():
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
            "model": "roberta"
        }
    
    try:
        import torch
        roberta, tokenizer, id2label, device = load_roberta_model()
        
        inputs = tokenizer(text[:1000], return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = roberta(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = float(probs[pred_idx].item())
            
            scores = {}
            for idx in range(len(probs)):
                label = id2label.get(idx, f"label_{idx}")
                scores[label] = round(float(probs[idx].item()), 4)
            
            # Ensure standard keys exist
            for std_key in ['positive', 'negative', 'neutral']:
                if std_key not in scores:
                    scores[std_key] = 0.0
            
            return {
                "sentiment": id2label.get(pred_idx, "neutral"),
                "confidence": round(confidence, 4),
                "scores": scores,
                "model": "roberta"
            }
    except Exception as e:
        logger.warning(f"RoBERTa analysis failed, falling back to logistic: {e}")
        res = analyze_sentiment_logistic(text)
        res["model"] = "logistic_regression (fallback)"
        return res


def predict_sentiment_roberta_batch(texts: List[str], batch_size: int = 32) -> List[str]:
    """Batch predict sentiments with RoBERTa for high-speed batch processing"""
    if not texts:
        return []
    
    try:
        import torch
        roberta, tokenizer, id2label, device = load_roberta_model()
        
        results = []
        for i in range(0, len(texts), batch_size):
            batch = [str(t)[:1000] if t and isinstance(t, str) else "" for t in texts[i:i + batch_size]]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = roberta(**inputs).logits
                pred_indices = torch.argmax(logits, dim=-1).tolist()
                for idx in pred_indices:
                    results.append(id2label.get(idx, "neutral"))
        return results
    except Exception as e:
        logger.warning(f"RoBERTa batch prediction failed, falling back to logistic: {e}")
        return [predict_sentiment_logistic(t) for t in texts]


def predict_sentiment_logistic_batch(texts: List[str]) -> List[str]:
    """Batch predict sentiments with Logistic Regression"""
    global model, vectorizer
    if not texts:
        return []
    
    if model is None or vectorizer is None:
        try:
            model, vectorizer = load_logistic_model()
        except Exception:
            return [predict_sentiment_logistic(t) for t in texts]
    
    try:
        cleaned_texts = [str(t) if t and isinstance(t, str) else "" for t in texts]
        vecs = vectorizer.transform(cleaned_texts)
        predictions = model.predict(vecs)
        
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(vecs)
            classes = list(model.classes_)
            if 'positive' in classes:
                pos_idx = classes.index('positive')
                adjusted = []
                for p, pred in zip(probas, predictions):
                    pos_val = p[pos_idx]
                    if pos_val < config.NEUTRAL_THRESHOLD_LOW:
                        adjusted.append("negative")
                    elif pos_val > config.NEUTRAL_THRESHOLD_HIGH:
                        adjusted.append("positive")
                    else:
                        adjusted.append("neutral")
                return adjusted
        return list(predictions)
    except Exception as e:
        logger.warning(f"Logistic batch prediction failed, using single predict: {e}")
        return [predict_sentiment_logistic(t) for t in texts]


# =====================================================================
# Unified Public API
# =====================================================================

def predict_sentiment(text: str, model_name: str = "logistic_regression", **kwargs) -> str:
    """
    Predict sentiment using the selected model.
    
    Args:
        text: Input text to analyze
        model_name: 'logistic_regression' or 'roberta'
    
    Returns:
        Sentiment: "positive", "negative", or "neutral"
    """
    if _is_roberta(model_name):
        return predict_sentiment_roberta(text)
    return predict_sentiment_logistic(text)


def predict_sentiment_batch(texts: List[str], model_name: str = "logistic_regression") -> List[str]:
    """
    Predict sentiments for a list of texts using the selected model.
    """
    if _is_roberta(model_name):
        return predict_sentiment_roberta_batch(texts)
    return predict_sentiment_logistic_batch(texts)


def analyze_sentiment(text: str, model_name: str = "logistic_regression", **kwargs) -> Dict:
    """
    Get detailed sentiment analysis including confidence scores and probability breakdown.
    
    Returns:
        Dict with keys: sentiment, confidence, scores, model
    """
    if _is_roberta(model_name):
        return analyze_sentiment_roberta(text)
    return analyze_sentiment_logistic(text)


def get_available_models() -> Dict[str, bool]:
    """Get list of available sentiment models"""
    return {
        "logistic_regression": model is not None or os.path.exists(config.MODEL_PATH),
        "roberta": _is_roberta_available()
    }
