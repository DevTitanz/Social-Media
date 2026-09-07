"""
Sentiment Analysis Module
Uses Trained Logistic Regression Model with TF-IDF Vectorizer
"""
import pickle
import os
import logging
import config
from typing import Dict

logger = logging.getLogger(__name__)

class ModelNotFoundError(Exception):
    pass

def load_model():
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
    
    model = pickle.load(open(config.MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(config.VECTORIZER_PATH, "rb"))
    return model, vectorizer

try:
    model, vectorizer = load_model()
except ModelNotFoundError as e:
    logger.warning(str(e))
    model = None
    vectorizer = None


def predict_sentiment(text: str) -> str:
    """
    Predict sentiment using trained Logistic Regression model.
    
    Args:
        text: Input text to analyze
    
    Returns:
        Sentiment: "positive", "negative", or "neutral"
    """
    global model, vectorizer
    
    if not text or not isinstance(text, str):
        return "neutral"
    
    text = text.strip()
    if not text:
        return "neutral"
    
    if model is None or vectorizer is None:
        try:
            model, vectorizer = load_model()
        except ModelNotFoundError as e:
            raise RuntimeError(
                "Sentiment model not loaded. Please run train_model.py to train the model."
            )
    
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


def analyze_sentiment(text: str) -> Dict:
    """
    Get detailed sentiment analysis including confidence scores.
    
    Returns:
        Dict with keys: sentiment, confidence, scores
    """
    global model, vectorizer
    
    if not text or not isinstance(text, str) or not text.strip():
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "scores": {},
            "model": "logistic_regression"
        }
    
    if model is None or vectorizer is None:
        try:
            model, vectorizer = load_model()
        except ModelNotFoundError:
            raise RuntimeError("Model not loaded. Run train_model.py")
    
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
            confidence = max(float(proba[pos_idx]), float(proba[neg_idx]))
    
    return {
        "sentiment": prediction,
        "confidence": confidence,
        "scores": scores,
        "model": "logistic_regression"
    }


def get_available_models() -> Dict[str, bool]:
    """Get list of available sentiment models"""
    return {
        "model": model is not None
    }
