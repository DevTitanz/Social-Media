
import os
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import config

os.makedirs(config.MODEL_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(
    "data/training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    header=None
)

df.columns = ["target","id","date","flag","user","text"]
df = df[["target","text"]]

df["sentiment"] = df["target"].replace({0:"negative",4:"positive"})

print(f"Dataset loaded: {len(df)} samples")
print(f"Class distribution:\n{df['sentiment'].value_counts()}")

X = df["text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

print("\nVectorizing text...")
vectorizer = TfidfVectorizer(
    max_features=10000, 
    stop_words="english", 
    ngram_range=(1,2),
    min_df=2,
    max_df=0.95
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Training model...")
model = LogisticRegression(
    max_iter=2000,
    solver='lbfgs',
    C=1.0,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_vec, y_train)

print("\nEvaluating model...")
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                  Predicted")
print(f"                Neg    Pos")
print(f"Actual Neg   [{cm[0][0]:6d} {cm[0][1]:6d}]")
print(f"       Pos   [{cm[1][0]:6d} {cm[1][1]:6d}]")

print("\nSample predictions with probabilities:")
sample_indices = [0, 100, 500]
for idx in sample_indices:
    if idx < len(X_test):
        sample_text = X_test.iloc[idx]
        sample_vec = vectorizer.transform([sample_text])
        pred = model.predict(sample_vec)[0]
        proba = model.predict_proba(sample_vec)[0]
        classes = model.classes_
        
        print(f"\nText: {sample_text[:80]}...")
        print(f"Prediction: {pred}")
        for i, cls in enumerate(classes):
            print(f"  {cls}: {proba[i]:.3f}")

pickle.dump(model, open(config.MODEL_PATH, "wb"))
pickle.dump(vectorizer, open(config.VECTORIZER_PATH, "wb"))

print(f"\nModel saved to: {config.MODEL_PATH}")
print(f"Vectorizer saved to: {config.VECTORIZER_PATH}")
print("\nModel training complete!")
