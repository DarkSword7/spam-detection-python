import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os
import sys

# Ensure we can import from src if running as script
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocess import preprocess_text

def train():
    print("Loading dataset...")
    df = pd.read_csv("data/spam.csv")
    
    print("Preprocessing data (this may take a while)...")
    # Apply preprocessing
    df['clean_text'] = df['message'].apply(preprocess_text)
    
    X = df['clean_text']
    y = df['label']
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Vectorizing...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*30)
    print("MODEL EVALUATION RESULTS")
    print("="*30)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("="*30 + "\n")
    
    print("Saving model and vectorizer...")
    if not os.path.exists("model"):
        os.makedirs("model")
        
    joblib.dump(model, "model/spam_model.pkl")
    joblib.dump(vectorizer, "model/vectorizer.pkl")
    print("Artifacts saved to model/ directory.")

if __name__ == "__main__":
    train()
