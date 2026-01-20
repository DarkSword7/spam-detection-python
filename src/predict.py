import joblib
import os
import sys

# Ensure we can import from src if running as script
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocess import preprocess_text

# Load artifacts once to avoid reloading on every request if imported
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model/spam_model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), '../model/vectorizer.pkl')

model = None
vectorizer = None

def load_artifacts():
    global model, vectorizer
    if model is None or vectorizer is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError("Model artifacts not found. Please train the model first.")
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)

def predict_message(text: str) -> dict:
    """
    Predicts whether a message is SPAM or HAM.
    
    Args:
        text (str): The input message.
        
    Returns:
        dict: {
            "label": "SPAM" | "HAM",
            "spam_probability": float,
            "ham_probability": float
        }
    """
    load_artifacts()
    
    # Preprocess
    clean_text = preprocess_text(text)
    
    # Vectorize
    text_vectorized = vectorizer.transform([clean_text])
    
    # Predict
    prediction = model.predict(text_vectorized)[0]
    probabilities = model.predict_proba(text_vectorized)[0]
    
    # Map prediction to label (1=SPAM, 0=HAM)
    label = "SPAM" if prediction == 1 else "HAM"
    
    # Probabilities: index 0 is class 0 (HAM), index 1 is class 1 (SPAM)
    ham_prob = probabilities[0]
    spam_prob = probabilities[1]
    
    return {
        "label": label,
        "spam_probability": float(spam_prob),
        "ham_probability": float(ham_prob)
    }

if __name__ == "__main__":
    # Test
    try:
        test_msg = "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/12345 to claim now."
        result = predict_message(test_msg)
        print(result)
    except Exception as e:
        print(f"Error: {e}")
