# Professional Spam Detection System

A production-grade Machine Learning application for detecting SMS spam, featuring a professional Streamlit dashboard.

## 📌 Overview

This system uses Natural Language Processing (NLP) and Machine Learning to classify messages as **SPAM** or **HAM** (legitimate). It is built with a focus on reproducibility, modularity, and a professional user interface.

## 🏗️ Architecture

```
spam_detection_system/
│
├── data/               # Dataset storage
├── model/              # Trained ML models (Pickle files)
├── src/                # Source code
│   ├── preprocess.py   # Text cleaning pipeline
│   ├── train_model.py  # Model training script
│   └── predict.py      # Inference logic
├── app.py              # Streamlit Dashboard
└── requirements.txt    # Dependencies
```

## 🚀 How to Run

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data & Train Model
The system requires the SMS Spam Collection dataset and a trained model.

```bash
# Download dataset
python download_data.py

# Train the model
python -m src.train_model
```

### 3. Launch Dashboard
```bash
streamlit run app.py
```

## 🛠️ Technologies Used

- **Python 3.10+**
- **Scikit-learn**: Multinomial Naive Bayes, TF-IDF
- **NLTK**: Text preprocessing (Stemming, Tokenization)
- **Streamlit**: Interactive Web GUI
- **Pandas & NumPy**: Data manipulation
- **Joblib**: Model serialization

## 📊 Model Performance

- **Accuracy**: ~98%
- **Precision**: ~97%
- **Recall**: ~93%
- **F1-Score**: ~95%

## 🔮 Future Scope

- API integration (FastAPI/Flask)
- Deep Learning models (LSTM/BERT)
- Real-time email integration
