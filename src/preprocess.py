import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
def download_nltk_data():
    resources = ['punkt', 'stopwords', 'punkt_tab']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource, quiet=True)

download_nltk_data()

def preprocess_text(text: str) -> str:
    """
    Preprocesses the input text by:
    1. Converting to lowercase
    2. Removing punctuation and numbers
    3. Tokenizing
    4. Removing stopwords
    5. Stemming
    6. Rejoining
    """
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove punctuation and numbers
    # Keep only alphabets and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 3. Tokenize
    tokens = nltk.word_tokenize(text)
    
    # 4. Remove English stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Apply stemming using PorterStemmer
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # 6. Rejoin tokens into clean text
    clean_text = ' '.join(tokens)
    
    return clean_text

if __name__ == "__main__":
    # Test
    sample = "Hello! This is a 100% SPAM message... buy now!!!"
    print(f"Original: {sample}")
    print(f"Cleaned: {preprocess_text(sample)}")
