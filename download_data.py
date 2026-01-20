import urllib.request
import zipfile
import os
import pandas as pd

def download_and_format_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = "data/smsspamcollection.zip"
    extract_path = "data/extracted"
    csv_path = "data/spam.csv"

    if not os.path.exists("data"):
        os.makedirs("data")

    print("Downloading dataset...")
    urllib.request.urlretrieve(url, zip_path)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # The file is usually named 'SMSSpamCollection' and is tab-separated
    raw_file_path = os.path.join(extract_path, "SMSSpamCollection")
    
    print("Formatting to CSV...")
    df = pd.read_csv(raw_file_path, sep='\t', header=None, names=['label', 'message'])
    
    # Convert labels: spam -> 1, ham -> 0 (User requested this in Phase 2)
    # Actually User said: "Convert labels: spam -> 1, ham -> 0" in Phase 2 AI Action.
    # But usually it's better to keep string labels in CSV and encode later, OR encode now.
    # The instruction says: "Convert labels: spam -> 1, ham -> 0" under "AI Action".
    # So I will do it here or in the loading step. 
    # Let's do it here to have a clean CSV ready for training.
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    
    # Validate no null
    df.dropna(inplace=True)
    
    # Save
    df.to_csv(csv_path, index=False)
    
    print(f"Dataset saved to {csv_path}")
    print(df['label'].value_counts())

    # Cleanup
    os.remove(zip_path)
    # import shutil
    # shutil.rmtree(extract_path)

if __name__ == "__main__":
    download_and_format_data()
