print("[DEBUG] feature_extractor.py loaded")  # ✅ Debug line

from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(text_series, max_features=5000):
    """
    Converts a Pandas Series of cleaned text into TF-IDF feature vectors.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(text_series)
    return X, vectorizer
