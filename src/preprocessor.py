import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def preprocess_dataframe(df):
    # Use the correct column name 'text_' from your CSV
    df['cleaned_review'] = df['text_'].apply(clean_text)
    return df

print("[DEBUG] preprocessor.py loaded")
