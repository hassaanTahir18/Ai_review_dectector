# main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
from src.preprocessor import preprocess_dataframe
from src.feature_extractor import extract_features


# Step 1: Load dataset
df = pd.read_csv("data/reviews.csv")
print(f"[INFO] Loaded {len(df)} reviews.")

# Step 2: Preprocess text
df = preprocess_dataframe(df)
print("[INFO] Preprocessing complete.")

# Step 3: Extract features
X, vectorizer = extract_features(df['cleaned_review'])
print(f"[INFO] Extracted features. Shape: {X.shape}")

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

# Step 5: Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test)
print("[INFO] Model Evaluation:")
print(classification_report(y_test, y_pred, target_names=['CG (Fake)', 'OR (Original)']))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Step 7: Save model & vectorizer
joblib.dump(model, 'models/fake_review_model.pkl')
print("[INFO] Model saved to 'models/fake_review_model.pkl'")

joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
print("[INFO] Vectorizer saved to 'models/tfidf_vectorizer.joblib'")
