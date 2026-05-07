AI-Powered Fake Review Detector
An end-to-end Machine Learning web application designed to classify product reviews as either "Fake" (Computer Generated) or "Original". The platform utilizes a custom Natural Language Processing (NLP) pipeline and a Logistic Regression model to analyze text semantics, featuring dynamic multilingual support via real-time translation.

🚀 Key Features
Multilingual Processing Pipeline: Integrates the googletrans API to automatically detect and translate non-English reviews into English before passing them to the prediction model.

NLP Text Preprocessing: Custom cleaning module utilizing NLTK to strip punctuation, remove alphanumeric noise, and filter stop-words for optimized tokenization.

Machine Learning Classification: Employs a scikit-learn Logistic Regression model trained on TF-IDF (Term Frequency-Inverse Document Frequency) vectorized text features.

Real-Time Web Interface: A responsive frontend built with Flask and HTML/CSS, providing users with instant classification results and confidence probability scores.

Automated Logging System: Tracks original inputs, translation outputs, and prediction accuracies in a dedicated log file for future model retraining and debugging.

🛠️ Technical Stack
Backend & API: Python, Flask

Machine Learning: scikit-learn, Logistic Regression, Joblib

Natural Language Processing: NLTK, TF-IDF Vectorizer

Data Manipulation: Pandas, NumPy

Translation Engine: Googletrans API

📂 Project Structure
app.py: Main Flask application handling routes, translation, and live predictions.

main.py: Model training script (loads data, extracts features, trains, and serializes the model).

predict.py: CLI testing script for quick model validation.

src/: Core logic modules (preprocessor.py, feature_extractor.py, data_loader.py).

models/: Stores serialized fake_review_model.pkl and tfidf_vectorizer.joblib.

logs/: Stores translation_log.txt for monitoring API health and model inputs.
