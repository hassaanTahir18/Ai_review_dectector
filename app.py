from flask import Flask, render_template, request
import joblib
from src.preprocessor import clean_text
from googletrans import Translator
import os

app = Flask(__name__)

# Load model and vectorizer once at startup
model = joblib.load('models/fake_review_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.joblib')

# Initialize the translator
translator = Translator()

# Path to log file
log_file_path = 'logs/translation_log.txt'

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None

    if request.method == "POST":
        review = request.form["review"]

        try:
            translated = translator.translate(review, dest='en').text
        except Exception as e:
            translated = review
            log_message = f"ERROR: Translation failed: {e}\n"
        else:
            log_message = (
                f"Original: {review}\n"
                f"Translated: {translated}\n"
            )

        # Log to file
        os.makedirs('logs', exist_ok=True)  # Ensure the folder exists
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write(log_message + '\n')

        cleaned = clean_text(translated)
        vect = vectorizer.transform([cleaned])

        proba = model.predict_proba(vect)[0]
        prediction = model.predict(vect)[0]
        class_probabilities = dict(zip(model.classes_, proba))

        if prediction == 'OR':
            result = "Original review"
            probability = class_probabilities.get('OR', 0) * 100
        else:
            result = "Fake Review"
            probability = class_probabilities.get('CG', 0) * 100

        # Log result
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"Prediction: {result} - {probability:.2f}%\n")
            log_file.write("-" * 40 + "\n")

    return render_template("index.html", result=result, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)
