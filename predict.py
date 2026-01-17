import joblib
from src.preprocessor import clean_text  # your cleaning function

# Load saved model and vectorizer
model = joblib.load('models/fake_review_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.joblib')

def predict_review(text):
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    
    # Get probabilities for each class
    proba = model.predict_proba(vect)[0]  # [prob_fake, prob_original]
    
    # Get predicted label
    prediction = model.predict(vect)[0]
    
    # Create a dict for easier access (adjust class order if needed)
    class_probabilities = dict(zip(model.classes_, proba))
    
    return prediction, class_probabilities

if __name__ == "__main__":
    review = input("Enter a review to classify: ")
    label, probs = predict_review(review)
    
    print(f"Prediction: {'Fake Review (CG)' if label == 'CG' else 'Original Review (OR)'}")
    print(f"Probability of being Fake (CG): {probs.get('CG', 0)*100:.2f}%")
    print(f"Probability of being Original (OR): {probs.get('OR', 0)*100:.2f}%")
