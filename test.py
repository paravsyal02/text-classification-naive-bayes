import joblib
import numpy as np
import pandas as pd

# Load the trained model and vectorizer
model = joblib.load('best_model.pkl')  # Replace with your model filename
vectorizer = joblib.load('tfidf_vectorizer.sav')  # Replace with your vectorizer filename

# New test reviews
test_reviews = [
    "The movie was alright, neither good nor bad. Just an average experience.",
    "I absolutely loved the film. The storyline was captivating and the acting was brilliant.",
    "This movie was a waste of time. I wouldn't recommend it to anyone.",
    "It had some interesting moments, but overall, it didn't leave a strong impression.",
    "An incredible film with outstanding performances. Definitely worth watching!",
    "I was expecting more from this movie. It didn't meet my expectations.",
    "The film was decent, but it dragged in the middle. Could have been better.",
    "Fantastic movie! The direction and cinematography were top-notch.",
    "Not my cup of tea. I didn't connect with the story or characters.",
    "A well-crafted movie with a compelling plot and excellent character development."
]

# Transform and predict
for review in test_reviews:
    transformed_review = vectorizer.transform([review])
    probabilities = model.predict_proba(transformed_review)[0]
    sentiment = 'negative' if probabilities[1] > 0.5 else 'positive'
    print(f"Review: {review}")
    print(f"Probabilities: {probabilities}")
    print(f"Predicted Sentiment: {sentiment}\n")

