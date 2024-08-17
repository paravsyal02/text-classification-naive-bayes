from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('best_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.sav')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the review text from the request
    data = request.get_json(force=True)
    review = data['review']
    
    # Transform the review using the loaded TF-IDF vectorizer
    transformed_review = vectorizer.transform([review])
    
    # Predict using the loaded model and get the probabilities
    probabilities = model.predict_proba(transformed_review)[0]

    print(f"Probabilities: {probabilities}")

    # Define thresholds
    lower_threshold = 0.45
    upper_threshold = 0.55

    # Determine sentiment based on thresholds
    if probabilities[1] > upper_threshold:
        sentiment = 'negative'
    elif probabilities[1] < lower_threshold:
        sentiment = 'positive'
    else:
        sentiment = 'neutral'
    
    # Return the result as a JSON response
    return jsonify({'prediction': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
