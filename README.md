# Text Classification with Naive Bayes

This project is a Flask-based web application deployed on Render for text classification using Naive Bayes algorithms. The application accepts text input via a POST request and returns the predicted sentiment (positive, negative, or neutral) based on the IMDB movie review dataset.

## Features

- **Naive Bayes Models**: Utilizes MultinomialNB, BernoulliNB, and ComplementNB.
- **TF-IDF Vectorization**: Text data is converted to TF-IDF features.
- **Sentiment Analysis**: Predicts whether a review is positive, negative, or neutral.

## Deployed Application

The application is deployed on Render and can be accessed at: [Text Classification Naive Bayes](https://text-classification-naive-bayes.onrender.com)

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- `pip`
- Postman (for testing API endpoints)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/paravsyal02/text-classification-naive-bayes.git
    cd text-classification-naive-bayes
    ```

2. Create a virtual environment and activate it:

    ```bash
    python3 -m venv textclassification
    source textclassification/bin/activate  # On Windows: textclassification\Scripts\activate
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the application locally:

    ```bash
    python app.py
    ```

    The application will be available at `http://127.0.0.1:5000`.

## API Usage

### POST /predict

Predict the sentiment of a review.

- **URL**: `/predict`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Body Format**:

    ```json
    {
        "review": "Your review text here"
    }
    ```

### Example Request in Postman

1. Open Postman and create a new `POST` request.
2. Set the URL to:

    ```plaintext
    https://text-classification-naive-bayes.onrender.com/predict
    ```

3. In the `Body` tab, select `raw` and choose `JSON` as the format.
4. Enter the following JSON object:

    ```json
    {
        "review": "This movie was amazing!"
    }
    ```

5. Click `Send` to receive the prediction.

### Example Response

```json
{
    "prediction": "positive"
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

This `README.md` file provides a comprehensive guide on setting up the project, using the API with Postman, and understanding the expected input/output format.
