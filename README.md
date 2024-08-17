Here's a `README.md` file that provides a comprehensive guide on setting up, deploying, and testing your text classification project, including instructions for testing with Postman.

```markdown
# Text Classification with Naive Bayes

This repository contains a Flask-based web application for text classification using Naive Bayes classifiers. The application is deployed on Render and can classify text into sentiments like positive, neutral, and negative.

## Project Overview

- **Language**: Python
- **Framework**: Flask
- **Model**: Naive Bayes (MultinomialNB, BernoulliNB, ComplementNB)
- **Vectorizer**: TF-IDF

## Deployed Application

The application is live at the following URL:

[https://text-classification-naive-bayes.onrender.com](https://text-classification-naive-bayes.onrender.com)

### Endpoints

- **POST /predict**: Predict the sentiment of the provided text.

## Local Setup

To set up the project locally, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. Create a Virtual Environment

```bash
python3 -m venv textclassification
source textclassification/bin/activate  # On Windows use `textclassification\Scripts\activate`
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask Application

```bash
flask run
```

The application will be available at `http://127.0.0.1:5000/`.

## Deployment on Render

To deploy the application on Render:

1. Sign up for an account at [Render](https://render.com/).
2. Create a new Web Service.
3. Connect your GitHub repository.
4. Set the build command to:
   ```bash
   pip install -r requirements.txt
   ```
5. Set the start command to:
   ```bash
   flask run --host=0.0.0.0 --port=10000
   ```
6. Deploy the service.

## Testing the Application with Postman

### 1. Set Up Postman

- Open Postman on your computer.

### 2. Create a New Request

- Click on "New" and select "Request."
- Name your request (e.g., "Test Text Classification").

### 3. Choose the Request Type

- Set the request type to `POST`.

### 4. Enter the Request URL

- In the URL field, enter the following URL:
  ```
  https://text-classification-naive-bayes.onrender.com/predict
  ```

### 5. Set Up the Request Body

- Click on the "Body" tab.
- Select the "raw" option.
- Choose "JSON" from the dropdown menu next to the raw option.
- Enter your JSON payload in the following format:

```json
{
  "text": "Your text here"
}
```

For example:

```json
{
  "text": "I love this movie!"
}
```

### 6. Send the Request

- Click the "Send" button.
- You should receive a JSON response containing the predicted sentiment.

### 7. Review the Response

- The response will look something like this:

```json
{
  "sentiment": "positive"
}
```

## Example Payloads

Here are some example payloads you can test with:

- **Positive Sentiment:**

  ```json
  {
    "text": "This is a fantastic product!"
  }
  ```

- **Negative Sentiment:**

  ```json
  {
    "text": "I hate this experience."
  }
  ```

- **Neutral Sentiment:**

  ```json
  {
    "text": "It was okay, nothing special."
  }
  ```

## License

This project is licensed under the MIT License.

---

Feel free to contribute or raise issues if you encounter any problems.
```

### Key Details Covered in the README

- **Project Overview**: Brief explanation of the project, model, and deployment.
- **Local Setup**: Instructions for setting up the project locally.
- **Deployment**: Guide for deploying the application on Render.
- **Testing with Postman**: Detailed steps to test the `/predict` endpoint using Postman, including the format for the request and example payloads.
- **Example Payloads**: Provides ready-to-use examples to quickly test the application.

This README will help anyone understand, set up, deploy, and test the application effectively.
