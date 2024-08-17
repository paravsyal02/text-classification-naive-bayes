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
