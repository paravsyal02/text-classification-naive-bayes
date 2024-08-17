# %%
"""
## Text Classification Using Naive Bayes

### Overview

Text classification involves categorizing text data into predefined labels or classes. Naive Bayes is a widely used algorithm for this task due to its simplicity and efficiency.

### Naive Bayes Algorithm

Naive Bayes is a probabilistic classifier based on Bayes' Theorem. It operates under the assumption that features (e.g., words) are independent of each other given the class label. Despite this simplifying assumption, it often performs well in practice.

### Key Types

1. **Multinomial Naive Bayes**: Ideal for text where features are word counts or frequencies. Commonly used for document classification.

2. **Bernoulli Naive Bayes**: Suitable when features are binary, indicating the presence or absence of words.

3. **Complement Naive Bayes**: Designed to handle imbalanced datasets by correcting class imbalances.

### Advantages

- **Simple and Fast**: Easy to implement and computationally efficient.
- **Effective with High-Dimensional Data**: Works well with large feature sets like text data.
- **Requires Less Training Data**: Performs effectively even with smaller datasets.

### Applications

- **Spam Detection**: Identifying spam emails versus legitimate ones.
- **Sentiment Analysis**: Determining the sentiment (positive, negative, neutral) of text.
- **Topic Classification**: Assigning documents to specific topics or categories.

### Conclusion

Naive Bayes is a robust and straightforward method for text classification. Its effectiveness in handling text data and simplicity in implementation make it a popular choice for various NLP tasks.

"""

# %%
#Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib import style
style.use('ggplot')
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words =  set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# %%
#Load the dataset
df = pd.read_csv("IMDB Dataset.csv")
df.head()

# %%
# Inspecting data
df.shape

# %%
df.info()

# %%
sns.countplot(x='sentiment', data=df)
plt.title("Sentiment distribution")

# %%
for i in range(5):
    print("Review: ", [i])
    print(df['review'].iloc[i], "\n")
    print("Sentiment: ", df['sentiment'].iloc[i], "\n\n")

# %%
def no_of_words(text):
    words= text.split()
    word_count = len(words)
    return word_count

# %%
df['word count'] = df['review'].apply(no_of_words)

# %%
df.head()

# %%
fig, ax = plt.subplots(1,2, figsize=(10,6))
ax[0].hist(df[df['sentiment'] == 'positive']['word count'], label='Positive', color='blue', rwidth=0.9);
ax[0].legend(loc='upper right');
ax[1].hist(df[df['sentiment'] == 'negative']['word count'], label='Negative', color='red', rwidth=0.9);
ax[1].legend(loc='upper right');
fig.suptitle("Number of words in review")
plt.show()

# %%
fig, ax = plt.subplots(1,2, figsize=(10,6))
ax[0].hist(df[df['sentiment'] == 'positive']['review'].str.len(), label='Positive', color='blue', rwidth=0.9);
ax[0].legend(loc='upper right');
ax[1].hist(df[df['sentiment'] == 'negative']['review'].str.len(), label='Negative', color='red', rwidth=0.9);
ax[1].legend(loc='upper right');
fig.suptitle("Number of words in review")
plt.show()

# %%
df.sentiment.replace("positive", 1, inplace=True)
df.sentiment.replace("negative", 2, inplace=True)

# %%
df.head()

# %%
# Data Preprocessing
def data_processing(text):
    text= text.lower()
    text = re.sub('<br />', '', text)
    text = re.sub(r"https\S+|www\S+|http\S+", '', text, flags = re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)

# %%
df.review = df['review'].apply(data_processing)

# %%
duplicated_count = df.duplicated().sum()
print("Number of duplicate entries: ", duplicated_count)

# %%
df = df.drop_duplicates('review')

# %%
stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data

# %%
df.review = df['review'].apply(lambda x: stemming(x))

# %%
df['word count'] = df['review'].apply(no_of_words)
df.head()

# %%
pos_reviews =  df[df.sentiment == 1]
pos_reviews.head()

# %%
text = ' '.join([word for word in pos_reviews['review']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in positive reviews', fontsize = 19)
plt.show()

# %%
from collections import Counter
count = Counter()
for text in pos_reviews['review'].values:
    for word in text.split():
        count[word] +=1
count.most_common(15)

# %%
pos_words = pd.DataFrame(count.most_common(15))
pos_words.columns = ['word', 'count']
pos_words.head()

# %%
px.bar(pos_words, x='count', y='word', title='Common words in positive reviews', color = 'word')

# %%
neg_reviews =  df[df.sentiment == 2]
neg_reviews.head()

# %%
text = ' '.join([word for word in neg_reviews['review']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in negative reviews', fontsize = 19)
plt.show()


# %%
count = Counter()
for text in neg_reviews['review'].values:
    for word in text.split():
        count[word] +=1
count.most_common(15)

# %%
neg_words = pd.DataFrame(count.most_common(15))
neg_words.columns = ['word', 'count']
neg_words.head()


# %%
px.bar(neg_words, x='count', y='word', title='Common words in negative reviews', color = 'word')

# %%
X = df['review']
Y = df['sentiment']

# %%
vect = TfidfVectorizer()
X_tfidf = vect.fit_transform(df['review'])

# %%
x_train, x_test, y_train, y_test = train_test_split(X_tfidf, Y, test_size=0.3, random_state=42)

# %%
print("Size of x_train: ", (x_train.shape))
print("Size of y_train: ", (y_train.shape))
print("Size of x_test: ", (x_test.shape))
print("Size of y_test: ", (y_test.shape))

# %%
# Import Required Libraries
from sklearn.naive_bayes import MultinomialNB , BernoulliNB , ComplementNB
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score , confusion_matrix , classification_report

# %%
# Initialize and Train Models
models = {
    'MultinomialNB' : MultinomialNB(),
    'BernoulliNB' : BernoulliNB(),
    'ComplementNB' : ComplementNB()
}

results = {}

# %%
# Evaluate Models
for name , model in models.items():

    #Train the model
    model.fit(x_train , y_train)

    # Make predictions
    y_pred = model.predict(x_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': report
    }

#Print results
for name, metrics in results.items():
    print(f"Model: {name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    print("Classification Report:")
    print(metrics['classification_report'])
    print("-" * 80)


# %%
from sklearn.naive_bayes import MultinomialNB

best_model = MultinomialNB()
best_model.fit(x_train, y_train)

# %%
import joblib

#Assuming best_model is your trained model
joblib.dump(best_model , 'best_model.pkl')

# Save the vectorizer
joblib.dump(vect, 'tfidf_vectorizer.sav')