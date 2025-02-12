# Fake News Detection
## Project: Fake News Detection ML Project
### Overview
This project focuses on building a machine learning (ML) model to detect fake news by classifying news articles as either real or fake. Fake news detection is an essential task to combat misinformation and ensure the credibility of information. The model is trained using a dataset containing news article titles and corresponding labels (`real` or `fake`).
### Dataset
The dataset used has 3000+ data’s 
The key columns used for detecting fake news are:
#### - title: The title of the news article.
#### - label: A binary label indicating whether the news article is fake (1) or real (0).
### Data Preprocessing
Before training the model, performed the following preprocessing steps:
#### - Tokenization: Breaking down the news titles into words.
#### - Stopwords Removal: Removing common stopwords (e.g., "and", "the") using `nltk`.
#### - TF-IDF Vectorization: Converting text data into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF).
### Model Training & Evaluation
The model is a binary classification model, trained using several ML models like Logistic Regression, Random Forest, KNN, Decision Tree and SVM to check each model performance.
After training, the model is evaluated using common metrics such as: Accuracy, Precision, Recall, F1-score 
### Model Testing 
The model was tested using three methods:
#### 1. Single News Title Input:
   - The pre-trained SVM model and vectorizer were loaded using `pickle` to reuse the saved components for news detection.
   - A single news title was manually input, transformed into a feature vector using the pre-saved vectorizer, and then predicted by the loaded SVM model as either real or fake news.
#### 2. Full News Article Input:
   - A detailed news article was provided, transformed into a vector, and processed by the same pre-trained model for prediction.
#### 3. CSV File Input:
   - Multiple news articles were read from a CSV file, vectorized, and predicted in bulk using the pre-trained model.
