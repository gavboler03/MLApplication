import numpy as np
import pandas as pd
import itertools
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def Model():
    # Load Data
    fake = pd.read_csv("Fake.csv")
    trueNew = pd.read_csv("True.csv")

    # Make a new column for fake and true datasets
    fake["Truth"] = 0
    trueNew["Truth"] = 1

    # Combine fake and true datasets
    news = pd.concat([fake, trueNew])
    news.head()

    # Clean data
    news.dropna()

    # Get labels
    truth = news.Truth
    truth.head()
    news.columns

    # Split dataset
    X_train, X_test, Y_train, Y_test = train_test_split(
        news["story"], truth, test_size=0.2, random_state=7
    )

    # Use the TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)

    # Initialize PassiveAggresive Classifier
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, Y_train)

    # Set Accuracy
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(Y_test, y_pred)
    acc = f"Accuracy: {round(score*100,2)}%"

    # Confusion Matrix
    confusion_matrix(Y_test, y_pred, labels=[0, 1])
    return acc
