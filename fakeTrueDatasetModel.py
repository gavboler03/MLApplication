import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Load Data*
fake = pd.read_csv("Fake.csv")
trueNew = pd.read_csv("True.csv")


# Create New Column For New DataFrame
fake["Truth"] = 0
trueNew["Truth"] = 1

# Combine Fake and True DataSets
news = pd.concat([fake, trueNew])

# Clean Data
# news.dropna()

# Split Data Into Test and Training sets
X_train, X_test, Y_train, Y_test = train_test_split(
    news["story"], news["Truth"], test_size=0.2, random_state=7
)

# Set the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Train the PassiveAgressive Classifier
pac = PassiveAggressiveClassifier(max_iter=100)
pac.fit(tfidf_train, Y_train)

# Set Accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {round(score*100,2)}%")

# Set Confusion Matrix
confusion_matrix(Y_test, y_pred, labels=[0, 1])
