# Email Spam Detection using Machine Learning
# Author: Sarthak Bansal

import numpy as np
import pandas as pd
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download stopwords
nltk.download('stopwords')

# -----------------------------
# Load Dataset
# -----------------------------
# NOTE: Keep spam.csv in the same folder
data = pd.read_csv("spam.csv", encoding="ISO-8859-1")

# Selecting required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# -----------------------------
# Text Preprocessing
# -----------------------------
ps = PorterStemmer()
corpus = []

for msg in data['message']:
    review = re.sub('[^a-zA-Z]', ' ', msg)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# -----------------------------
# Feature Extraction
# -----------------------------
cv = CountVectorizer(max_features=4000)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(data['label'])
y = y.iloc[:, 1].values

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Model Training
# -----------------------------
rfc = RandomForestClassifier()
dtc = DecisionTreeClassifier()
mnb = MultinomialNB()

rfc.fit(X_train, y_train)
dtc.fit(X_train, y_train)
mnb.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
pred_rfc = rfc.predict(X_test)
pred_dtc = dtc.predict(X_test)
pred_mnb = mnb.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
print("\nRandom Forest Classifier")
print("Accuracy:", accuracy_score(y_test, pred_rfc))

print("\nDecision Tree Classifier")
print("Accuracy:", accuracy_score(y_test, pred_dtc))

print("\nMultinomial Naive Bayes")
print("Accuracy:", accuracy_score(y_test, pred_mnb))

print("\nClassification Report (Naive Bayes)")
print(classification_report(y_test, pred_mnb))

# -----------------------------
# Save Models
# -----------------------------
pickle.dump(rfc, open("RFC.pkl", "wb"))
pickle.dump(dtc, open("DTC.pkl", "wb"))
pickle.dump(mnb, open("MNB.pkl", "wb"))

print("\nModels saved successfully!")
