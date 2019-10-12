import numpy as np
import pandas as pd

df = pd.read_csv("reddit_train.csv")


X = df["comments"]
y = df["subreddits"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2)
from sklearn.feature_extraction.text import TfidfVectorizer
Tfidf = TfidfVectorizer()
vectors_train_idf = Tfidf.fit_transform(X_train)
vectors_test_idf = Tfidf.transform(X_test)

from sklearn.preprocessing import normalize
vectors_train_normalized = normalize(vectors_train_idf)
vectors_test_normalized = normalize(vectors_test_idf)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(vectors_train_normalized, y_train)
y_pred = lr.predict(vectors_test_normalized)
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred))
