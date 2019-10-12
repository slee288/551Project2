import numpy as np
import pandas as pd

df = pd.read_csv("reddit_train.csv")

X = df["comments"]
y = df["subreddits"]

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# split the dataset into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2)

# vectorize the dataset: used tf-idf method here
Tfidf = TfidfVectorizer()
vectors_train_idf = Tfidf.fit_transform(X_train)
vectors_test_idf = Tfidf.transform(X_test)

# normalize the dataset
vectors_train_normalized = normalize(vectors_train_idf)
vectors_test_normalized = normalize(vectors_test_idf)

# use logistic regression to fit the model & predict the outcomes
lr = LogisticRegression()
lr.fit(vectors_train_normalized, y_train)
y_pred = lr.predict(vectors_test_normalized)

# checking the accuracy
print(metrics.classification_report(y_test, y_pred))

df_final = pd.read_csv("reddit_test.csv")
X_final = df_final["comments"]
vectors_X = Tfidf.transform(X_final)
vectors_X_normalized = normalize(vectors_X)

y_final = lr.predict(vectors_X_normalized)
print(y_final)

predict_arr = np.c_[df_final["id"], y_final]
predict_dataset = pd.DataFrame({"Id": predict_arr[:, 0], "Category":predict_arr[:,1]})
predict_dataset.to_csv("out1.csv", index = False)
