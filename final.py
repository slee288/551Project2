import numpy as np
import pandas as pd

df = pd.read_csv("reddit_train.csv")

import nltk
import string
from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.preprocessing import normalize

lemmatizer = spacy.load('en', disable = ['parser', 'ner'])
def tokenize(text):
    tokens = lemmatizer(text)
    tokens = [token.lemma_ for token in tokens]
    tokens = [w for w in tokens]
    tokens = [w for w in tokens if w not in string.punctuation and len(w) > 2 and w != '-PRON-']  # remove punctuations, words less than 3 characters, and pronouns
    return tokens

X = df["comments"].values
y = df["subreddits"].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 124)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words = 'english')
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_cv)
X_test_tfidf = tfidf_transformer.transform(X_test_cv)

# normalize the dataset
# vectors_train_normalized = normalize(X_train_tfidf)
# vectors_test_normalized = normalize(X_test_tfidf)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_tfidf, y_train)
y_pred = lr.predict(X_test_tfidf)
# lr.fit(vectors_train_normalized, y_train)
# y_pred = lr.predict(vectors_test_normalized)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy
print(accuracy(y_test, y_pred))
# from sklearn import metrics
# print(metrics.classification_report(y_test, y_pred))

# ===================== For final test purpose only ========================
df_final = pd.read_csv("reddit_test.csv")
X_final = df_final["comments"]
vectors_X = cv.transform(X_final)
vectors_X_tfidf = tfidf_transformer.transform(vectors_X)

y_final = lr.predict(vectors_X_tfidf)
print(y_final)

predict_arr = np.c_[df_final["id"], y_final]
predict_dataset = pd.DataFrame({"Id": predict_arr[:, 0], "Category":predict_arr[:,1]})
predict_dataset.to_csv("out.csv", index = False)
#============================================================================
