from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn import metrics
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd

train_table = pd.read_csv("reddit_train.csv")
train_table = train_table.drop(columns='id')
comments_train = train_table.iloc[:500, 0].values
subreddits = train_table.iloc[:500,1].values
classes = np.unique(subreddits)

# test_table = pd.read_csv("reddit_test.csv")
# test_table = test_table.drop(columns='id')
# comments_test = test_table.iloc[:, 0].values

# split the dataset into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(comments_train, subreddits, train_size = 0.8, test_size = 0.2)

# vectorize the dataset: used tf-idf method here
vectorizer = CountVectorizer(stop_words='english')
vectors_train = vectorizer.fit_transform(X_train)
vectors_test = vectorizer.transform(X_test)

def fit(X, y):
	voc_len = X.shape[1]
	nb_class = len(classes)
	N = X.shape[0]
	prior = np.zeros(nb_class)
	condprob = np.zeros((voc_len, nb_class))
	for c in range(nb_class):
		Nc = (y == classes[c]).sum()
		prior[c] = Nc/N
		for t in range(voc_len):
			Nct = count_docs_in_class_term(X, y, classes[c], t)
			condprob[t][c] = (Nct+1.0)/(Nc+2.0)
	return prior, condprob

def predit(X_test, X_train, prior, condprob):
	predict_y = []
	for x in X_test:
		score = np.log(prior)
		for c in range(len(classes)):
			for t in range(X_train.shape[1]):
				if x[t] != 0:
					score[c] += np.log(condprob[t][c])
				else:
					score[c] += np.log(1-condprob[t][c])
		predict_y.append(classes[np.argmax(score)])
	return predict_y

def count_docs_in_class_term(X, y, c, t):
	count = 0
	for i in range(X.shape[0]):
		if y[i] == c and X[i][t] != 0:
			count += 1
	return count

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

prior, condprob = fit(vectors_train.toarray(), subreddits)
y_pred = predit(vectors_test.toarray(), vectors_train.toarray(), prior, condprob)
print(accuracy(y_test, y_pred))