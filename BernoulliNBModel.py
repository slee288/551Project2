import numpy as np
import pandas as pd

train_table = pd.read_csv("reddit_train.csv")
train_table = train_table.drop(columns='id')
comments_train = train_table.iloc[:1000, 0].values
subreddits = train_table.iloc[:,1].values
classes = np.unique(subreddits)

test_table = pd.read_csv("reddit_test.csv")
test_table = test_table.drop(columns='id')
comments_test = test_table.iloc[:, 0].values

def fit(X, y):
	#num class
	nb_class = len(classes)
	#extract entire vocabulary
	V = extract_vocabulary(X)
	prior = np.zeros(nb_class)
	condprob = np.zeros((len(V), nb_class))
	# count number of docs
	N = X.shape[0]
	class_doc = separate_classes(X, y)
	for c in range(nb_class):
		Nc = len(class_doc[classes[c]])
		prior[c] = Nc/N
		for t in range(0, len(V)):
			Nct = count_docs_in_class_term(class_doc, classes[c], V[t])
			condprob[t][c] = (Nct+1)/(Nc+2)
	return V, prior, condprob

def predit(C, V, prior, condprob, D):
	predict_y = []
	for d in D:
		Vd = d.split()
		score = np.zeros(len(C))
		for c in range(len(C)):
			score[c] = np.log(prior[c])
			for t in range(len(V)):
				if V[t] in Vd:
					score[c] += np.log(condprob[t][c])
				else:
					score[c] += np.log(1-condprob[t][c])
		predict_y.append(C[np.argmax(score)])
	return predict_y

def extract_vocabulary(X):
	vocabulary = []
	for x in X:
		line = x.split()
		vocabulary += line
	return np.unique(vocabulary)

def separate_classes(X, y):
	separate = {}
	for c in np.unique(y):
		separate[c] = [x for x, t in zip(X, y) if t == c]
	return separate

def count_docs_in_class_term(dictionary, c, t):
	count = 0
	for d in dictionary[c]:
		if t in d.split():
			count += 1
	return count

V, prior, condprob = fit(comments_train, subreddits)
predict_y = predit(np.unique(subreddits), V, prior, condprob, comments_test)
print(predict_y)