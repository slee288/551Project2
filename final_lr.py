import numpy as np
import pandas as pd
import timeit

df = pd.read_csv("reddit_train.csv")

X = df["comments"].values
y = df["subreddits"].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 1234)

#preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 50000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Timer begins
start = timeit.default_timer()

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C = 2.4)
lr.fit(X_train_tfidf, y_train)

from sklearn import metrics
from sklearn.metrics import accuracy_score
y_pred = lr.predict(X_test_tfidf)
accuracy_score(y_test, y_pred)

#-----------------------------GRID SEARCH CROSS VALIDATION------------------------------
from sklearn.model_selection import GridSearchCV

X_final_train = tfidf_vectorizer.fit_transform(X)

tuned_parameters = [{'C' : [2.2, 2.3, 2.4, 2.5]}]
n_folds = 5

grid_search = GridSearchCV(estimator = lr, param_grid = tuned_parameters, cv = n_folds, refit = False, n_jobs = -1)

grid_search.fit(X_final_train, y)

scores = grid_search.cv_results_['mean_test_score']
scores_std = grid_search.cv_results_['std_test_score']
print('scores:',scores)
print('scores_std',scores_std)
print(grid_search.best_params_)

# Timer stops
stop = timeit.default_timer()
print("Time Execution: {}".format(stop - start))
#-----------------------------END OF GRID SEARCH-----------------------------------------

#-----------------------------FINAL TEST PURPOSE ONLY-----------------------
X_final_train = tfidf_vectorizer.fit_transform(X)

df_final = pd.read_csv("reddit_test.csv")
X_final_test = df_final["comments"]
X_final_test = tfidf_vectorizer.transform(X_final_test)

lr.fit(X_final_train, y)
y_final = lr.predict(X_final_test)

predict_arr = np.c_[df_final["id"], y_final]
predict_dataset = pd.DataFrame({"Id": predict_arr[:, 0], "Category":predict_arr[:,1]})
predict_dataset.to_csv("out_lr.csv", index = False)
#--------------------------END OF FINAL TEST-----------------------------------
