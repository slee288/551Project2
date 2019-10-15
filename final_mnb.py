import numpy as np
import pandas as pd
import timeit

df = pd.read_csv("reddit_train.csv")

X = df["comments"].values
y = df["subreddits"].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 1234)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 50000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Timer begins
start = timeit.default_timer()

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB(alpha = 0.32)
mnb.fit(X_train_tfidf, y_train)

from sklearn import metrics
from sklearn.metrics import accuracy_score
y_pred = mnb.predict(X_test_tfidf)
accuracy_score(y_test, y_pred)

#-----------------------------GRID SEARCH CROSS VALIDATION------------------------------
from sklearn.model_selection import GridSearchCV

X_final_train = tfidf_vectorizer.fit_transform(X)

tuned_parameters = [{'alpha' : [0.30, 0.31, 0.32, 0.33, 0.34]}]
n_folds = 5

grid_search = GridSearchCV(estimator = mnb, param_grid = tuned_parameters, cv = n_folds, refit = False, n_jobs = -1)

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
X_final_test = df_final["comments"].values
X_final_test = tfidf_vectorizer.transform(X_final_test)

mnb.fit(X_final_train, y)
y_final = mnb.predict(X_final_test)

predict_arr = np.c_[df_final["id"], y_final]
predict_dataset = pd.DataFrame({"Id": predict_arr[:, 0], "Category":predict_arr[:,1]})
predict_dataset.to_csv("out_mnb.csv", index = False)
#--------------------------END OF FINAL TEST-----------------------------------
