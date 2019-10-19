import numpy as np
import pandas as pd
import timeit

df = pd.read_csv("reddit_train.csv")

X = df["comments"].values
y = df["subreddits"].values

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 50000)

# Timer begins
start = timeit.default_timer()

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=200)
# neigh.fit(X_train_tfidf, y_train)


#-----------------------------GRID SEARCH CROSS VALIDATION------------------------------
from sklearn.model_selection import GridSearchCV

X_final_train = tfidf_vectorizer.fit_transform(X)

n_folds = 5

tuned_parameters = {
	'n_neighbors': [140, 150, 160, 170, 180],
	'weights': ['uniform', 'distance'],
	'metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(estimator = neigh, param_grid = tuned_parameters, cv = n_folds, refit = False, n_jobs = -1)

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
