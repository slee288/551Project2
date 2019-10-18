import numpy as np
import pandas as pd
import timeit

df = pd.read_csv("reddit_train.csv")

X = df["comments"].values
y = df["subreddits"].values

# DOMAIN KNOWLEDGE
# ADD MY OWN FEATURES
    # - ex. NBA: look for most frequent words
# Try n-gram on countvectorizer

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 1234)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import word_tokenize
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('english')
analyzer = CountVectorizer().build_analyzer()

def stem_analyzer(document):
    return (stemmer.stem(w) for w in analyzer(document))
# class StemmedCountVectorizer(CountVectorizer):
#     def build_analyzer(self):
#         analyzer = super(StemmedCountVectorizer, self).build_analyzer()
#         return lambda document: ([stemmer.stem(w) for w in analyzer(document)])
#
stemmed_cv = CountVectorizer(stop_words = 'english', analyzer = stem_analyzer)
# stemmed_cv = StemmedCountVectorizer(stop_words = 'english', analyzer = stem_analyzer)
X_train_cv = stemmed_cv.fit_transform(X_train)
X_test_cv = stemmed_cv.transform(X_test)
print(X_train_cv.shape)

# def stem_analyzer(document):
#     return (stemmer.stem(w) for w in analyzer(document))
# class StemmedTfidfVectorizer(TfidfVectorizer):
#     def build_analyzer(self):
#         analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
#         return lambda document: ([stemmer.stem(w) for w in analyzer(document)])

tfidf_vectorizer = TfidfTransformer()     # 10000
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_cv)
X_test_tfidf = tfidf_vectorizer.transform(X_test_cv)
print(X_train_tfidf.shape)

# Timer begins
start = timeit.default_timer()

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB(alpha = 0.25)
mnb.fit(X_train_tfidf, y_train)

from sklearn import metrics
from sklearn.metrics import accuracy_score
y_pred = mnb.predict(X_test_tfidf)
print(accuracy_score(y_test, y_pred))

#-----------------------------GRID SEARCH CROSS VALIDATION------------------------------
# from sklearn.model_selection import GridSearchCV
#
# X_final_train_cv = stemmed_cv.fit_transform(X)
# X_final_train = tfidf_vectorizer.fit_transform(X_final_train_cv)
#
# tuned_parameters = [{'alpha' : [0.25, 0.26, 0.27, 0.28]}]
# # tuned_parameters = [{'alpha' : [0.30, 0.31, 0.32, 0.33, 0.34]}]
# n_folds = 10
#
# grid_search = GridSearchCV(estimator = mnb, param_grid = tuned_parameters, cv = n_folds, refit = False, n_jobs = -1)
#
# grid_search.fit(X_final_train, y)
#
# scores = grid_search.cv_results_['mean_test_score']
# scores_std = grid_search.cv_results_['std_test_score']
# print('scores:',scores)
# print('scores_std',scores_std)
# print(grid_search.best_params_)
#
# # Timer stops
# stop = timeit.default_timer()
# print("Time Execution: {}".format(stop - start))
#
#-----------------------------END OF GRID SEARCH-----------------------------------------

#------------------------------Bagging Classifier Purpose-------------------
from sklearn.ensemble import BaggingClassifier
from sklearn import model_selection

X_final_train_cv = stemmed_cv.fit_transform(X)
X_final_train = tfidf_vectorizer.fit_transform(X_final_train_cv)

bg = BaggingClassifier(mnb, max_samples = 0.6, max_features = 0.5, n_estimators = 800)
results = model_selection.cross_val_score(bg, X_final_train, y, cv = 5)
print(results.mean())
# print(bg.score(X_final_train, y))
# Timer stops
stop = timeit.default_timer()
print("Time Execution: {}".format(stop - start))
#------------------------------End of Baggin classifier----------------------

#-----------------------------FINAL TEST PURPOSE ONLY-----------------------
X_final_train_cv = stemmed_cv.fit_transform(X)
X_final_train = tfidf_vectorizer.fit_transform(X_final_train_cv)

df_final = pd.read_csv("reddit_test.csv")
X_final_test = df_final["comments"].values
X_final_test_cv = stemmed_cv.transform(X_final_test)
X_final_test = tfidf_vectorizer.transform(X_final_test_cv)

mnb.fit(X_final_train, y)
y_final = mnb.predict(X_final_test)

predict_arr = np.c_[df_final["id"], y_final]
predict_dataset = pd.DataFrame({"Id": predict_arr[:, 0], "Category":predict_arr[:,1]})
predict_dataset.to_csv("out_mnb1.csv", index = False)
#--------------------------END OF FINAL TEST-----------------------------------
