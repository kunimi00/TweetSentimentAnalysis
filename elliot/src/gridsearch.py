import sys
sys.path.append('../')

from src.processing import *
import argparse
from time import time
from src.gridsearchhelper import *
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, f1_score
import pickle

choices = ['svm', 'logistic_regression',
           'knn',
           'naive_bayes_bernoulli', 'naive_bayes_binomial', 'neural_network']
vectorizer = CountVectorizer(binary=False, ngram_range=(1, 1),
                          tokenizer=tokenize,
                           lowercase=True)

# svc = Pipeline([('vectorizer', vectorizer), ('svm', SVC(C=0.1, kernel='linear'))])
# logistic_regression = Pipeline([('vectorizer', vectorizer), ('logistic_regression', LogisticRegression())])
# knn = Pipeline([('vectorizer', vectorizer), ('knn', KNeighborsClassifier())])
# naive_bayes_bernoulli = Pipeline([('vectorizer', vectorizer), ('naive_bayes_bernoulli', BernoulliNB())])
# naive_bayes_binomial = Pipeline([('vectorizer', vectorizer), ('naive_bayes_binomial', MultinomialNB())])
# neural_network = Pipeline([('vectorizer', vectorizer), ('neural_network', MLPClassifier())])
# classifiers = [
#     svc, logistic_regression, knn, naive_bayes_bernoulli, naive_bayes_binomial, neural_network
# ]

classifiers = [SVC(), LogisticRegression(),
               KNeighborsClassifier(),
               BernoulliNB(), MultinomialNB(),
               MLPClassifier(max_iter=300)]

classifiers = dict(zip(choices, classifiers))

params = [

    # {'ngram_range': [(1,1), (1,2), (1,3), (2,2)]},
    [
        {'kernel': ['linear'], 'C': [1, 10]},
        {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
    ],
    {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
    {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    {'alpha': [0, 0.5, 1, 2]},
    {'alpha': [0, 0.5, 1, 2]}, {'hidden_layer_sizes': [(10), (100), (500), (1000)]}
]
params = dict(zip(choices, params))
print(params)
print(classifiers)

# def main(arguments):



if __name__ == '__main__':

    begin = time()
    parser = argparse.ArgumentParser(
        description="Perform a grid search over parameters for defined classifiers by cross validation on training data",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('train', help="Input training data file in format TEXT<tab>LABEL or TEXT<tab>TOPIC<tab>LABEL",
                        type=argparse.FileType('r'))

    args = vars(parser.parse_args(sys.argv[1:]))
    train = args['train'].name
    train_tweets, train_labels = load_data(train)
    X = vectorizer.fit_transform(train_tweets)

    helper = EstimatorSelectionHelper(classifiers, params)
    my_func = make_scorer(f1_average, greater_is_better=True)
    helper.fit(X, np.array(train_labels), scoring=my_func, n_jobs=-1)
    # print(KNeighborsClassifier().fit(X[:5,:], train_labels[:5]).predict(X[:5,:]))

    # knn = KNeighborsClassifier(algorithm='brute')
    # clf = GridSearchCV(knn, params['knn'], cv=2)
    # clf.fit(X[:15,:], np.array(train_labels[:15])[:,np.newaxis])
    # print(clf.best_params_)
    helper.score_summary(sort_by='min_score').to_pickle('gridresult.pkl')
    print(helper.score_summary(sort_by='min_score'))

# if __name__ == '__main__':
#     sys.exit(main(sys.argv[1:]))
