#!/usr/bin/env python

"""
This is the main script for scoring a classifier either on the test set or by cross validation on training data
"""

import sys
sys.path.append('../')
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import argparse
from time import time

from src.lexicons import *

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from src.statistics import *




choices = ['svm', 'logistic_regression', 'knn', 'naive_bayes_bernoulli', 'naive_bayes_binomial', 'neural_network',
               'most_frequent', 'stratified', 'random', 'lexicon']
classifiers = [SVC(C=0.1, kernel='linear', class_weight='balanced'), LogisticRegression(class_weight='balanced'), KNeighborsClassifier(), BernoulliNB(), MultinomialNB(),
               MLPClassifier(), DummyClassifier(strategy='most_frequent'), DummyClassifier(strategy='stratified'),
               DummyClassifier(strategy='uniform'), ANEWPredictor()]
classifiers = dict(zip(choices, classifiers))

preprocessor = lambda text: preprocess(text, word_transformation='', lowercase=True)
bag_of_words_extractor = CountVectorizer(binary=True, ngram_range=(1, 1),
                                         tokenizer=lambda text: preprocess(text, word_transformation='', lowercase=True),
                                         lowercase=True)
swn_extractor = SentiWordNet_Extractor()
sentiment140_extractor_uni = Sentiment140ExtractorUnigram()
sentiment140_extractor_bi = Sentiment140ExtractorBigram()
custom_extractor = Custom_Extractor()
anew_extractor = ANEWExtractor()
compiled_lexicon_extractor = CompiledExtractor()
bingliu_extractor = BingLiuExtractor()
AFINN_extractor = AFINNExtractor()
extractors = [bag_of_words_extractor, swn_extractor, sentiment140_extractor_uni, sentiment140_extractor_bi, custom_extractor, anew_extractor,
              compiled_lexicon_extractor, bingliu_extractor, AFINN_extractor]
extractors = [bag_of_words_extractor]

def main(arguments):
    begin = time()
    parser = argparse.ArgumentParser(
        description="Train a classifier and either cross validate on training data or score on the test data",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('train', help="Input training data file in format TEXT<tab>LABEL or TEXT<tab>TOPIC<tab>LABEL",
                        type=argparse.FileType('r'))
    parser.add_argument('classifier', choices=choices,
                        help="Name of the classifier to use on the data: \n{{{}}}".format('|'.join(choices)),
                        metavar='classifier')
    parser.add_argument('-t', '--test', help="""Input test data file in same format as TRAIN. If not provided, 
                        cross validation is executed on the train data""",
                        type=argparse.FileType('r'))

    # retrieve arguments
    args = vars(parser.parse_args(arguments))
    train = args['train'].name
    test= args['test'].name if args['test'] else None
    classifier = args['classifier']

    # load data
    train_tweets, train_labels = load_data(train)
    print("Number of training tweets : {}".format(len(train_tweets)))
    # get chosen classifier
    estimator = classifiers[classifier]
    print("Classifier : {}".format(classifier))
    # create the features union and the pipeline
    features = create_feature_unions(extractors)
    pipeline = create_pipeline(features, estimator)

    # whether fit the model or not (always true excepted is lexicon is the chosen estimator
    fit=True
    # if test data is provided, then train on train and test on test
    print(plot_class_repartition(train_tweets, train_labels))
    # print(check_data_format(train_tweets, train_labels))
#     if test:
#         test_tweets, test_labels = load_data(test)
#         print("Number of testing tweets : {}".format(len(test_tweets)))
#         if classifier == 'lexicon':
#             pipeline = estimator
#             print("Lexicon is a particular classifier : it scores directly on the test set")
#             fit = False
#         training = time()
#         predictions, precision, recall, fscore, support, accuracy = evaluate_model_predict(pipeline, train_tweets,
#                                                                                            train_labels, test_tweets,
#                                                                                            test_labels, fit=fit)
#     # if not test data, perform cross validation on train data
#     else:
#         if classifier == 'lexicon':
#             pipeline = estimator
#             print("Lexicon is a particular classifier : it scores directly on the whole train set")
#             fit = False
#         training = time()
#         predictions, precision, recall, fscore, support, accuracy = evaluate_model_predict(pipeline, train_tweets,
#                                                                                        train_labels, fit=fit)
#     end = time() - begin
#     end_training = time() - training
#     # print results
#     print('Precision: {}'.format(precision))
#     print('Recall: {}'.format(recall))
#     print('Fscore: {}'.format(fscore))
#     print('Support: {}'.format(support))
#     print('Average fscore : {}'.format((fscore[0] + fscore[-1]) / 2))
#     print('Average recall : {}'.format((recall[0] + recall[-1]) / 2))
#     print("Accuracy: {mean:.3f}".format(mean=accuracy.mean()))
#     print("Done in {0:.2f}".format(end))
#     print("Training in {0:.2f}".format(end_training))
#


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
