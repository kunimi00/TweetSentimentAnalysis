from nltk.corpus import wordnet as wn
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, recall_score
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import VectorizerMixin
import pandas as pd
from scipy.sparse import issparse
import numpy as np
import random
import dill as pickle

def get_wordnet_pos(treebank_tag):
    """
    Converts treebank pos tags to wordnet pos tags
    :param treebank_tag:
    :return:
    """
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None

def load_data(filename):
    """
    Load data
    :param filename: file path
    :return: tweets, labels
    """
    file = open(filename, "r")
    tweets, labels = [], []
    format = 'A' if len(file.readline().split('\t')) == 2 else 'B'
    if format == 'A':
        for line in file:
            split_string = line.split("\t")
            if len(split_string) == 2:
                tweets.append(split_string[0])
                labels.append(split_string[1].replace('\n', ''))
    else:
        topics = []
        for line in file:
            split_string = line.split("\t")
            if len(split_string) == 3:
                tweets.append(split_string[0])
                topics.append(split_string[1])
                labels.append(split_string[2].replace('\n', ''))

    return tweets, labels

def check_data_format(tweets, labels=None):
    if labels == None:
        if not bool(tweets) and isinstance(tweets, list) and all(isinstance(elem, str) for elem in tweets):
            raise ValueError('Incorrect format of tweets. Must be list of strings')
        else:
            return True
    elif (len(labels)==0) or (not bool(tweets) and isinstance(tweets, list) and all(isinstance(elem, str) for elem in tweets)\
    and bool(labels) and isinstance(labels, list) and all(isinstance(elem, str) for elem in labels)):
        raise ValueError('Incorrect format of tweets and labels. Must be both list of strings')
    else:
        return True

def check_transformer(transformer):
    """
    Verify that the object inherits from scikit transformer to be compatible in model fitting
    :param transformer: object to check
    :return: True or raises a ValueError
    """
    if not issubclass(type(transformer), TransformerMixin) and not issubclass(type(transformer), VectorizerMixin):
        raise ValueError('Intermediate steps or FeatureUnion steps must be scikit transformers !')
    else:
        return True

def create_feature_unions(transformer_list):
    """
    Union (stack) different feature extraction
    Each transformer must return arrays of same length than training samples in order to stack successfully
    :param transformer_list: list of tuples (name, features extractor) or list of features extractors
    :return: scikit FeatureUnion
    """
    if isinstance(transformer_list, list):
        for i, transformer in enumerate(transformer_list):
            if not isinstance(transformer, tuple):
                check_transformer(transformer)
                transformer_list[i] = ('feature_default' + str(i), transformer)
            else:
                check_transformer(transformer[1])
    else:
        raise ValueError('You must provide a list of transformers')
    return FeatureUnion(transformer_list)

def create_pipeline(features_union, estimator, intermediate=None ):
    """
    A typical machine learning pipeline consists in extracting features, combining them and feeding to a classifier.
    Very useful for avoiding bias in cross validation
    :param features_union: a list in scikit format for concatenating features
    :param intermediate: intermediate step between features and estimator (can be normalizing, scaling...)
    :param estimator: final estimator of the pipeline
    :return: scikit Pipeline
    """
    if intermediate:
        if isinstance(intermediate, list):
            for i, transformer in enumerate(intermediate):
                if not isinstance(transformer, tuple):
                    check_transformer(transformer)
                    intermediate[i] = ('intermediate_default' + str(i), transformer)
                else:
                    check_transformer(transformer[1])
        else:
            check_transformer(intermediate)
            pipeline = Pipeline([('features', features_union), ('intermediate_steps', intermediate), ('estimator', estimator)])
    else:
        pipeline = Pipeline([('features', features_union), ('estimator', estimator)])

    return pipeline

def evaluate_model_predict(pipeline, train, train_labels, test = None, test_labels = None, fit=True,
                           folds=StratifiedKFold(n_splits=3, shuffle=True, random_state=None)):
    """
    Either cross validate the model on train data, either train ont train and predict on test.
    :param pipeline: scikit pipeline
    :param train: train data
    :param train_labels: train data labels
    :param test: test data. If not, model is cross valdiated on train.
    :param test_labels: test data labels
    :param fit: whether to fit the model or not. If not, pipeline argument must implement a predict(x,y) method w/o fitting.
    :param cv: folding object, default is a stratified split randomly shuffled
    :return: predictions, precision, recall, fscore, accuracy at a micro level over the folds.
    Also returns support, which is the repartition of samples over classes.
    """

    if not test:
        if fit:
            predictions = cross_val_predict(pipeline, train, train_labels, cv=folds)
        else:
            predictions = pipeline.predict(train, train_labels)
        precision, recall, fscore, support = precision_recall_fscore_support(train_labels, predictions)
        accuracy = accuracy_score(train_labels, predictions)
    else:
        if fit:
            model = train_model(pipeline, train, train_labels)
            predictions = predict(model, test)
        else:
            predictions = pipeline.predict(test, test_labels)
        precision, recall, fscore, support = precision_recall_fscore_support(test_labels, predictions)
        accuracy = accuracy_score(test_labels, predictions)

    return predictions, precision, recall, fscore, support, accuracy

def evaluate_model_score(pipeline, train, labels, cv=StratifiedKFold(n_splits=3, shuffle=True,
                                                                      random_state=None), score = 'accuracy'):
    """
    Cross validate score the model and print score
    :param pipeline: scikit pipeline
    :param train: train data
    :param labels: train data labels
    :param cv: folding object, default is a stratified split randomly shuffled
    :return: scores
    """
    scores = cross_val_score(pipeline, train, labels, cv, scoring=score)

    return scores

def train_model(pipeline, train, train_labels, file='model.pkl'):
    """
    Train a model (usually a pipeline)
    :param pipeline: model, must have an scikit estimator
    :param train: train data
    :param labels: train labels
    :param file: file name to save the trained model
    :return: trained model
    """
    pipeline.fit(train, train_labels)

    try:
        with open(file, 'wb') as f:
            pickle.dump(pipeline, f)
    except pickle.PicklingError:
        print("Model could not be saved")
    return pipeline

def load_model(file='model.pkl'):
    """
    Load a pretrained model
    :param file: pickle file
    :return: model
    """
    try:
        with open(file, 'rb') as f:
            model = pickle.load(f)
    except pickle.PicklingError:
        print("Model could not be loaded")
    return model

def predict(model, test):
    """
    Predict labels with a model. Model must be already trained.
    :param model: trained model
    :param test: test data
    :return: predictions
    """
    try:
        predictions = model.predict(test)
        return predictions
    except ValueError:
        raise ValueError('Model has not been fitted or cannot predict')

def convert_to_dataframe(X, y=None, columns=None):
    """
    Converts a bag of words (or any X matrix) into a pandas dataframe for easier manipulation.
    :param X: sparse matrix or numpy array
    :param y: labels
    :return: pandas DataFrame or SparseDataFrame
    """
    dataframe = None
    if issparse(X):
        dataframe = pd.SparseDataFrame(X, columns=columns).fillna(0)
    else:
        try:
            dataframe = pd.DataFrame(X, columns=columns).fillna(0)
        except ValueError:
            print("X must be either numpy array or dict or DataFrame")
    if y:
        dataframe['label'] = y

    return dataframe

def inc(*args):
    """
    Increment by one all args. Useful for multiple increments
    :param args: values to increment
    :return: incremented values
    """
    for i in args:
        yield i+1

def list_to_numpy_vector(lists):
    """
    Convert a list to numpy vectors of dim (n,1), useful for stacking
    :param lists: either a lists of list or a list of numbers
    :return: list of numpy vectors or one vector
    """
    vectors = []
    if isinstance(lists, list):
        if all(isinstance(elem, list) for elem in lists):
            for l in lists:
                vectors.append(np.array(l)[:,np.newaxis])
        else:
            vectors = np.array(lists)[:,np.newaxis]
        return vectors
    else:
        raise ValueError("object must be a list")

def shuffle_pairs(l1, l2):
    """
    Zip and shuffle two lists
    :param l1: first list
    :param l2: second list
    :return: shuffled lists
    """
    zipped = list(zip(l1, l2))
    random.shuffle(zipped)
    l1, l2 = zip(*zipped)

    return l1, l2

def f1_average(y_true, y_pred):
    """
    Return macro f1 but over negative and positive classes only.
    Assumes that negative class is first in the set of labels (sorted(set(...)).
    Can be applied to a binary classification, thus equivalent to a macro averaged f1_score :
    sklearn f1_score(average='macro'). !If labels have been collected by cross validation then the f1 is micro averaged
    across the folds (but still macro averaged across classes).
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: f1_score macro averaged
    """
    class_labels = sorted(set(y_true))
    score = f1_score(y_true, y_pred, labels=class_labels, average=None)
    return (score[0] + score[-1])/2

def recall_average(y_true, y_pred):
    """
    Return average recall but over negative and positive classes only.
    Assumes that negative class is first in the set of labels (sorted(set(...)).
    Can be applied to a binary classification, thus equivalent to a macro averaged recall :
    sklearn recall_score(average='macro'). !If labels have been collected by cross validation then the f1 is micro
    averaged across the folds (but still macro averaged across classes).
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: recall_score averaged
    """
    class_labels = sorted(set(y_true))
    score = recall_score(y_true, y_pred, labels=class_labels, average=None)
    return (score[0] + score[-1])/2

def macro_MAE(y_true, y_predicted, classes=[-2, -1, 0, 1, 2]):
    """
    Calculate macro averaged error across classes
    :param y_true:  true labels
    :param y_predicted: predicted labels
    :param classes: classes to consider (must be numerical)
    :return: MAE
    """
    y_true, y_predicted = np.array(y_true), np.array(y_predicted)
    indexes = {i:np.where(y_true==i)[0] for i in classes}
    error = 0
    for i in classes:
        diffs = np.subtract(y_true[indexes[i]], y_predicted[indexes[i]])
        diffs = np.absolute(diffs)
        macro_error = np.sum(diffs)/float(len(diffs))
        error += macro_error
    return error/float(len(classes))