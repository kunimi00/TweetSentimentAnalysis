import gensim
from gensim import utils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import string
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np

class Word2VecProvider(object):
    """
    Interface to provide word2vec interface access on top of gensim
    http://zablo.net/blog/post/twitter-sentiment-analysis-python-scikit-word2vec-nltk-xgboost
    """
    word2vec = None
    dimensions = 0

    def load(self, path_to_word2vec):
        """
        Load a word2vec embeddings file
        :param path_to_word2vec: path to the file
        :return: Nothing
        """
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(path_to_word2vec, binary=False)
        self.word2vec.init_sims(replace=True)
        self.dimensions = self.word2vec.vector_size

    def get_vector(self, word):
        """
        Get the vector associated to a given word. None if the word is not in vocabulary.
        :param word:
        :return:
        """
        if word not in self.word2vec.vocab:
            return None
        return self.word2vec.syn0norm[self.word2vec.vocab[word].index]

    def get_similarity(self, word1, word2):
        """
        Get the cosine similarity between two words. None if one of them not in the vocabulary.
        :param word1:
        :param word2:
        :return:
        """
        if word1 not in self.word2vec.vocab or word2 not in self.word2vec.vocab:
            return None

        return self.word2vec.similarity(word1, word2)


def process_for_doc2vec(tweets, labels, pos_start=0, neu_start=0, neg_start=0):
    """
    Change format of tweets and labels in annotated labels (positive_0, positive_1, neutral_0...) and list of tokens
    Necessary for TaggedDocument from gensim
    :param tweets: list of strings (tweets)
    :param labels: list of strings (labels)
    :param pos_start: a potential initial index for labeling (useful in order to stack several sources for training)
    :param neu_start: a potential initial index for labeling (useful in order to stack several sources for training)
    :param neg_start: a potential initial index for labeling (useful in order to stack several sources for training)
    :return:
    """
    positives, neutrals, negatives = [], [], []
    for i, t in enumerate(zip(tweets, labels)):
            if t[1] == 'positive':
                positives.append(t)
            if t[1] == 'neutral':
                neutrals.append(t)
            if t[1] == 'negative':
                 negatives.append(t)
    pos, neu, neg = [], [], []

    for i,t in enumerate(positives):
        tup = (t[0], t[1] + "_" + str(pos_start + i))
        pos.append(tup)
    for i,t in enumerate(neutrals):
        tup = (t[0], t[1] + "_" + str(neu_start + i))
        neu.append(tup)
    for i,t in enumerate(negatives):
        tup = (t[0], t[1] + "_" + str(neg_start + i))
        neg.append(tup)
    tweets = [t[0] for t in pos+neu+neg]
    labels = [t[1] for t in pos+neu+neg]

    strip_tweets = []
    sentences = []
    translator = str.maketrans('', '', string.punctuation)
    for t in tweets:
        strip_tweets.append(t.translate(translator).lower())
    for t in strip_tweets:
        words = t.split(' ')
        sentences.append(words)
    return sentences, labels

class LabeledLineSentence(object):
    """
    Iterator class to browse documents and labels to construct a Doc2Vec model
    """
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list
    def __iter__(self):
        """
        Iterate method that yields TaggedDocuments to train a model
        :return: TaggedDocument iterator
        """
        for idx, doc in enumerate(self.doc_list):
            yield TaggedDocument(words=doc,tags=[self.labels_list[idx]])

def doc2vec_train(tweets, labels, min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7, epochs=10):
    """
    Train a doc2vec over the tweets and labels
    :param tweets: list of strings (tweets)
    :param labels: list of strings (labels)
    :param min_count: ignore all words with total frequency lower than this
    :param window: maximum distance between the predicted word and context words used for prediction within a document
    :param size: dimensionality of the feature vectors
    :param sample: threshold for configuring which higher-frequency words are randomly downsampled
    :param negative:  if > 0, negative sampling will be used, the int for negative specifies how many “noise words”
    should be drawn (usually between 5-20). Default is 5. If set to 0, no negative sampling is used
    :param workers:  worker threads to train the model
    :param epochs: number of epochs in training
    :return: trained model
    """
    labeled_line_sentences = LabeledLineSentence(tweets, labels)
    model = Doc2Vec(min_count=min_count, window=window, size=size, sample=sample, negative=negative, workers=workers)
    model.build_vocab(labeled_line_sentences)
    model.train(labeled_line_sentences, total_examples=len(list(labeled_line_sentences)), epochs=epochs)
    return model


# class Doc2VecExtractor(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.labels = None
#
#     def transform(self, doc, y=None):
#         sentences, labels = process_for_doc2vec(doc, self.labels)
#         it = LabeledLineSentence(sentences, labels)
#         model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)
#         model.build_vocab(it)
#         model.train(it, total_examples=len(list(it)), epochs=10)
#         train_arrays = np.zeros((len(labels), 100))
#
#         for i, label in enumerate(labels):
#             train_arrays[i] = model.docvecs[label]
#         print(train_arrays.shape)
#         return train_arrays
#
#     def fit(self, X, y=None):
#         self.labels = y
#         return self