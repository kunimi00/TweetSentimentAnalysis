from tqdm import tqdm
from src.util import *
from src.processing import *
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from nltk.corpus import sentiwordnet as swn, opinion_lexicon
import numpy as np
from scipy.sparse import csr_matrix
import csv

anew_lexicon = "../lexicons/ANEW.csv"
sentiment140_unigrams = "../lexicons/unigrams-pmilexicon.txt"
sentiment140_bigrams = "../lexicons/bigrams-pmilexicon.txt"
compiled_lexicon = "../lexicons/compiled.csv"
AFINN_lexicon = "../lexicons/AFINN.txt"
AFINN_emoticons = "../lexicons/AFINN_emoticons.txt"
# Transformators used to extract features using different sentiment lexicons

class SentiWordNet_Extractor(BaseEstimator, TransformerMixin):
    """
    Extract features from a text dataset based on SentiWordNet lexicon recensing score of words (neg, obj, pos)
    """
    def __init__(self):
        self.columns = ["SentiWordNet_positive_score", "SentiWordNet_neutral_score", "SentiWordNet_negative_score"]

    def transform(self, doc, y=None):
        """
        Return a matrix/vector (len(doc), 4) of features. Scores are naively summed.
        :param doc: list of strings
        :param y: labels (useless)
        :return: sparse matrix of scores
        """
        check_data_format(doc)
        positive, negative, neutral = [], [], []
        lines = [preprocess(line, word_transformation='', handle_negation=False, lowercase=True) for line in doc]
        for line in tqdm(lines):
            pos, neg, neu = 0, 0, 0
            tagged = pos_tag(line)
            for word, tag in tagged:
                wntag = get_wordnet_pos(tag)
                synsets = wn.synsets(word, pos=wntag)
                if not synsets:
                    continue
                synset = synsets[0]
                swn_synset = swn.senti_synset(synset.name())
                pos_score = swn_synset.pos_score()
                neu_score = swn_synset.obj_score()
                neg_score = swn_synset.neg_score()
                pos += pos_score
                neg += neg_score
                neu += neu_score
            positive.append(pos)
            negative.append(neg)
            neutral.append(neu)
        features = list_to_numpy_vector([positive, neutral, negative])
        return csr_matrix(np.hstack(features))

    def fit(self, X, y=None):
        return self


class BingLiuExtractor(BaseEstimator, TransformerMixin):
    """
    Extract features from a text dataset based on the BingLiu opinion lexicon recensing negative/positive words
    """
    def __init__(self):
        self.opinion_lexicon = opinion_lexicon
        self.columns = ["Bing_Liu_pos_pos", "Bing_Liu_neg_pos", "Bing_Liu_pos_neg", "Bing_Liu_neg_neg"]

    def transform(self, doc, y=None):
        """
        Return a matrix/vector (len(doc), 4) of features. Counts are extracted according to negated contexts.
        :param doc: list of strings
        :param y: labels (useless)
        :return: sparse matrix of counts
        """
        check_data_format(doc)
        pos_positives, neg_positives, pos_negatives, neg_negatives = [], [], [], []
        olp = set(self.opinion_lexicon.positive())
        oln = set(self.opinion_lexicon.negative())
        lines = [preprocess(line, word_transformation='lemmatize', lowercase=True) for line in doc]
        for line in tqdm(lines):
            pos_pos, pos_neg, neg_neg, neg_pos = 0, 0, 0, 0
            for word in line:
                if word.endswith("_NEG"):
                    w = word.strip('_NEG')
                    if w in olp:
                        pos_neg += 1
                    elif w in oln:
                        neg_neg += 1
                else:
                    if word in olp:
                        pos_pos += 1
                    elif word in oln:
                        neg_pos += 1
            pos_positives.append(pos_pos)
            neg_positives.append(neg_pos)
            pos_negatives.append(pos_neg)
            neg_negatives.append(neg_neg)
        features = list_to_numpy_vector([pos_positives, neg_positives, pos_negatives, neg_negatives])
        return csr_matrix(np.hstack(features))

    def fit(self, X, y=None):
        return self

from itertools import tee
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

class Sentiment140ExtractorUnigram(BaseEstimator, TransformerMixin):
    """
    Extract features from a text dataset based on the Sentiment140 lexicon recensing polarity score of unigrams
    """
    def __init__(self):
        self.columns = ["Sentiment140_uni_pos_pos", "Sentiment140_uni_neg_pos", "Sentiment140_uni_pos_neg",
                        "Sentiment140_uni_neg_neg"]
        self.unigrams = pd.read_csv(sentiment140_unigrams, sep='\t', names=['unigram', 'score', 'o1', 'o2']).drop(
            ["o1", "o2"], axis=1)
        self.words_uni = pd.Series(self.unigrams.score.values, index=self.unigrams.unigram).to_dict()

    def transform(self, doc, y=None):
        """
        Return a matrix/vector (len(doc), 4) of features. Counts are extracted according to negated contexts.
        :param doc: list of strings
        :param y: labels (useless)
        :return: sparse matrix of counts
        """
        check_data_format(doc)
        pos_positives, pos_negatives, neg_negatives, neg_positives = [], [], [], []
        lines = [preprocess(line, word_transformation='', lowercase=True) for line in doc]
        for line in tqdm(lines):
            pos_pos, pos_neg, neg_neg, neg_pos = 0, 0, 0, 0
            for word in line:
                if word.endswith("_NEG"):
                    w = word.strip('_NEG')
                    if w in self.words_uni:
                        score = self.words_uni[w]
                        if score > 0:
                            pos_neg += 1
                        else:
                            neg_neg += 1
                elif word in self.words_uni:
                    score = self.words_uni[word]
                    if score >0:
                        pos_pos += 1
                    else:
                        neg_pos += 1

            pos_positives.append(pos_pos)
            neg_positives.append(neg_pos)
            pos_negatives.append(pos_neg)
            neg_negatives.append(neg_neg)

        features = list_to_numpy_vector([pos_positives, neg_positives, pos_negatives, neg_negatives])
        return csr_matrix(np.hstack(features))


    def fit(self, X, y=None):
        return self


class Sentiment140ExtractorBigram(BaseEstimator, TransformerMixin):
    """
    Extract features from a text dataset based on the Sentiment140 lexicon recensing polarity score of bigrams
    """
    def __init__(self):
        self.columns = ["Sentiment140_bi_pos_pos", "Sentiment140_bi_neg_pos", "Sentiment140_bi_pos_neg",
                        "Sentiment140_bi_neg_neg"]
        self.bigrams = pd.read_csv(sentiment140_bigrams, sep='\t', names=['bigram', 'score', 'o1', 'o2'], quoting=csv.QUOTE_NONE).drop(
            ["o1", "o2"], axis=1)
        self.words_bi = pd.Series(self.bigrams.score.values, index=self.bigrams.bigram).to_dict()

    def transform(self, doc, y=None):
        """
        Return a matrix/vector (len(doc), 4) of features. Counts are extracted according to negated contexts.
        :param doc: list of strings
        :param y: labels (useless)
        :return: sparse matrix of counts
        """
        check_data_format(doc)
        pos_positives, pos_negatives, neg_negatives, neg_positives = [], [], [], []
        lines = [preprocess(line, word_transformation='', lowercase=True) for line in doc]
        for line in tqdm(lines):
            pos_pos, pos_neg, neg_neg, neg_pos = 0, 0, 0, 0

            for w1, w2 in pairwise(line):
                w = ' '.join([w1.strip('_NEG'), w2.strip('_NEG')])
                if w1.endswith("_NEG") or w2.endswith("_NEG"):
                    if w in self.words_bi:
                        score = self.words_bi[w]
                        if score > 0:
                            pos_neg += 1
                        else:
                            neg_neg += 1
                elif w in self.words_bi:
                    score = self.words_bi[w]
                    if score >0:
                        pos_pos += 1
                    else:
                        neg_pos += 1


            pos_positives.append(pos_pos)
            neg_positives.append(neg_pos)
            pos_negatives.append(pos_neg)
            neg_negatives.append(neg_neg)

        features = list_to_numpy_vector([pos_positives, neg_positives, pos_negatives, neg_negatives])
        return csr_matrix(np.hstack(features))


    def fit(self, X, y=None):
        return self


class ANEWExtractor(BaseEstimator, TransformerMixin):
    """
    Extract features from a text dataset based on the ANEW lexicon recensing polarity valence ([1,9]) of 13,915 words
    Negative : valence in [1,4[
    Neutral : valence in [4,6]
    Positive : valence in ]6,9]
    """
    def __init__(self):
        self.lexicon = pd.read_csv(anew_lexicon).drop("id", axis=1)
        self.words = pd.Series(self.lexicon["V.Mean.Sum"].values, index=self.lexicon.Word).to_dict()
        self.columns = ["ANEW_pos_pos", "ANEW_pos_neg", "ANEW_neutral", "ANEW_neg_pos", "ANEW_neg_neg"]

    def transform(self, doc, y=None):
        """
        Return a matrix/vector (len(doc), 5) of features. Counts are extracted according to negated contexts.
        :param doc: list of strings
        :param y: labels (useless)
        :return: sparse matrix of counts
        """
        check_data_format(doc)
        neutral = []
        positive_pos, negative_pos, positive_neg, negative_neg = [], [], [], []
        for line in doc:
            neu = 0
            pos_pos, neg_pos, pos_neg, neg_neg = 0, 0, 0, 0
            pos_counts = Counter()
            tokens = preprocess(line, lowercase=True, handle_negation=True, word_transformation='lemmatize')
            gen = (token for token in tokens if token.strip('_NEG') in self.words)
            for token in gen:

                score = self.words[token.strip('_NEG')]
                if token.endswith('_NEG'):
                    if score > 6:
                        pos_neg += 1
                    elif score < 4:
                        neg_neg += 1
                    else:
                        neu += 1
                else:
                    if score > 6:
                        pos_pos += 1
                    elif score < 4:
                        neg_pos += 1
                    else:
                        neu += 1

            neutral.append(neu)
            positive_pos.append(pos_pos)
            positive_neg.append(pos_neg)
            negative_pos.append(neg_pos)
            negative_neg.append(neg_neg)


        features = list_to_numpy_vector([positive_pos, positive_neg, neutral, negative_pos, negative_neg])
        return csr_matrix(np.hstack(features))

    def fit(self, X, y=None):
        return self


class CompiledExtractor(BaseEstimator, TransformerMixin):
    """
    Extract features from a text dataset based on a compiled lexicon (mpqa, bingliu...) of positive and negative words
    """
    def __init__(self):
        self.lexicon = pd.read_csv(compiled_lexicon, encoding='latin1').fillna(0)
        self.words = pd.Series(self.lexicon.sentiment.values, index=self.lexicon.word).to_dict()
        self.columns = ["Compiled_positive", "Compiled_neutral", "Compiled_negative", "Compiled_NOUN_pos",
                        "Compiled_NOUN_neg", "Compiled_VERB_pos", "Compiled_VERB_neg", "Compiled_ADJ_pos",
                        "Compiled_ADJ_neg", "Compiled_ADV_pos", "Compiled_ADV_neg"]
    def transform(self, doc, y=None):
        """
        Return a matrix/vector (len(doc), 11) of features
        In negated contexts, counts are reversed. Counts by pos tag are also extracted
        :param doc: list of strings
        :param y: labels (useless)
        :return: sparse matrix of counts
        """
        check_data_format(doc)
        negative, positive, neutral = [], [], []
        NOUN_pos, NOUN_neg, VERB_pos, VERB_neg, ADJ_pos, ADJ_neg, ADV_pos, ADV_neg = [], [], [], [], [], [], [], []
        for line in doc:
            pos, neg, neu = 0, 0, 0
            pos_counts = Counter()
            tokens = preprocess(line, lowercase=True)
            gen = (tup for tup in pos_tag(tokens) if tup[0].strip('_NEG') in self.words)
            for token, tag in gen:
                wntag = get_wordnet_pos(tag)
                if token.endswith('_NEG'):
                    if self.words[token.strip('_NEG')] == 'positive':
                        neg += 1
                        if wntag:
                            pos_counts[wntag + '_neg'] += 1
                    elif self.words[token.strip('_NEG')] == 'negative':
                        pos += 1
                        if wntag:
                            pos_counts[wntag + '_pos'] += 1
                    else:
                        neu += 1
                else:
                    if self.words[token.strip('_NEG')] == 'positive':
                        pos += 1
                        if wntag:
                            pos_counts[wntag + '_pos'] += 1
                    elif self.words[token.strip('_NEG')] == 'negative':
                        neg += 1
                        if wntag:
                            pos_counts[wntag + '_neg'] += 1
                    else:
                        neu += 1
            negative.append(neg)
            positive.append(pos)
            neutral.append(neu)
            NOUN_pos.append(pos_counts['n_pos'])
            NOUN_neg.append(pos_counts['n_neg'])
            VERB_pos.append(pos_counts['v_pos'])
            VERB_neg.append(pos_counts['v_neg'])
            ADJ_pos.append(pos_counts['a_pos'])
            ADJ_neg.append(pos_counts['a_neg'])
            ADV_pos.append(pos_counts['r_pos'])
            ADV_neg.append(pos_counts['r_neg'])
        features = list_to_numpy_vector([positive, neutral, negative, NOUN_pos, NOUN_neg, VERB_pos, VERB_neg, ADJ_pos,
                                         ADJ_neg, ADV_pos, ADV_neg])
        return csr_matrix(np.hstack(features))

    def fit(self, X, y=None):
        return self

class AFINNExtractor(BaseEstimator, TransformerMixin):
    """
    Extract a score feature from a text dataset with the AFINN lexicon
    """
    def __init__(self):
        self.lexicon = pd.read_csv(AFINN_lexicon, sep='\t', names=['word', 'score'], encoding='latin1')
        self.emoticons = pd.read_csv(AFINN_emoticons, sep='\t', names=['emoticon', 'score'], encoding='latin1')
        self.columns = ["AFINN_score"]
        self.words = pd.Series(self.lexicon.score.values, index=self.lexicon.word).to_dict()
        self.emoticons = pd.Series(self.emoticons.score.values, index=self.emoticons.emoticon).to_dict()

    def transform(self, doc, y=None):
        """
        Return a matrix/vector (len(doc), 1) of scores
        :param doc: list of strings
        :param y: labels (useless)
        :return: sparse vector of scores
        """
        check_data_format(doc)
        scores = []
        for line in doc:
            score= 0
            tokens = preprocess(line, lowercase=True, handle_negation=True, word_transformation='')
            gen_lexicon = (token for token in tokens if token.strip('_NEG') in self.words)
            gen_emoticon = (token for token in tokens if token.strip('_NEG') in self.emoticons)
            for token in gen_lexicon:
                if token.endswith('_NEG'):
                    score+= -self.words[token.strip('_NEG')]
                else:
                    score+=self.words[token.strip('_NEG')]
            for token in gen_emoticon:
                score += self.emoticons[token.strip('_NEG')]

            scores.append(score)
        features = list_to_numpy_vector([scores])
        return csr_matrix(np.hstack(features))

    def fit(self, X, y=None):
        return self


class ANEWPredictor:
    """
    Naive class predicting labels based on ANEW lexicon score
    """
    def __init__(self):
        self.lexicon = pd.read_csv(anew_lexicon).drop("id", axis=1)
        self.words = pd.Series(self.lexicon["V.Mean.Sum"].values, index=self.lexicon.Word).to_dict()

    def predict(self, tweets, y):
        """
        Predict the labels of a tweets dataset by naively defining the label of a tweet as the most represented count
        of words for each category
        :param tweets: list of tweets (strings)
        :param y: list of labels (strings)
        :return:
        """
        check_data_format(tweets, y)
        labels = []
        lab = np.array(sorted(set(y)))
        for line in tweets:
            pos, neg, neu  = 0, 0, 0
            tokens = preprocess(line, lowercase=True, handle_negation=True, word_transformation='lemmatize')
            gen = (token for token in tokens if token.strip('_NEG') in self.words)
            for token in gen:
                score = self.words[token.strip('_NEG')]
                if score > 6:
                    pos += 1
                elif score < 4:
                    neg += 1
                else:
                    neu += 1
            max_index = np.argmax([neg, neu, pos])
            labels.append(lab[max_index])

        return labels

    def fit(self, X, y=None):
        return self

