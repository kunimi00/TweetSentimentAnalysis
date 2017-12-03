import re
from nltk import pos_tag, WordNetLemmatizer, PorterStemmer
from nltk.sentiment.util import mark_negation, NEGATION_RE
from scipy.sparse import csr_matrix
from src.util import *
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter


emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose 
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
mention_RE = r'(?:@[\w_]+)'
hashtag_RE = r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"
url_RE = r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'
numbers_RE = r'(?:(?:\d+,?)+(?:\.?\d+)?)'
dashapostrophe_RE = r"(?:[a-z][a-z'\-_]+[a-z])"  # words with - and '
words_RE = r'(?:[\w_]+)' # other words
anything_RE = r'(?:\S)' # anything else

regex_str = [
    emoticons_str,
    mention_RE,
    hashtag_RE,
    url_RE,
    numbers_RE,
    dashapostrophe_RE,
    words_RE,
    anything_RE
]

tokens_RE = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_RE = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)

normaleyes_RE = r'[:=]'
nosearea_RE = r'(|o|O|-)'  ## rather tight precision, \S might be reasonable...
happymouths_RE = r'[D\)\]]'
sadmouths_RE = r'[\(\[]'
happy_RE = re.compile('(\^_\^|' + normaleyes_RE + nosearea_RE + happymouths_RE + ')')
sad_RE = re.compile(normaleyes_RE + nosearea_RE + sadmouths_RE)
mention_RE = re.compile(mention_RE)
url_RE = re.compile(url_RE)
hashtag_RE = re.compile(hashtag_RE)
dotdotdot_RE = re.compile(r"\.\s?\.\s?\.")
elongated_RE = re.compile(r"(\w)\1{2}")
negation_RE = NEGATION_RE

def tokenize(s):
    """
    Tokenize text according to regex, and clean hashtags and urls
    :param s: sentence to tokenize
    :return: tokens
    """
    s = re.sub(r'(?:@[\w_]+)', '@mention', s)
    s = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', 'http://url', s)
    tokens = tokens_RE.findall(s)
    return tokens

def preprocess(s, lowercase=False, tokenizer=tokenize, word_transformation = '', handle_negation = True):
    """
    Improve tokenization with different options
    :param s: sentence to tokenize
    :param lowercase: lowercase or not tokens
    :param tokenizer: which tokenizer to use
    :param word_transformation: stemming, lemmatize or nothing
    :param handle_negation:
    :return: tokens
    """
    tokens = tokenizer(s)
    if word_transformation == 'stem':
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    elif word_transformation == 'lemmatize':
        lemmatizer = WordNetLemmatizer()
        tagged = pos_tag(tokens)
        tokens = []
        for word, tag in tagged:
            wntag = get_wordnet_pos(tag)
            if wntag is None:
                lemma = lemmatizer.lemmatize(word)
            else:
                lemma = lemmatizer.lemmatize(word, pos=wntag)
            tokens.append(lemma)
    if lowercase:
        tokens = [token if emoticon_RE.search(token) else token.lower() for token in tokens]
    if handle_negation:
        tokens = mark_negation(tokens)
    return tokens


class Custom_Extractor(BaseEstimator, TransformerMixin):
    """
    Extract handmade features from text
    """
    def __init__(self):
        self.columns = ['# exclamations', '# interrogations', '# positive_emoticons', '# negative_emoticons',
                        '# uppercases', '# dotdotdot', '# quotations', '# hashtags',
                        '# elongated', '# negated', '# NOUN', ' # VERB', '# ADJ', '# ADV']

    def transform(self, doc, y=None):
        """
        Transforms a list of document into a matrix of features
        :param doc: list of strings
        :param y: labels (useless here)
        :return: sparse matrix of features
        """
        exclamations = []
        interrogations = []
        positive_emoticons = []
        negative_emoticons = []
        uppercases = []
        dotdotdot = []
        quotations = []
        mentions = []
        hashtags = []
        urls = []
        elongated = []
        negated = []
        NOUN, VERB, ADJ, ADV = [], [], [], []

        for line in doc:
            tags = Counter()
            neg = 0
            tokens = tokenize(line)
            tagged = pos_tag(tokens)
            for word, tag in tagged:
                if NEGATION_RE.search(word):
                    neg += 1
                wntag = get_wordnet_pos(tag)
                if wntag:
                    tags[wntag] += 1
            NOUN.append(tags[wn.NOUN])
            VERB.append(tags[wn.VERB])
            ADJ.append(tags[wn.ADJ])
            ADV.append(tags[wn.ADV])

            negated.append(neg)
            count = Counter(c if c in ['!', '?', '"'] else 'uppercase' if c.isupper() else None for c in line)
            exclamations.append(count['!'])
            interrogations.append(count['?'])
            quotations.append(count['"'] / 2)
            uppercases.append(count['uppercase'])

            positive_emoticons.append(len(happy_RE.findall(line)))
            negative_emoticons.append(len(sad_RE.findall(line)))
            dotdotdot.append(len(dotdotdot_RE.findall(line)))
            mentions.append(len(mention_RE.findall(line)))
            hashtags.append(len(hashtag_RE.findall(line)))
            urls.append(len(url_RE.findall(line)))
            elongated.append(len(elongated_RE.findall(line)))

            features = list_to_numpy_vector([exclamations, interrogations, positive_emoticons, negative_emoticons,
                                             uppercases, dotdotdot, quotations, hashtags, elongated, negated, NOUN,
                                             VERB, ADJ, ADV])
        return csr_matrix(np.hstack(features))

    def fit(self, X, y=None):
        return self

