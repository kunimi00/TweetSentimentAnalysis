# Python requistes
from collections import Counter
from scipy.sparse import csr_matrix, hstack
from tqdm import tqdm

import numpy as np
import time
import pandas as pd
import os
import re

import pickle
import matplotlib.pyplot as plt


# Graph / Visualization
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
import pygraphviz


## Tweet preprocessor
import preprocessor as p

## NLTK tokenization / lemmatization
import nltk

from nltk.tokenize import treebank
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
from nltk.corpus import sentiwordnet as swn
from nltk.corpus.reader import WordListCorpusReader
from nltk.corpus import opinion_lexicon

from nltk.wsd import lesk
from nltk.stem.wordnet import WordNetLemmatizer
import nltk.sentiment


## Better WSD library
from pywsd.lesk import simple_lesk, original_lesk, cosine_lesk, adapted_lesk
from pywsd import disambiguate
from pywsd.similarity import max_similarity


## Negation

replace_dict = {
    "don't": "do not",
    "won't": "will not",
    "didn't": "did not",
    "doesn't": "does not",
    "can't": "can not",
    "couldn't": "could not",
    "isn't": "is not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "ain't": "is not",
    "aren't": "are not",
}

def replace_word(text):
    for word in replace_dict:
        if word in text:  # Small first letter
            text = text.replace(word, replace_dict[word])
        elif word[0].title() + word[1:] in text:  # Big first letter
            text = text.replace(word[0].title() + word[1:],
                                replace_dict[word][0].title() + replace_dict[word][1:])

    return text

def neg_tagging(word_list):
    string = ' '.join(word_list)
    transformed = re.sub(r'\b(?:not|never|no)\b[\w\s]+[^\w\s]', 
           lambda match: re.sub(r'(\s+)(\w+)', r'\1NEG_\2', match.group(0)), 
           string,
           flags=re.IGNORECASE)
    
    return transformed

def negate(word_list):
    negged_sentence = neg_tagging(word_list)
    negged_tokens = negged_sentence.split()

    org_tokens = []
    token_pair_list = []
    
    for word in negged_tokens:
        negation = False
        if word.startswith('NEG_'):
            negation = True
            word = word[4:]
        token_pair_list.append((word, negation))
        org_tokens.append(word)
    
    return org_tokens, token_pair_list



## get all definition from all synsets

synset_list = list(wn.all_synsets())

all_pairs_from_definition = []

# for ss in tqdm([wn.synset('amazing.s.02'), wn.synset('good.a.01')]):
for ss in tqdm(synset_list):
    df = ss.definition()
    curr_df_pair_list = disambiguate(df, cosine_lesk)
    df_pair_txt_list = []
    for curr_df_pair in curr_df_pair_list:
        if curr_df_pair[1] is None:
            df_pair_txt_list.append(curr_df_pair)
        else:
            df_pair_txt_list.append((curr_df_pair[0], curr_df_pair[1].name()))
    all_pairs_from_definition.append((ss.name(), df_pair_txt_list))

with open('all_wn_synset_definition_da_cosine.txt', 'w') as fp:
    fp.write(all_pairs_from_definition)

pickle.dump( all_pairs_from_definition, open( "all_wn_synset_definition_da_cosine.p", "wb" ) )




# ## Prepare graphical model 
# if loadSavedGraph == False:

#     synset_list = list(wn.all_synsets())
#     wordnet_graph_synset = nx.Graph(engine='sfdp', pack=True)

#     seen = set()
#     for ss in tqdm(synset_list):
#         wordnet_graph_synset.add_node(ss.name())
#         for lm in ss.lemmas():
#             _lm = lm.name()
#             if not _lm in seen:
#                 seen.add(_lm)
#                 wordnet_graph_synset.add_node(_lm)
#             wordnet_graph_synset.add_edge(_lm, ss.name())

#     path = './wordnet_graph_synset.p'
#     nx.write_gpickle(wordnet_graph_synset, path)


#     ## new cleaned wordnet SYNSET graph
#     wordnet_graph_synset_cleaned = nx.Graph(engine='sfdp', pack=True)

#     for ss in tqdm(synset_list):
#         wordnet_graph_synset_cleaned.add_node(ss.name())

#     for ss in tqdm(synset_list):
#         nb_list = [k for m in [n for n in wordnet_graph_synset.neighbors(ss.name())] for k in wordnet_graph_synset.neighbors(m)]    
#         for nb in nb_list:
#             if wordnet_graph_synset_cleaned.has_edge(ss.name(), nb) == False:
#                 wordnet_graph_synset_cleaned.add_edge(ss.name(), nb)

#     path = './wordnet_graph_synset_cleaned.p'
#     nx.write_gpickle(wordnet_graph_synset_cleaned, path)


# else:
#     wordnet_graph_synset = nx.read_gpickle("wordnet_graph_synset.p")
#     wordnet_graph_synset_cleaned = nx.read_gpickle("wordnet_graph_synset_cleaned.p")













