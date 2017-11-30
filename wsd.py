# Python requistes
from collections import Counter
from scipy.sparse import csr_matrix, hstack
from tqdm import tqdm

import numpy as np
import time
import pandas as pd
import os
import re

import matplotlib.pyplot as plt


# Graph / Visualization
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
import pygraphviz


# Tweet preprocessor
import preprocessor as p

# NLTK tokenization / lemmatization
import nltk

from nltk.tokenize import treebank
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.corpus.reader import WordListCorpusReader
from nltk.corpus import opinion_lexicon

from nltk.wsd import lesk

from nltk.stem.wordnet import WordNetLemmatizer

import nltk.sentiment


# Better WSD library
from pywsd.lesk import simple_lesk, original_lesk, cosine_lesk, adapted_lesk
from pywsd import disambiguate
from pywsd.similarity import max_similarity

file_path = "./dataset/semEval_train_2016/semeval_train_A.txt"

file = open(file_path,"r")

labels = []
tweets = []
subjects = []

for line in file:
    split_string = line.split("\t")
    if len(split_string) == 2:
        tweets.append(split_string[0])
        # subjects.append(split_string[1])
        labels.append(split_string[1])

print(len(tweets))
print(len(labels))
print(tweets[0])



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

    tokens = []
    token_pair_list = []
    
    for word in negged_tokens:
        negation = False
        if word.startswith('NEG_'):
            negation = True
            word = word[4:]
        token_pair_list.append((word, negation))
        tokens.append(word)
    
    return tokens, token_pair_list

                
def GetDisambiguation(tweet_sentence):
    cleaned_tweet = p.clean(tweet_sentence)
    replaced_tweet = replace_word(cleaned_tweet)
    
    ## Can replace this by using other WSD options (different Lesk algorithms / similarity options)
    # da_token_pair_list = disambiguate(replaced_tweet, max_similarity, similarity_option='res')
    da_token_pair_list = disambiguate(replaced_tweet, cosine_lesk)
    
    return da_token_pair_list



# da_pair_list = []
# # for i in tqdm(range(len(tweets))):

# for i in tqdm(range(len(tweets))):
#     start = time.time()
#     da_pair_list.append(GetDisambiguation(tweets[i]))
#     print(time.time()-start)

# import pickle

# with open('./da_pair_list.p', 'wb') as fp:
#     pickle.dump(SentiGraphFeature, fp)

# with open('./da_pair_list.p', 'rb') as fp:
# 	loaded_pair = pickle.load(fp)

# print(len(loaded_score))
# print('Done')


from multiprocessing import Process, Queue, Manager
import pickle

manager = Manager()
result_q = Queue()

def doDisambiguation(num, l):
    da_list = []
    for tw in tqdm(l):
        da_list.append(GetDisambiguation(tw))
    # result_q.put(str(da_list))
    print(str(da_list))
    with open(str(num) + '_list.txt', 'wb') as fp:
        fp.write(str(da_list))
    print(str(num) + ' : file saved')



da_pair_list = []

p1 = Process(target=doDisambiguation, args=(1, tweets[:2000]))
p2 = Process(target=doDisambiguation, args=(2, tweets[2000:4000]))
p3 = Process(target=doDisambiguation, args=(3, tweets[4000:6000]))
p4 = Process(target=doDisambiguation, args=(4, tweets[6000:8000]))
p5 = Process(target=doDisambiguation, args=(5, tweets[8000:10000]))
p6 = Process(target=doDisambiguation, args=(6, tweets[10000:]))

p1.start()
p2.start()
p3.start()
p4.start()
p5.start()
p6.start()

# for _ in range(2):
#     da_pair_list.append(q.get())
#     print('got results2')

p1.join()
p2.join()
p3.join()
p4.join()
p5.join()
p6.join()


# da_pair_list.append(result_q.get())
# da_pair_list.append(result_q.get())
# da_pair_list.append(result_q.get())
# da_pair_list.append(result_q.get())
# da_pair_list.append(result_q.get())
# da_pair_list.append(result_q.get())

# with open('./da_pair_list.p', 'wb') as fp:
#     pickle.dump(da_pair_list, fp)

# with open('./da_pair_list.p', 'rb') as fp:
#     loaded_pair = pickle.load(fp)

# print(len(loaded_pair))
print('Done')

