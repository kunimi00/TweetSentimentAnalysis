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


from multiprocessing import Process, Queue
import pickle


def getDisambiguatedList(tweet_list):
    score_list = []
    for tw in tqdm(tweet_list):
        score_list.append(GetDisambiguation(tw))
    return score_list

def doDisambiguation(q, l):
    q.put(getDisambiguatedList(l))


da_pair_list = []
tweet_partition_num = len(tweets)//2

q = Queue()

p1 = Process(target=doDisambiguation, args=(q, tweets[:tweet_partition_num]))
p2 = Process(target=doDisambiguation, args=(q, tweets[tweet_partition_num:]))
p1.start()
p2.start()
p1.join()
p2.join()
r1 = q.get()
r2 = q.get()



da_pair_list.extend(r1)
da_pair_list.extend(r2)


with open('./da_pair_list.p', 'wb') as fp:
    pickle.dump(da_pair_list, fp)

with open('./da_pair_list.p', 'rb') as fp:
    loaded_pair = pickle.load(fp)

print(len(loaded_pair))
print('Done')







# from multiprocessing import Process, Queue

# def do_sum(q,l):
#     q.put(sum(l))

# def main():
#     my_list = range(1000000)

#     q = Queue()

#     p1 = Process(target=do_sum, args=(q,my_list[:500000]))
#     p2 = Process(target=do_sum, args=(q,my_list[500000:]))
#     p1.start()
#     p2.start()
#     r1 = q.get()
#     r2 = q.get()
#     print r1+r2

# if __name__=='__main__':
#     main()



