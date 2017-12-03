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

    token_negation_pair_list = []
    
    for word in negged_tokens:
        negation = False
        if word.startswith('NEG_'):
            negation = True
            word = word[4:]
        token_negation_pair_list.append((word, negation))
    
    return token_negation_pair_list


## Prepare synset dictionary

synset_list = list(wordnet.all_synsets())
synset_dict_ss_i = dict()
synset_dict_i_ss = dict()
for i, ss in enumerate(synset_list):
    synset_dict_ss_i[ss.name()] = i
    synset_dict_i_ss[i] = ss.name()
                

## Disambiguate each tweet with multicore processing

def GetDisambiguation(tweet_sentence):
    cleaned_tweet = p.clean(tweet_sentence)
    replaced_tweet = replace_word(cleaned_tweet)

    replaced_tweet_list = replaced_tweet.split(" ")
    
    ## Can replace this by using other WSD options (different Lesk algorithms / similarity options)

    da_token_pair_list = disambiguate(replaced_tweet, max_similarity, similarity_option='res')
    # da_token_pair_list = disambiguate(replaced_tweet, cosine_lesk)

    da_token_list = []
    for pair in da_token_pair_list:
        da_token_list.append(pair[0])
    
    token_negation_pair_list = negate(da_token_list)

    print(da_token_pair_list)
    print(len(da_token_pair_list))
    print(token_negation_pair_list)
    print(len(token_negation_pair_list))

    return da_token_pair_list, token_negation_pair_list


from multiprocessing import Process, Queue, Manager
import pickle

manager = Manager()
result_q = Queue()

def doDisambiguation(num, l):

<<<<<<< HEAD
    with open('./wsd_result/wsd_cosine_lesk' + str(num) + '.txt', 'w') as fp:
=======
    # with open(str(num) + '_wsd_cosine_lesk.txt', 'w') as fp:
    with open(str(num) + '_wsd_res_similarity.txt', 'w') as fp:
>>>>>>> 9e13a126ec945d5fdfbfd312a4f26501fde42e97
        for tw in tqdm(l):
            da_token_pair_list, token_negation_pair_list = GetDisambiguation(tw)
            for i in range(len(da_token_pair_list)):
                if da_token_pair_list[i][0] != "":
                    fp.write("%s  " % da_token_pair_list[i][0])
                else:
                    fp.write("_  ")
                fp.write("%s  " % da_token_pair_list[i][1])
                fp.write("%s  " % token_negation_pair_list[i][1])
            fp.write("\n") 

    print(str(num) + ' : file saved')
    print('done')        



da_pair_list = []

p1 = Process(target=doDisambiguation, args=(1, tweets[:1500]))
p2 = Process(target=doDisambiguation, args=(2, tweets[1500:3000]))
p3 = Process(target=doDisambiguation, args=(3, tweets[3000:4500]))
p4 = Process(target=doDisambiguation, args=(4, tweets[4500:6000]))
p5 = Process(target=doDisambiguation, args=(5, tweets[6000:7500]))
p6 = Process(target=doDisambiguation, args=(6, tweets[7500:9000]))
p7 = Process(target=doDisambiguation, args=(7, tweets[9000:10500]))
p8 = Process(target=doDisambiguation, args=(8, tweets[10500:]))

# p1 = Process(target=doDisambiguation, args=(1, tweets[:2]))
# p2 = Process(target=doDisambiguation, args=(2, tweets[2:4]))
# p3 = Process(target=doDisambiguation, args=(3, tweets[4:6]))
# p4 = Process(target=doDisambiguation, args=(4, tweets[6:8]))
# p5 = Process(target=doDisambiguation, args=(5, tweets[8:10]))
# p6 = Process(target=doDisambiguation, args=(6, tweets[10:12]))
# p7 = Process(target=doDisambiguation, args=(7, tweets[12:14]))
# p8 = Process(target=doDisambiguation, args=(8, tweets[14:16]))


p1.start()
p2.start()
p3.start()
p4.start()
p5.start()
p6.start()
p7.start()
p8.start()


p1.join()
p2.join()
p3.join()
p4.join()
p5.join()
p6.join()
p7.join()
p8.join()


print('Done')

