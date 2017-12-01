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
from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn
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

## String to List 
import ast


verbose = False
loadSavedGraph = False

## Load the disambiguated word list
loaded_da_token_pair_list = []
for i in range(8):
    ## Either cosine_lesk ones OR res_similarity ones
    with open(str(num) + '_wsd_cosine_lesk.txt', 'r') as f:
        tmp_list = ast.literal_eval(f.read())
        loaded_da_token_pair_list.extend(tmp_list)


## Prepare graphical model 

if loadSavedGraph == False:

    synset_list = list(wn.all_synsets())
    wordnet_graph_synset = nx.Graph(engine='sfdp', pack=True)

    seen = set()
    for ss in tqdm(synset_list):
        wordnet_graph_synset.add_node(ss.name())
        for lm in ss.lemmas():
            _lm = lm.name()
            if not _lm in seen:
                seen.add(_lm)
                wordnet_graph_synset.add_node(_lm)
            wordnet_graph_synset.add_edge(_lm, ss.name())

    path = './wordnet_graph_synset.p'
    nx.write_gpickle(wordnet_graph_synset, path)


    ## new cleaned wordnet SYNSET graph
    wordnet_graph_synset_cleaned = nx.Graph(engine='sfdp', pack=True)

    for ss in tqdm(synset_list):
        wordnet_graph_synset_cleaned.add_node(ss.name())

    for ss in tqdm(synset_list):
        nb_list = [k for m in [n for n in wordnet_graph_synset.neighbors(ss.name())] for k in wordnet_graph_synset.neighbors(m)]    
        for nb in nb_list:
            if wordnet_graph_synset_cleaned.has_edge(ss.name(), nb) == False:
                wordnet_graph_synset_cleaned.add_edge(ss.name(), nb)

    path = './wordnet_graph_synset_cleaned.p'
    nx.write_gpickle(wordnet_graph_synset_cleaned, path)

else:
    wordnet_graph_synset = nx.read_gpickle("wordnet_graph_synset.p")
    wordnet_graph_synset_cleaned = nx.read_gpickle("wordnet_graph_synset_cleaned.p")


## Distance function for synset graph

def DIST_synset(ss, graph, p_ss='good.a.01', n_ss='bad.a.01'):
    _ss = ss
    if verbose:
        print(wn.synset(ss).definition())
    if graph.has_node(_ss) and graph.has_node(p_ss) and graph.has_node(n_ss):

        if nx.has_path(graph, p_ss, n_ss):
            p_n = nx.shortest_path_length(graph, p_ss, n_ss)
        else:
            if verbose:
                print('No path between '+ p_ss + ' and ' + n_ss)
            return 0
        
        if nx.has_path(graph, _ss, n_ss):
            n_w = nx.shortest_path_length(graph, n_ss, _ss)
        else:
            return 0
        
        if nx.has_path(graph, _ss, p_ss):
            p_w = nx.shortest_path_length(graph, p_ss, _ss)
        else:
            return 0

        if verbose:
            print('distance to '+ n_ss + ' : ' + str(n_w))
            print('distance to '+ p_ss + ' : ' + str(p_w))
            print('distance between '+ n_ss + ' and ' + p_ss + ' : ' + str(p_n))
        return (n_w - p_w) / p_n
    else :
        print(_ss + ' is not in the graph.')
        return 0



## Prepare all antonyms for every word in opinion lexicon

bingliu_pos_synset_pair_list = []
for word in opinion_lexicon.positive():
    for i in wn.synsets(word):
        for j in i.lemmas(): 
            if j.antonyms(): 
                bingliu_pos_synset_pair_list.append((j.synset().name(), j.antonyms()[0].synset().name()))

bingliu_neg_synset_pair_list = []
for word in opinion_lexicon.negative():
    for i in wn.synsets(word):
        for j in i.lemmas(): 
            if j.antonyms(): 
                ## NOTE that we put antonyms first
                bingliu_neg_synset_pair_list.append((j.antonyms()[0].synset().name(), j.synset().name()))


def GetSentenceSentiScore(da_token_pair_list, label, graph, cmp_ss_pair_lists):
    
    da_token_list = []
    for pair in da_token_pair_list:
        da_token_list.append(pair[0])
    
    tokens, token_pair_list = negate(da_token_list)
    
    if verbose:
        print("####### Tweet #######" )
        print("")
        print(tweet_sentence)
        print(" -> " + replaced_tweet)
        print("")
        print(da_token_list)
        print("") 
        print(da_token_pair_list)
        print("") 
        print(token_pair_list)
        print("")     
    
    assert(len(da_token_pair_list) == len(token_pair_list))
    
    score = 0
    for j in range(len(da_token_pair_list)):
        curr_neg = False
        if token_pair_list[j][1]:
            if verbose:
                print("negated : " + token_pair_list[j][0])
            curr_neg = True
        
        curr_ss = da_token_pair_list[j][1]
        curr_score = 0
        if curr_ss:
            for cmp_ss_pair_list in cmp_ss_pair_lists:
                for cmp_ss_pair in cmp_ss_pair_list:
                    curr_score += DIST_synset(curr_ss.name(), graph, cmp_ss_pair[0], cmp_ss_pair[1])
                curr_score /= len(cmp_ss_pair_list)
            
            if curr_neg:
                curr_score *= -1
            
            if verbose:
                print("   " + str(token_pair_list[j]) + " -> " + curr_ss.name() + " : " + str(curr_score))
                print("   definition : " + wn.synset(curr_ss.name()).definition())
                print("   score from graph : " + str(curr_score))
                print("")
            score += curr_score            
            
        else:
            continue

    if verbose:
        print(">>> Tweet TOTAL score = " + str(score) + " / " + label)
        print("")
    
    return score


## Custom word set
'''
custom_words_set = ['good', 'awesome', 'beautiful', 'boom', 'celebrate', 'charm', 'cheerful', 
                    'clean', 'confident', 'convenient', 'cozy']

custom_synset_pair_list = []

for word in custom_words_set:
    for i in wn.synsets(word):
        # if i.pos() in ['a', 's', 'r', 'v']: # no nouns
        for j in i.lemmas(): 
            if j.antonyms(): 
                custom_synset_pair_list.append((j.synset().name(), j.antonyms()[0].synset().name()))
'''




## Calculate the distance for each tweet with multicore processing

'''
SentiGraphFeature = []

for i in tqdm(range(len(loaded_da_token_pair_list))):
    start = time.time()
    SentiGraphFeature.append(GetSentenceSentiScore(loaded_da_token_pair_list[i], wordnet_graph_synset_cleaned, [bingliu_pos_synset_pair_list, bingliu_neg_synset_pair_list]))
    print(time.time()-start)
'''

from multiprocessing import Process
import pickle


def doDisambiguation(num, l):
    da_list = []
    for tw in tqdm(l):
        da_list.append(GetDisambiguation(tw))
    print(str(da_list))
    with open(str(num) + '_wsd_cosine_lesk.txt', 'w') as fp:
        fp.write(str(da_list))
    print(str(num) + ' : file saved')



da_pair_list = []

p1 = Process(target=doDisambiguation, args=(1, tweets[:1500]))
p2 = Process(target=doDisambiguation, args=(2, tweets[1500:3000]))
p3 = Process(target=doDisambiguation, args=(3, tweets[3000:4500]))
p4 = Process(target=doDisambiguation, args=(4, tweets[4500:6000]))
p5 = Process(target=doDisambiguation, args=(5, tweets[6000:7500]))
p6 = Process(target=doDisambiguation, args=(6, tweets[7500:9000]))
p7 = Process(target=doDisambiguation, args=(7, tweets[9000:10500]))
p8 = Process(target=doDisambiguation, args=(8, tweets[10500:]))

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











'''
with open('./senti_score.p', 'wb') as fp:
    pickle.dump(SentiGraphFeature, fp)

with open('./senti_score.p', 'rb') as fp:
	loaded_score = pickle.load(fp)

print(len(loaded_score))
print('Done')
'''