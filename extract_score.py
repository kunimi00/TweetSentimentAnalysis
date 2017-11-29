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
for line in file:
    split_string = line.split("\t")
    if len(split_string) == 2:
        tweets.append(split_string[0])
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



# ## Instead, we decided to build a graph using synsets.

# #### In the paper, they created a graph based on paths between actual lemmas. 
# #### However, we're going to try a graphical model using only synset because they provide more accurate(diambiguated) sense for any sentence.


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



# new cleaned wordnet SYNSET graph
wordnet_graph_synset_cleaned = nx.Graph(engine='sfdp', pack=True)

for ss in tqdm(synset_list):
    wordnet_graph_synset_cleaned.add_node(ss.name())

for ss in tqdm(synset_list):
    nb_list = [k for m in [n for n in wordnet_graph_synset.neighbors(ss.name())] for k in wordnet_graph_synset.neighbors(m)]    
    for nb in nb_list:
        if wordnet_graph_synset_cleaned.has_edge(ss.name(), nb) == False:
            wordnet_graph_synset_cleaned.add_edge(ss.name(), nb)



def DIST_synset(ss, graph, p_ss='good.a.01', n_ss='bad.a.01'):
    _ss = ss
#     print(wn.synset(ss).definition())
    if graph.has_node(_ss) and graph.has_node(p_ss) and graph.has_node(n_ss):

        if nx.has_path(graph, p_ss, n_ss):
            p_n = nx.shortest_path_length(graph, p_ss, n_ss)
        else:
#             print('No path between '+ p_ss + ' and ' + n_ss)
            return 0
        
        if nx.has_path(graph, _ss, n_ss):
            n_w = nx.shortest_path_length(graph, n_ss, _ss)
        else:
            return 0
        
        if nx.has_path(graph, _ss, p_ss):
            p_w = nx.shortest_path_length(graph, p_ss, _ss)
        else:
            return 0

#         print('distance to '+ n_ss + ' : ' + str(n_w))
#         print('distance to '+ p_ss + ' : ' + str(p_w))
#         print('distance between '+ n_ss + ' and ' + p_ss + ' : ' + str(p_n))
        return (n_w - p_w) / p_n
    else :
        print(_ss + ' is not in the graph.')
        return 0



## Prepare all antonyms for every word in opinion lexicon

bingliu_pos_synset_pair_list = []
for word in opinion_lexicon.positive():
    for i in wn.synsets(word):
        # if i.pos() in ['a', 's', 'r', 'v']: # no nouns
        for j in i.lemmas(): 
            if j.antonyms(): 
                bingliu_pos_synset_pair_list.append((j.synset().name(), j.antonyms()[0].synset().name()))

bingliu_neg_synset_pair_list = []
for word in opinion_lexicon.negative():
    for i in wn.synsets(word):
        # if i.pos() in ['a', 's', 'r', 'v']: # no nouns
        for j in i.lemmas(): 
            if j.antonyms(): 
                ## NOTE that we put antonyms first
                bingliu_neg_synset_pair_list.append((j.antonyms()[0].synset().name(), j.synset().name()))


def GetSentenceSentiScore(tweet_sentence, label, graph, cmp_ss_pair_lists):
    cleaned_tweet = p.clean(tweet_sentence)
    replaced_tweet = replace_word(cleaned_tweet)
    
    ## Can replace this by using other WSD options (different Lesk algorithms / similarity options)
    
#     da_token_pair_list = disambiguate(replaced_tweet, max_similarity, similarity_option='res')
    da_token_pair_list = disambiguate(replaced_tweet, cosine_lesk)
    
    da_token_list = []
    for pair in da_token_pair_list:
        da_token_list.append(pair[0])
    
    tokens, token_pair_list = negate(da_token_list)
    
#     print("####### Tweet #######" )
#     print("")
#     print(tweet_sentence)
#     print(" -> " + replaced_tweet)
#     print("")
#     print(da_token_list)
#     print("") 
#     print(da_token_pair_list)
#     print("") 
#     print(token_pair_list)
#     print("")     
    
    assert(len(da_token_pair_list) == len(token_pair_list))
    
    score = 0
    for j in range(len(da_token_pair_list)):
        curr_neg = False
        if token_pair_list[j][1]:
#             print("negated : " + token_pair_list[j][0])
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
                
#             print("   " + str(token_pair_list[j]) + " -> " + curr_ss.name() + " : " + str(curr_score))
#             print("   definition : " + wn.synset(curr_ss.name()).definition())
#             print("   score from graph : " + str(curr_score))
#             print("")
            score += curr_score            
            
        else:
            continue
    print("Tweet " + str(i) + " : TOTAL score = " + str(score) + " / " + label)
#     print("")
    
    return score


## custom word set
custom_words_set = ['good', 'awesome', 'beautiful', 'boom', 'celebrate', 'charm', 'cheerful', 
                    'clean', 'confident', 'convenient', 'cozy']

custom_synset_pair_list = []

for word in custom_words_set:
    for i in wn.synsets(word):
        # if i.pos() in ['a', 's', 'r', 'v']: # no nouns
        for j in i.lemmas(): 
            if j.antonyms(): 
                ## NOTE that we put antonyms first
                custom_synset_pair_list.append((j.antonyms()[0].synset().name(), j.synset().name()))



SentiGraphFeature = []
# for i in tqdm(range(len(tweets))):

for i in tqdm(range(len(tweets))):
	start = time.time()
	SentiGraphFeature.append(GetSentenceSentiScore(tweets[i], labels[i], wordnet_graph_synset_cleaned, [bingliu_pos_synset_pair_list]))
	print(time.time()-start)

import pickle

with open('./senti_score.p', 'wb') as fp:
    pickle.dump(SentiGraphFeature, fp)

with open('./senti_score.p', 'rb') as fp:
	loaded_score = pickle.load(fp)

print(len(loaded_score))
print('Done')
