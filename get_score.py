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

## String to List 
import ast


verbose = False
loadSavedGraph = False

## Load the disambiguated word list
loaded_all_tweets_list = []
for i in range(8):
    ## Either cosine_lesk ones OR res_similarity ones
    with open(str(i+1) + '_wsd_cosine_lesk.txt', 'r') as f:
        tmp_str = f.read()
        tmp_list = tmp_str.split('\n')
        tmp_list = tmp_list[:-1]

        loaded_curr_file_list = []
        for line in tmp_list:
            ll = line.split("  ")[:-1]
            curr_tweet = list()
            for i in range(0, len(ll), 3):
                curr_tweet.append([ll[i], ll[i+1], ll[i+2]])
            loaded_curr_file_list.append(curr_tweet)

        loaded_all_tweets_list.extend(loaded_curr_file_list)


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



def GetSentenceSentiScore(one_tweet_triple_list, graph = wordnet_graph_synset_cleaned, 
                        cmp_ss_pair_lists = [bingliu_pos_synset_pair_list, bingliu_neg_synset_pair_list]):

    if verbose:
        print("####### Tweet #######" )
        print("")
        print(one_tweet_triple_list)
        print("") 
    
    score = 0
    returning_tweet_words = ""
    for j in range(len(one_tweet_triple_list)):

        returning_tweet_words += one_tweet_triple_list[j][0] + " "

        curr_neg = False
        if eval(one_tweet_triple_list[j][2]):
            if verbose:
                print("negated : " + one_tweet_triple_list[j][0])
            curr_neg = True
        
        tmp_str = one_tweet_triple_list[j][1][8:-2]
        if len(tmp_str) > 0:
            curr_ss = wn.synset(tmp_str)
            curr_score = 0
            if curr_ss:
                for cmp_ss_pair_list in cmp_ss_pair_lists:
                    for cmp_ss_pair in cmp_ss_pair_list:
                        curr_score += DIST_synset(curr_ss.name(), graph, cmp_ss_pair[0], cmp_ss_pair[1])
                    curr_score /= len(cmp_ss_pair_list)
                
                if curr_neg:
                    curr_score *= -1
                
                if verbose:
                    print("   " + str(one_tweet_triple_list[j]) + " -> " + curr_ss.name() + " : " + str(curr_score))
                    print("   definition : " + wn.synset(curr_ss.name()).definition())
                    print("   score from graph : " + str(curr_score))
                    print("")
                score += curr_score            
                
            else:
                continue
        else:
            continue

    if verbose:
        print(">>> Tweet TOTAL score = " + str(score) + " / " + label)
        print("")
    
    return returning_tweet_words, score


## Custom positive word set
custom_words_set_pos = ['good', 'awesome', 'beautiful', 'boom', 'celebrate', 'charm', 'cheerful', 
                    'clean', 'confident', 'convenient', 'cozy', 'divine', 'easy', 'efficient',
                    'elegant', 'encourage', 'enjoy', 'entertain', 'excelent', 'exciting', 'fabulous',
                    'fresh', 'gentle', 'glad', 'generous', 'gorgeous', 'happy', 'joy', 'lovely', 
                    'lucky', 'outstanding', 'pleasing', 'pride', 'proper', 'sexy', 'smart',
                    'bright', 'comfortable', 'cool', 'faithful', 'famous', 'fine', 'handsome',
                    'healthy', 'honor', 'prefer', 'improve', 'inspire', 'intelligent', 'master',
                    'modest', 'nice', 'optimal', 'positive', 'peaceful', 'prosper', 'recommend',
                    'super', 'victory', 'wonderful', 'refresh', 'satisfy', 'sensational', 'smooth', 
                    'splendid', 'success', 'thoughtful', 'trust', 'win' ]

custom_pos_synset_pair_list = []

for word in custom_words_set_pos:
    for i in wn.synsets(word):
        for j in i.lemmas(): 
            if j.antonyms(): 
                custom_pos_synset_pair_list.append((j.synset().name(), j.antonyms()[0].synset().name()))


## Custom negative word set
custom_words_set_neg = ['abnormal', 'abort', 'abuse', 'afraid', 'angry', 'arrogant', 'ashamed', 
                    'awful', 'bad', 'bitch', 'blame', 'boring', 'brutal', 'bullshit',
                    'cancer', 'chaotic', 'cheat', 'cocky', 'conflict', 'confuse', 'controversial',
                    'corrupt', 'creepy', 'curse', 'dangerous', 'dead', 'defect', 'depression', 'destroy', 
                    'die', 'dick', 'disappoint', 'disbelief', 'discomfort', 'disgraceful', 'disobey',
                    'disturbing', 'dump', 'embarrass', 'error', 'exhaust', 'fail', 'fake', 
                    'false', 'fool', 'freak', 'fraud', 'grief', 'hard', 'harmed', 'idiot',
                    'ignorant', 'misfortune', 'mistake', 'murder', 'negative', 'painful', 'pervert', 
                    'poor', 'problem', 'racist', 'reject', 'sad', 'scold', 'screwed', 'selfish',
                    'silly', 'sloppy', 'stink', 'stupid', 'suck', 'terrible', 'trash', 'weak' ]

custom_neg_synset_pair_list = []

for word in custom_words_set_neg:
    for i in wn.synsets(word):
        for j in i.lemmas(): 
            if j.antonyms(): 
                custom_neg_synset_pair_list.append((j.synset().name(), j.antonyms()[0].synset().name()))






## Custom positive selected synset set
custom_synsets_set_pos = ['good.a.01', 'amazing.s.02', 'beautiful.a.01', 'boom.n.03', 'celebrate.v.02', 'appeal.n.02', 'cheerful.a.01', 
                    'clean.a.01', 'confident.a.01', 'convenient.a.01', 'cozy.s.01', 'divine.s.01', 'easy.a.01', 'efficient.a.01'
                    'elegant.a.01', 'encourage.v.02', 'enjoy.v.01', 'entertain.v.01', 'excel.v.01', 'excite.v.01', 'fabulous.s.01',
                    'fresh.a.01', 'gentle.s.02', 'glad.a.01', 'generous.a.01', 'gorgeous.s.01', 'happy.a.01', 'joy.n.01', 'lovely.s.01', 
                    'lucky.a.02', 'outstanding.s.01', 'pleasing.a.01', 'pride.n.01', 'proper.a.01', 'sexy.a.01', 'smart.s.07',
                    'bright.a.01', 'comfortable.a.01', 'cool.s.06', 'faithful.a.01', 'celebrated.s.01', 'all_right.s.01', 'fine-looking.s.01',
                    'healthy.a.01', 'honor.n.02', 'prefer.v.01', 'better.v.02', 'inspire.v.02', 'intelligent.a.01', 'maestro.n.01',
                    'modest.a.01', 'nice.a.01', 'optimum.s.01', 'positive.a.01', 'plus.s.02', 'peaceful.a.01', 'thrive.v.02', 'recommend.v.03',
                    'extremely.r.02', 'ace.s.01', 'victory.n.01', 'fantastic.s.02', 'refresh.v.02', 'satisfy.v.02', 'sensational.a.01', 'smooth.s.07', 'fluent.s.01', 
                    'excellent.s.01', 'brilliant.s.03', 'glorious.s.03', 'success.n.02', 'thoughtful.s.01', 'trust.v.01', 'win.n.01' ]


custom_pos_sel_ss_pair_list = []

for ss in custom_synsets_set_pos:
    if len(wn.synset(ss).lemmas())>0:
        j = wn.synset(ss).lemmas()[0]
        if j.antonyms(): 
            custom_pos_sel_ss_pair_list.append((wn.synset(ss).name(), j.antonyms()[0].synset().name()))


# Custom negative selected synset set
custom_synsets_set_neg = ['abnormal.a.01', 'abort.v.01', 'maltreatment.n.01', 'mistreat.v.01', 'afraid.a.01', 'angry.a.01', 'arrogant.s.01', 'ashamed.a.01', 
                    'atrocious.s.02', 'bad.a.01', 'cunt.n.01','gripe.v.01', 'blame.v.01', 'boring.s.01', 'brutal.s.02', 'bullshit.n.01',
                    'cancer.n.01', 'chaotic.s.01', 'cheat.n.05', 'cocky.s.01', 'conflict.n.01', 'confuse.v.02', 'controversial.a.01',
                    'corrupt.v.01', 'creepy.s.01', 'curse.v.01', 'dangerous.a.01', 'dead.n.01', 'defect.n.02', 'depression.n.01', 'destroy.v.02', 
                    'die.v.01', 'cock.n.01', 'disappoint.v.01', 'incredulity.n.01', 'discomfort.n.01', 'disgraceful.s.01', 'disobey.v.01',
                    'distressing.s.01', 'shit.n.04', 'embarrass.v.01', 'error.n.03', 'exhaust.v.01', 'fail.v.01', 'fake.n.01', 
                    'false.a.01', 'fool.n.01', 'freak.n.01', 'fraud.n.01', 'fraud.n.03', 'difficult.a.01', 'harm.v.01', 'idiot.n.01',
                    'ignorant.s.01', 'misfortune.n.01', 'mistake.n.01', 'murder.v.01', 'negative.a.01', 'painful.a.01', 'pervert.n.01', 
                    'poor.a.02', 'poor.s.06', 'problem.n.01','trouble.n.01', 'racist.n.01', 'reject.v.01', 'sad.a.01', 'grouch.v.01', 'cheat.v.02', 'selfish.a.01',
                    'pathetic.s.03', 'sloppy.s.01', 'malodor.n.01','reek.v.02', 'stupid.a.01', 'suck.v.04', 'awful.s.02', 'rubbish.n.01', 'weak.a.01' ]



custom_neg_sel_ss_pair_list = []

for ss in custom_synsets_set_neg:
    if len(wn.synset(ss).lemmas())>0:
        j = wn.synset(ss).lemmas()[0]
        if j.antonyms():
            custom_neg_sel_ss_pair_list.append((j.name(), j.antonyms()[0].synset().name()))







## Calculate the distance for each tweet with multicore processing

from multiprocessing import Process
import pickle


# def GetSentenceSentiScore(one_tweet_triple_list, graph, cmp_ss_pair_lists)

def doGetSentenceSentiScore(num, l):
    with open('./scores/dscore_custom_neg' + str(num) + '.txt', 'w') as fp:
        for tw in tqdm(l):
            # curr_tweet_words, curr_tw_score = GetSentenceSentiScore(tw, cmp_ss_pair_lists = [custom_pos_synset_pair_list])
            curr_tweet_words, curr_tw_score = GetSentenceSentiScore(tw, cmp_ss_pair_lists = [custom_neg_synset_pair_list])
            # curr_tweet_words, curr_tw_score = GetSentenceSentiScore(tw, cmp_ss_pair_lists = [bingliu_neg_synset_pair_list])
            fp.write("%s  %s \n" % (curr_tweet_words, curr_tw_score))

    print(str(num) + ' : file saved')
    print('done')


da_pair_list = []

p1 = Process(target=doGetSentenceSentiScore, args=(1, loaded_all_tweets_list[:1500]))
p2 = Process(target=doGetSentenceSentiScore, args=(2, loaded_all_tweets_list[1500:3000]))
p3 = Process(target=doGetSentenceSentiScore, args=(3, loaded_all_tweets_list[3000:4500]))
p4 = Process(target=doGetSentenceSentiScore, args=(4, loaded_all_tweets_list[4500:6000]))
p5 = Process(target=doGetSentenceSentiScore, args=(5, loaded_all_tweets_list[6000:7500]))
p6 = Process(target=doGetSentenceSentiScore, args=(6, loaded_all_tweets_list[7500:9000]))
p7 = Process(target=doGetSentenceSentiScore, args=(7, loaded_all_tweets_list[9000:10500]))
p8 = Process(target=doGetSentenceSentiScore, args=(8, loaded_all_tweets_list[10500:]))

# p1 = Process(target=doGetSentenceSentiScore, args=(1, loaded_all_tweets_list[:2]))
# p2 = Process(target=doGetSentenceSentiScore, args=(2, loaded_all_tweets_list[2:4]))
# p3 = Process(target=doGetSentenceSentiScore, args=(3, loaded_all_tweets_list[4:6]))
# p4 = Process(target=doGetSentenceSentiScore, args=(4, loaded_all_tweets_list[6:8]))
# p5 = Process(target=doGetSentenceSentiScore, args=(5, loaded_all_tweets_list[8:10]))
# p6 = Process(target=doGetSentenceSentiScore, args=(6, loaded_all_tweets_list[10:12]))
# p7 = Process(target=doGetSentenceSentiScore, args=(7, loaded_all_tweets_list[12:14]))
# p8 = Process(target=doGetSentenceSentiScore, args=(8, loaded_all_tweets_list[14:16]))

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



