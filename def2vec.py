import numpy as np
from nltk.corpus import wordnet as wn
from tqdm import tqdm
import gensim, logging

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import re
import preprocessor as p


all_words  = set(i for i in wn.words())

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

    token_pair_list = []
    
    for word in negged_tokens:
        negation = False
        if word.startswith('NEG_'):
            negation = True
            word = word[4:]
        token_pair_list.append((word, negation))
    
    return token_pair_list


def GetNegatedTokens(sentence):
    cleaned_sentence = p.clean(sentence)
    replaced_sentence = replace_word(cleaned_sentence)
    replaced_sentence_word_list = replaced_sentence.split(" ")

    token_negation_pair_list = negate(replaced_sentence_word_list)

    return token_negation_pair_list


all_synset_list = list(wn.all_synsets())
all_ss_df_token_list = []
for ss in all_synset_list:
    curr_def = GetNegatedTokens(ss.definition())
    for lemma in ss.lemmas():
        curr_def.append((lemma.name(), False))
    all_ss_df_token_list.append(curr_def)


word_df_word_pair_list = []

for df_token_pair_list in tqdm(all_ss_df_token_list):
    
    curr_ss_df_ss_list = []
    p_flag = False
    for df_token_pair in df_token_pair_list:
        if df_token_pair[0][0] == '(':
            p_flag = True
            continue
        if p_flag == True:
            if df_token_pair[0][-1] == ')':
                p_flag = False
                continue
        else:
            if df_token_pair[1] == False:
                curr_ss_df_ss_list.append(df_token_pair[0])
            else:
                if wn.synsets(df_token_pair[0]):
                    if wn.synsets(df_token_pair[0])[0].lemmas():
                        if wn.synsets(df_token_pair[0])[0].lemmas()[0].antonyms():
                            curr_ss_df_ss_list.append(wn.synsets(df_token_pair[0])[0].lemmas()[0].antonyms()[0].name())
            
    word_df_word_pair_list.append(curr_ss_df_ss_list)  


def_model_300 = gensim.models.Word2Vec(word_df_word_pair_list, size=300, window=30, min_count=1, workers=8, sg=1, iter=30)

fname = 'model_def2vec_300'
def_model_300.save(fname)

def_model_500 = gensim.models.Word2Vec(word_df_word_pair_list, size=500, window=30, min_count=1, workers=8, sg=1, iter=30)

fname = 'model_def2vec_500'
def_model_500.save(fname)

def_model_1000 = gensim.models.Word2Vec(word_df_word_pair_list, size=500, window=30, min_count=1, workers=8, sg=1, iter=30)

fname = 'model_def2vec_500'
def_model_1000.save(fname)


