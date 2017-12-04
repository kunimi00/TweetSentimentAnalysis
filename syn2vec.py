import numpy as np
from nltk.corpus import wordnet as wn
from tqdm import tqdm
import gensim, logging

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


ss_df_triple_list = []
for i in range(12):
    with open('wsd_result/ss_def_wsd_adapted_lesk_' + str(i+1) + '.txt', 'r') as fp:
        lines = fp.readlines()
        for j in range(0, len(lines), 3):
            if len(lines[j])>0:
                ss_df_triple_list.append((lines[j].replace('\n', ''), lines[j+2].replace('\n', '')))

synset_list = list(wn.all_synsets())


synset_name_list = []
for ss in synset_list:
    synset_name_list.append(ss.name())

ss_df_ss_pair_list = []

for ss_df_triple in tqdm(ss_df_triple_list):
    curr_df_ss_list = ss_df_triple[1].split('  ')[:-1]
    curr_ss_df_ss_list = []
    p_flag = True
    for i in range(0, len(curr_df_ss_list), 3):
        if curr_df_ss_list[i] == ')':
            p_flag = True
            continue 
        if p_flag == True:
            if curr_df_ss_list[i] == '(':
                p_flag = False
                continue
            else:
                if curr_df_ss_list[i+1] != 'None':
                    curr_tmp_ss = curr_df_ss_list[i+1][8:-2]
                    if curr_df_ss_list[i+2] != 'True':
                        curr_ss_df_ss_list.append(curr_tmp_ss)
                    else:
                        if wn.synset(curr_tmp_ss).lemmas():
                            if wn.synset(curr_tmp_ss).lemmas()[0].antonyms():
                                curr_ss_df_ss_list.append(wn.synset(curr_tmp_ss).lemmas()[0].antonyms()[0].name())
                else:
                    continue
        else:
            continue
            
    ss_df_ss_pair_list.append((ss_df_triple[0], curr_ss_df_ss_list))


sentences = []

for ss_pair in ss_df_ss_pair_list:
    curr_sent = []
    curr_sent.append(ss_pair[0])
    curr_sent.extend(ss_pair[1])
    sentences.append(curr_sent)


s2v_model = gensim.models.Word2Vec(sentences, size=500, window=30, min_count=1, workers=8, sg=1)



tsne = TSNE(n_components=2)


custom_synsets_set_pos = ['good.a.01', 'amazing.s.02', 'beautiful.a.01', 'boom.n.03', 'celebrate.v.02', 'appeal.n.02', 'cheerful.a.01', 
                    'clean.a.01', 'confident.a.01', 'convenient.a.01', 'cozy.s.01', 'divine.s.01', 'easy.a.01', 'efficient.a.01',
                    'elegant.a.01', 'encourage.v.02', 'enjoy.v.01', 'entertain.v.01', 'excel.v.01', 'excite.v.01', 'fabulous.s.01',
                    'fresh.a.01', 'gentle.s.02', 'glad.a.01', 'generous.a.01', 'gorgeous.s.01', 'happy.a.01', 'joy.n.01', 'lovely.s.01', 
                    'lucky.a.02', 'outstanding.s.01', 'pleasing.a.01', 'pride.n.01', 'proper.a.01', 'sexy.a.01', 'smart.s.07',
                    'bright.a.01', 'comfortable.a.01', 'cool.s.06', 'faithful.a.01', 'celebrated.s.01', 'all_right.s.01', 'fine-looking.s.01',
                    'healthy.a.01', 'honor.n.02', 'prefer.v.01', 'better.v.02', 'inspire.v.02', 'intelligent.a.01', 'maestro.n.01',
                    'modest.a.01', 'nice.a.01', 'optimum.s.01', 'positive.a.01', 'plus.s.02', 'peaceful.a.01', 'thrive.v.02', 'recommend.v.03',
                    'extremely.r.02', 'ace.s.01', 'victory.n.01', 'fantastic.s.02', 'refresh.v.02', 'satisfy.v.02', 'sensational.a.01', 'smooth.s.07', 'fluent.s.01', 
                    'excellent.s.01', 'brilliant.s.03', 'glorious.s.03', 'success.n.02', 'thoughtful.s.01', 'trust.v.01', 'win.n.01' ]

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

tmp_list = custom_synsets_set_pos + custom_synsets_set_neg

X_pos = s2v_model[custom_synsets_set_pos]
X_neg = s2v_model[custom_synsets_set_neg]

X_list = []
for xs in X_pos:
    X_list.append(xs)
for xs in X_neg:
    X_list.append(xs)

X_tsne = tsne.fit_transform(X_list)

plt.figure(figsize=(20,10))
plt.scatter(X_tsne[:len(X_pos), 0], X_tsne[:len(X_pos), 1], c='b', s=10)
plt.scatter(X_tsne[len(X_pos):, 0], X_tsne[len(X_pos):, 1], c='r', s=10)

for i, word in enumerate(custom_synsets_set_pos[:10]):
#     if i % 3 == 0:
    plt.annotate(word, xy=(X_tsne[i, 0], X_tsne[i, 1]), size = 12)

for i, word in enumerate(custom_synsets_set_neg[:10]):
#     if i % 3 == 0:
    plt.annotate(word, xy=(X_tsne[i+10, 0], X_tsne[i+10, 1]), size = 12)
plt.show()



pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_list)

plt.figure(figsize=(20,10))
plt.scatter(X_pca[:len(X_pos), 0], X_pca[:len(X_pos), 1], c='b', s=10)
plt.scatter(X_pca[len(X_pos):, 0], X_pca[len(X_pos):, 1], c='r', s=10)

for i, word in enumerate(custom_synsets_set_pos[:10]):
    plt.annotate(word, xy=(X_pca[i, 0], X_pca[i, 1]), size=15)

for i, word in enumerate(custom_synsets_set_neg[:10]):
    plt.annotate(word, xy=(X_pca[i+10, 0], X_pca[i+10, 1]), size=15)
plt.show()








