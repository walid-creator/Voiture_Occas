import pandas as pd
import numpy as np
import nltk
import nltk.corpus
import os
import regex as reg
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
import unicodedata


df = pd.read_csv("s3://projet-stat-ensai/Lacentrale.csv",sep =';', dtype=str)
print(df.shape)
print(type(df))
print(df.columns)
print(df.describe())



data = df.to_numpy()
data2 = df.values

text = "In Brazil they drive on the right-hand side of the road. Brazil has a large coastline on the eastern side of South America"
token = word_tokenize(text)

df["reference"]

def uniform(word) :
    u = unicodedata.normalize('NFKD', word).encode('ASCII', 'ignore')
    u2 = u.decode('ASCII')
    u3 = u2.lower()
    return(u3)

def liste_options() :
    m = df.shape[0]
    options = []
    op_par_voit = []
    for i in range(m) :
        print(i)
        if type(df['options'][i]) == str :
            words = word_tokenize(df['options'][i])
            l = []
            n = len(words)
            i = 0
            while i < n :
                w = ''
                while i < n and words[i] != '``' and words[i] != "''" and words != '"''"' and words[i] != ';' :
                    if len(w) > 0 :
                        w += ' '
                    w += words[i]
                    i += 1
                if w != '' :
                    w = uniform(w)
                    l.append(w)
                i += 1
            options += l
            op_par_voit.append(l)
        else :
            op_par_voit.append([])
    return(options,op_par_voit)

options,op_par_voit = liste_options()

fdist = FreqDist(options)
freq_options = fdist.most_common(10)


### Recupération des options présentes dans p% des voitures

def options_rec(p) :
    new_options = []
    for elt in fdist :
        if fdist[elt] > df.shape[0]*(p/100) :
            new_options.append(elt)
    return(new_options)

options2 = options_rec(10)
len(options2)


### Ajout des nouvelles options à la table

decompte = 0
for elt in options2 :
    decompte += 1
    print(decompte)
    df[elt] = 52513*['0']
    m = df.shape[0]
    for i in range(m):
        if elt in op_par_voit[i]:
            df[elt][i] = '1'

df.to_csv('D:/Projet info 2A/data.csv', index = False)

base2 = pd.read_csv('D:/Projet info 2A/data.csv')



### ACP sur les variables options retenues

from prince import MCA
from prince import PCA

base_acm = df.iloc[:,23:]
mca = MCA(n_components =96)
pca = PCA()
mca = mca.fit(base_acm)
pca = pca.fit(base_acm)

p = mca.n_components
eigen = mca.eigenvalues_
inertia=mca.explained_inertia_


"""
pst = PorterStemmer()
pst.stem('prenez')
"""