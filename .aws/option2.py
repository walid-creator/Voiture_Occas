import pandas as pd
import numpy as np
import nltk
import nltk.corpus
import os
import regex as reg
'''
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download()
'''
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
import unicodedata


df = pd.read_csv("/Users/famille//projetstat/.aws/aws_automobile1.csv",error_bad_lines=False,index_col=0)




data = df.to_numpy()
data2 = df.values

text = "In Brazil they drive on the right-hand side of the road. Brazil has a large coastline on the eastern side of South America"
token = word_tokenize(text)



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

### Ajout des nouvelles options à la table

for elt in freq_options :
    op = elt[0]
    df[op] = 52513*[0]
    m = df.shape[0]
    for i in range(m):
        if op in op_par_voit[i]:
            df[op][i] = 1
print(df.columns)



"""""
def options_rec() :
    new_options = []
    for elt in fdist :
        if fdist[elt] > 1000 :
            new_options.append(elt)
    return(new_options)

options2 = options_rec()