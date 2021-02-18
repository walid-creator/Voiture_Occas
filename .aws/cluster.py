# coding: utf-8
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy
import scipy
import sklearn
import matplotlib
import s3fs
#import nltk
#probleme cet import
df = pd.read_csv("/Users/famille//projetstat/.aws/automobile.csv",error_bad_lines=False,index_col=0)

#### Extraction des variables qualitative
vsQual = df[["modele_com","energie","boite_de_vitesse","premiere_main","departement", "porte"]]
print(vsQual.head())


# Suppression des variables département et porte et des valeurs manquantes
Qual = vsQual.drop(["departement", "porte"], axis = "columns").dropna()
print(Qual.isna().sum())

# ACM
from prince import MCA
mca = MCA(n_components = 13, copy=True,check_input=True,engine='auto',random_state=1)
mca = mca.fit(Qual)

# Nombre de composantes
p = mca.n_components
print(p)

# Valeurs propres
eigen = mca.eigenvalues_
eigendf = pd.DataFrame(eigen, columns = ["eigen"], index = ["Dim {}".format(i) for i in range(1, p+1,1)])
print(eigendf)

# Coordonnées des individus
rowCoord = mca.row_coordinates(Qual)
print(rowCoord.head())