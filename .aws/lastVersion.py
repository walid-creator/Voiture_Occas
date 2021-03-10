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
df = pd.read_csv("/Users/famille//projetstat/.aws/df_modelisation.csv",error_bad_lines=False,index_col=0)
#df = pd.read_csv("/Users/famille//projetstat/.aws/aws_automobile.csv",error_bad_lines=False,index_col=0)

#print(df.shape)
#print(df.head())
#print(df.columns)
#print(df.iloc[:,15:21].head(10))
#u'reference', u'url', u'user_id': variables d'indentification,
# u'type', u'marque', u'modele' : n'ont qu'une seule modalite
#variable quali: u'modele_com', u'departement', u'couleur',u'energie', u'porte'(nbr portes),u'boite_de_vitesse',premiere_main
#variable quanti: u'horsepower',u'engine', puissance_fiscale,u'kilometrage', 'prix_vente'
#format date: u'date_mec',u'date_mec', u'date_depublication'

#################description des variables:##########################
#type:PRO
#marque: RENAULT
#modele: CLIO
#date_mec: la date de mise en circulation
#horsepower: nombre de chevaux d'une voiture, 1 Ch = 0,736 kW
#engine: La puissance du moteur en kilowatt
#kilometrage: Le nombre de kilometres parcouru par la voiture
#energie: essence ou diesel
#boite_de_vitesse: automatique ou manuel
#porte: le nombre de portes
#options: GPS,
#couleur: ...
#premiere_main:...
#version:
#departement: ...
#puissance_fiscale: la puissance maximale du moteur en terme de kilowatt/heure,
#date_publication: ...
#date_depublication: ...
#prix_vente: en euro
######################traitement###########################################

#suppression des variables d'id:
'''
df.drop([u'reference', u'url', u'user_id'],axis=1,inplace=True)
#creation de la variable age

from pandas import to_datetime
df["date_mec"] = to_datetime(df["date_mec"])
df["date_publication"] = to_datetime(df["date_publication"]).dt.tz_localize(None)
df["date_depublication"] = to_datetime(df["date_depublication"]).dt.tz_localize(None)
df["age"] = (df["date_depublication"]  - df["date_mec"]).dt.days
'''
import numpy as np
#df[u'prix_vente']=pd.to_numeric(df[u'prix_vente'], downcast='integer')
#plt.bar(group_names, df[u'prix_vente'].value_counts())



#supression des variables avec une seule modalite
'''
df.drop([u'type', u'marque',u'modele'], axis=1, inplace=True)
'''

#valeurs manquantes
#Detection des valeurs manquantes:
#df[u'horsepower']=pd.to_numeric(df[u'horsepower'], downcast='integer')
#sans:u'prix_vente',u'departement'(a discretiser),u'energie', u'kilometrage'
#avec:  modele_com(998),u'horsepower'(NaN)(750),engine(890),u'porte'(NaN)(4),options:1928,couleur(5),u'puissance_fiscale'(NaN)(3)
#verifier la presence de val manquantes et les chercher:
''''
print(df.isna().sum())

print(df.isnull().sum())
valManCoul=[]
for i in range(df.shape[0]):
    if df[u'couleur'].isnull()[i]==True:
       print(i)
print(valManCoul)
'''




#uniformiser la variable couleur:
'''
for i in range(df.shape[0]):
    if type(df[u'couleur'][i])==str:
        df[u'couleur'][i]=df[u'couleur'][i].lower()
#print(df.loc[df[u'couleur']=="blnache",:])
#df.iloc[44695,15]="blanc"

df[u"couleur"].replace("blnache", "blanc",inplace= True)
df[u"couleur"].replace("noiir", "noir",inplace= True)
df[u"couleur"].replace("blanche", "blanc",inplace= True)
df[u"couleur"].replace("noire", "noir",inplace= True)
df[u"couleur"].replace("grise", "gris",inplace= True)
df[u"couleur"].replace("roue flamme", "rouge flamme",inplace= True)
df[u"couleur"].replace("blnache", "blanc",inplace= True)
df[u"couleur"].replace("*lanc", "blanc",inplace= True)

#valeur manquantes de couleur:
df[u"couleur"] = df[u"couleur"].fillna("autre")
df[u"couleur"].replace("2017el50360", "autre",inplace= True)
df[u"couleur"].replace("+", "autre",inplace= True)
df[u"couleur"].replace("eqb", "autre",inplace= True)
df[u"couleur"].replace("eehbrunral8007", "autre",inplace= True)
df[u"couleur"].replace(".", "autre",inplace= True)
df[u"couleur"].replace("inc", "autre",inplace= True)
df[u"couleur"].replace("inc.", "autre",inplace= True)
df[u"couleur"].replace("inconn", "autre",inplace= True)
df[u"couleur"].replace("inconnu", "autre",inplace= True)
df[u"couleur"].replace("inconnue", "autre",inplace= True)
df[u"couleur"].replace("inconu", "autre",inplace= True)
df[u"couleur"].replace("kpn", "autre",inplace= True)
df[u"couleur"].replace("n.c.", "autre",inplace= True)
df[u"couleur"].replace("n/a", "autre",inplace= True)
df[u"couleur"].replace("nc", "autre",inplace= True)
df[u"couleur"].replace("neutre.", "autre",inplace= True)
df[u"couleur"].replace("xx", "autre",inplace= True)
df[u"couleur"].replace("rqh", "autre",inplace= True)

df[u"couleur"].replace("non codifiee", "autre",inplace= True)
df[u"couleur"].replace("non codifie", "autre",inplace= True)

df[u"couleur"].replace("non renseigne", "autre",inplace= True)
df[u"couleur"].replace('autre / non affect\xc3\xa9', "autre",inplace= True)
for j in range(df.shape[0]):
    if type(df[u'couleur'][j])==str:
        if df[u'couleur'][j][0:2] == "n1":
            df[u'couleur'][j]='autre'
'''
#détection des couleurs au milieu de la chaine de caractère
'''
for j in df.index.tolist():
    if "noir" in df[u'couleur'][j].strip(".").split():
        df[u'couleur'][j]='noir'
    elif "rouge" in df[u'couleur'][j].strip(".").split():
        df[u'couleur'][j]='rouge'
    elif "jaune" in df[u'couleur'][j].strip(".").split():
        df[u'couleur'][j]='jaune'
    elif "gris" in df[u'couleur'][j].strip(".").split():
        df[u'couleur'][j]='gris'
    elif "blanc" in df[u'couleur'][j].strip(".").split():
        df[u'couleur'][j]='blanc'
    elif "rouge" in df[u'couleur'][j].strip(".").split():
        df[u'couleur'][j]='rouge'
    elif "bleu" in df[u'couleur'][j].strip(".").split():
        df[u'couleur'][j]='bleu'


df[u"couleur"].replace('1602832gris fonce', "gris",inplace= True)
df['couleur'] = df['couleur'].str.replace(u"é", "e")
df['couleur'] = df['couleur'].str.replace(u"É", "e")
df[u"couleur"].replace('9458blanc gl', "blanc",inplace= True)
df[u"couleur"].replace('orance valencia', "orange",inplace= True)
df[u"couleur"].replace('nior etoile', "noir",inplace= True)
df[u"couleur"].replace('black', "noir",inplace= True)
df[u"couleur"].replace('lanc glacier', "blanc",inplace= True)
for j in df.index:
    if df[u'couleur'][j][0:4]=="gris":
        df[u'couleur'][j]="gris"
    elif  df[u'couleur'][j][0:3]=="noi":
        df[u'couleur'][j] = "noir"
    elif df[u'couleur'][j][0:5]=="blanc":
        df[u'couleur'][j] = "blanc"
    elif df[u'couleur'][j][0:5]=="beige":
        df[u'couleur'][j] = "beige"
    elif df[u'couleur'][j][0:4]=="bleu":
        df[u'couleur'][j] = "bleu"
    elif df[u'couleur'][j][0:6]=="orange":
        df[u'couleur'][j] = "orange"
    elif df[u'couleur'][j][0:2]=="or":
        df[u'couleur'][j] = "or"
    elif df[u'couleur'][j][0:6]=="marron":
        df[u'couleur'][j] = "marron"
    elif df[u'couleur'][j][0:6]=="violet":
        df[u'couleur'][j] = "violet"
    elif df[u'couleur'][j][0:4]=="vert":
        df[u'couleur'][j] = "vert"
    elif df[u'couleur'][j][0:4]=="roug":
        df[u'couleur'][j] = "rouge"
    elif df[u'couleur'][j][0:5]=="jaune":
        df[u'couleur'][j] = "jaune"
    elif df[u'couleur'][j][0:6]=="platin":
        df[u'couleur'][j] = "platine"
    elif df[u'couleur'][j][0:4]=="tita":
        df[u'couleur'][j] = "titane"
'''
#unifomisation de couleur non encore termine
'''
eff = df["couleur"].value_counts()
pourcent = df["couleur"].value_counts(normalize = True)
eff = pd.concat([eff, pourcent], axis = "columns")
pourcent_col=pd.DataFrame(list(pourcent.items()),columns=['couleur','pourcentage'])
couleurs = pourcent_col[ pourcent_col["pourcentage"]<0.01].couleur
for nom in couleurs:
    df["couleur"].replace(nom, "autre",inplace=True)
print(df.couleur.value_counts())
'''
## Uniformisation de la variable boite_de_vitesse
"""
df["boite_de_vitesse"] = df["boite_de_vitesse"].replace(["mÃ©canique", "mécanique"], "meca")
df["boite_de_vitesse"] = df["boite_de_vitesse"].replace("automatique", "auto")
from numpy import unique
modalite = unique(df.boite_de_vitesse) # 2 modalités
"""
# Recodage de la variable energie en ne gardant que certaines modalités
'''
df.loc[df["energie"] == "Diesel","energie"] = "diesel"
df.loc[df["energie"] == "Electrique","energie"] = "electrique"
df.loc[df["energie"] == "Essence","energie"] = "essence"
df.loc[df["energie"] == "Hybride essence électrique","energie"] = "heelect"


# Get names of indexes for which column Stock has value : 'Bicarburation essence GPL','Bicarburation essence bio\xc3\xa9thanol','Biocarburant'
indexNames1 = df[ df["energie"] == 'Bicarburation essence GPL'].index
indexNames2 = df[ df["energie"] == 'Bicarburation essence bioéthanol'].index
indexNames3 = df[ df["energie"] == 'Biocarburant'].index
# Delete these row indexes from dataFrame
print(indexNames1)
df.drop(indexNames1, inplace=True)
df.drop(indexNames2, inplace=True)
df.drop(indexNames3, inplace=True)
'''

#uniformisation de la variable model_com
# Reduction du nombre de modalités qui representent mois de 1% de l'effectif total
'''
mod1 = ["CLIO 2 CAMPUS","CLIO 2 CAMPUS SOCIETE", "CLIO 2 RS", "CLIO 2 SOCIETE", "CLIO 2 V6 RS"]
mod2 = ["CLIO 3 COLLECTION", "CLIO 3 COLLECTION SOCIETE", "CLIO 3 ESTATE", "CLIO 3 RS", "CLIO 3 SOCIETE"]
mod3 = ["CLIO 4 ESTATE", "CLIO 4 RS", "CLIO 4 SOCIETE"]
mod4 = ["CLIO 5 SOCIETE"]
autre = ["CLIO", "MEGANE 4","CAPTUR","ZOE", "CLIO WILLIAMS", "CLIO SOCIETE", "TWINGO 3", "GRAND ESPACE 4"]
df.modele_com = df.modele_com.replace(mod1, "CLIO 2")
df.modele_com = df.modele_com.replace(mod2, "CLIO 2")
df.modele_com = df.modele_com.replace(mod3, "CLIO 2")
df.modele_com = df.modele_com.replace(mod4, "CLIO 2")
df.modele_com = df.modele_com.replace(autre, "autre")
'''


#Analyse descriptive
'''
print(df.loc[df[u'couleur'][0:5]=="blanc",:])
MissingData=df[u'couleur'].value_counts(dropna=False)
print(MissingData)

print(df.iloc[44695,22])
print(df['prix_vente'].astype('float').mean(axis=0))
print(df['prix_vente'].astype('float').max(axis=0))
print(df['prix_vente'].astype('float').min(axis=0))
print(df['prix_vente'].astype('float').median(axis=0))
print(df['prix_vente'].astype('float').std(axis=0))# ecrart type au sens statistique n-1 et saute eventuellement les na
#la moyenne et la mediane sont proches
#un ecrat type trop grand de 55622
#ou
'''
#print(df["prix_vente"].describe())



#reprensentation graphique pour les differentes correlations qualitatives
'''
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#df[u'prix_vente']=pd.to_numeric(df[u'prix_vente'], downcast='integer')
#plt.bar(group_names, df[u'prix_vente'].value_counts())

# set x/y labels and plot title
#plt.xlabel(u"prix_vente")
#plt.ylabel(u"count")
#plt.title(u"prix_vente")
'''

#correlation entre variables quantitatives et le prix
'''
print(df[[u"kilometrage", u"prix_vente"]].corr())#il est bien negative mais faible 0.051
print(df[[u"puissance_fiscale", u"prix_vente"]].corr())#0.022
print(df[[u"horsepower", u"prix_vente"]].corr())#0.035
print(df[[u"age", u"prix_vente"]].corr())# -0.04
print(df[[u"prix_vente", u"engine"]].corr())# -0.04
print(df[[u"age", u"kilometrage",u"puissance_fiscale",u"horsepower","engine"]].corr())#forte corrélation:age&kilometrage=0.76 et puissance_fiscale&horsepower=0.85
'''
#correlation entres variables qualitatives
'''
from scipy import stats
table=pd.crosstab(df[u"energie"],df[u"porte"])
print(table)
x=stats.chi2_contingency(table)
print("la stat de test est = ", x[0], " with a P-value of P =", x[1], "nbr de degrès de libertés =", x[2])
#le chi2 théorique est égale à 7,82 pour 3 degrès de libertés => on rejète
'''
#

'''
#print(df['couleur'].value_counts().to_frame())
#sns.boxplot(x=u"premiere_main", y=u"prix_vente", data=df)
#sns.boxplot(df[u"prix_vente"])
#plt.show()
'''


#Verification des modalites
from numpy import unique
#modalite=unique(df["modele_com"])
#modalite1=unique(df["horsepower"])
#modalite2=unique(df["couleur"])
#modalite3=unique(df["departement"])
#modalite4=unique(df["energie"])
#modalite5=unique(df["porte"])
#modalite6=unique(df["boite_de_vitesse"])
#modalite7=unique(df["premiere_main"])
#print(modalite2)
#print(df.head())




#df.to_csv('/Users/famille//projetstat/.aws/automobile.csv')
