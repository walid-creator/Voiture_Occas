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
df[u"couleur"].replace("non condifiee", "autre",inplace= True)
df[u"couleur"].replace("non renseigne", "autre",inplace= True)
df[u"couleur"].replace('autre / non affect\xc3\xa9', "autre",inplace= True)
for j in range(df.shape[0]):
    if type(df[u'couleur'][j])==str:
        if df[u'couleur'][j][0:2] == "n1":
            df[u'couleur'][j]='autre'

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
from numpy import unique
# Get names of indexes for which column Stock has value : 'Bicarburation essence GPL','Bicarburation essence bio\xc3\xa9thanol','Biocarburant'
indexNames1 = df[ df["energie"] == 'Bicarburation essence GPL'].index
indexNames2 = df[ df["energie"] == 'Bicarburation essence bio\xc3\xa9thanol'].index
indexNames3 = df[ df["energie"] == 'Biocarburant'].index
# Delete these row indexes from dataFrame
df["energie"].drop(indexNames1, inplace=True)
df["energie"].drop(indexNames2, inplace=True)
df["energie"].drop(indexNames3, inplace=True)
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







#valeurs manquantes
#Detection des valeurs manquantes:
#df[u'horsepower']=pd.to_numeric(df[u'horsepower'], downcast='integer')
#sans:u'prix_vente',u'departement'(a discretiser),u'energie', u'kilometrage'
#avec:  modele_com(998),u'horsepower'(NaN)(750),engine(890),u'porte'(NaN)(4),options:1928,couleur(5),u'puissance_fiscale'(NaN)(3)
#verifier la presence de val manquantes et les chercher:
'''
print(df.isna().sum())

print(df.isnull().sum())
valManCoul=[]
for i in range(df.shape[0]):
    if df[u'couleur'].isnull()[i]==True:
       print(i)
print(valManCoul)
'''



#faire un clustering en utilisant les variables quantitatives




#quanti=df[["kilometrage","age"]]#sans aleurs manquantes

from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#nuage de points
#indexNames= quanti[ quanti['prix_vente']>0.8*10**7].index

# Delete these row indexes from dataFrame
'''
quanti.drop(indexNames, inplace=True)
print(quanti)
plt.scatter(quanti['age'],quanti['prix_vente'])
plt.xlabel('Age')
plt.ylabel('prix_vente')
plt.show()
'''

#centrer et reduire
''''
scaler = MinMaxScaler()
#scaler.fit(quanti[[]])
#quanti["prix_vente"] = scaler.transform(quanti[['prix_vente']])
'''
'''
scaler.fit(df[['age']])
df['age'] = scaler.transform(df[['age']])

scaler.fit(df[['kilometrage']])
df['kilometrage'] = scaler.transform(df[['kilometrage']])

#methode des kmeans pour chaque choix de nbr de clusters
#choix de k

sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['age','kilometrage']])
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)
plt.show()
#par la methode du coude on trouve 2 ou 3 a tester
#methode des kmeans

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['age','kilometrage']])
y_predicted
df['cluster']=y_predicted

print(km.cluster_centers_) # centres des classes 

#construire chaque dataframe et repre graphique
df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
print(df3["couleur"].mode())
#df4 = quanti[quanti.cluster==3]
plt.scatter(df1.age,df1['kilometrage'],color='green')
plt.scatter(df2.age,df2['kilometrage'],color='red')
plt.scatter(df3.age,df3['kilometrage'],color='black')
#plt.scatter(df4.age,df4['kilometrage'],color='yellow')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('age')
plt.ylabel('kilometrage')
plt.show()
#print(df1.shape)#(17823, 3)
#print(df2.shape)#(17823, 3)
#print(df3.shape)#(27575, 3)
##print(df4.shape)###(2024, 3)


#imputation par le mode

df1[["modele_com","boite_de_vitesse", "porte"]] = df1[["modele_com","boite_de_vitesse", "porte"]].fillna(df1.mode)
df2[["modele_com","boite_de_vitesse", "porte"]] = df2[["modele_com","boite_de_vitesse", "porte"]].fillna(df2.mode)
df3[["modele_com","boite_de_vitesse", "porte"]] = df3[["modele_com","boite_de_vitesse", "porte"]].fillna(df3.mode)






print(df1[u"couleur"].mode()[0])
print(df2[u"couleur"].mode()[0])
print(df3[u"couleur"].mode()[0])
#imputation par le mode de la variable couleur
df1["couleur"].replace("autre",df1[u"couleur"].mode,inplace= True)
df2["couleur"].replace("autre",df2[u"couleur"].mode,inplace= True)
df3["couleur"].replace("autre",df3[u"couleur"].mode,inplace= True)



#imputation des variables quantitatives par la médiane:

# Convertion en float
df[["horsepower", "engine","puissance_fiscale"]] = df[["horsepower", "engine","puissance_fiscale"]].astype(float)
# Remplacer NAN en utilisant la valeur médiane
df1[["horsepower", "engine","puissance_fiscale"]] = df1[["horsepower", "engine","puissance_fiscale"]].fillna(df1[["horsepower", "engine","puissance_fiscale"]].median())
df2[["horsepower", "engine","puissance_fiscale"]] = df2[["horsepower", "engine","puissance_fiscale"]].fillna(df2[["horsepower", "engine","puissance_fiscale"]].median())
df3[["horsepower", "engine","puissance_fiscale"]] = df3[["horsepower", "engine","puissance_fiscale"]].fillna(df3[["horsepower", "engine","puissance_fiscale"]].median())


df= pd.concat([df1,df2,df3], ignore_index=True)
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
print(df.describe())
'''


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

#correlation entres variables quantitatives et le prix
'''
print(df[[u"kilometrage", u"prix_vente"]].corr())#il est bien negative mais faible 0.051
print(df[[u"puissance_fiscale", u"prix_vente"]].corr())#0.022
print(df[[u"horsepower", u"prix_vente"]].corr())#0.035
print(df[[u"AgeAnnee", u"prix_vente"]].corr())# 0.048
'''

#correlation entres variables qualitatives et le prix
'''
from scipy import stats
#pearson_coef, p_value = stats.pearsonr(df[u"AgeAnnee"], df[u"prix_vente"])
#print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
#print(df['couleur'].value_counts().to_frame())
#sns.boxplot(x=u"premiere_main", y=u"prix_vente", data=df)
#sns.boxplot(df[u"prix_vente"])
#plt.show()
'''

#valeurs influentes
'''
#print(df.shape)#52513x26

valmax=df[u'prix_vente'].quantile(0.9999999999999)
print(valmax)
'''
#Verification des mdalites
from numpy import unique
modalite=unique(df["modele_com"])
modalite1=unique(df["horsepower"])
modalite2=unique(df["couleur"])
modalite3=unique(df["departement"])
modalite4=unique(df["energie"])
modalite5=unique(df["porte"])
modalite6=unique(df["boite_de_vitesse"])
modalite7=unique(df["premiere_main"])
#print(modalite2)
#print(df.head())

#unifomisation de couleur non encore termine
"""
eff = df["couleur"].value_counts()
pourcent = df["couleur"].value_counts(normalize = True)
eff = pd.concat([eff, pourcent], axis = "columns")
pourcent_col=pd.DataFrame(list(pourcent.items()),columns=['couleur','pourcentage'])
couleurs = pourcent_col[ pourcent_col["pourcentage"]<0.01].couleur
print(couleurs)
"""
#=> pres de 500 couleurs dont le pourcentage est inferieur à 1%

#print(df[df["couleur"]=='verte fonc\xc3\xa9e plate'].prix_vente)
#print(df[df["couleur"]=='rouge normale m\xc3\xa9t.'].prix_vente)



#df.to_csv('/Users/famille//projetstat/.aws/automobile.csv')