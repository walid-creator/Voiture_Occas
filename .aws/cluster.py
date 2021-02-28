# coding: utf-8
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy
import scipy
import sklearn
import matplotlib
#import s3fs
#import nltk
#probleme cet import
df = pd.read_csv("/Users/famille//projetstat/.aws/automobile.csv",error_bad_lines=False,index_col=0)
#suppression de electrique et heelect
indexNames1 = df[ df["energie"] == 'electrique'].index
indexNames2 = df[ df["energie"] == 'heelect'].index
print(df[ df["energie"] == 'electrique'].prix_vente)
prix=df[ df["energie"] == 'heelect'].prix_vente
prix=list(prix)
#leurs prix sont proches de la moyenne des prix
#print(sum(prix)/len(prix))


df.drop(indexNames1, inplace=True)
df.drop(indexNames2, inplace=True)

from numpy import unique

#### Extraction des variables qualitatives et quantitatives
vsQual = df[["modele_com","energie","boite_de_vitesse","premiere_main","departement", "porte"]]
Quanti=df[[u'horsepower',u'engine',"age"]]# horsepower et age plus corrélé au prix
#vs=df[["modele_com","energie","boite_de_vitesse","premiere_main","departement", "porte","age","kilometrage"]].dropna()
######fadm


# Suppression des variables département et porte et des valeurs manquantes
Qual = vsQual.drop(["departement","porte"], axis = "columns").dropna()
Quanti = Quanti.dropna()
print(Quanti.describe())
#print(Qual.isna().sum())

# ACM et ACP
from prince import MCA
from prince import PCA

mca = MCA(n_components =4, copy=True,check_input=True,engine='auto',random_state=1,benzecri=False) #M-p puis >1/p
pca=PCA(n_components = 3, copy=True,check_input=True,engine='auto',random_state=1) #>1/p
mca = mca.fit(Qual)
pca=pca.fit(Quanti)
#famd = prince.FAMD(n_components=2,n_iter=3,copy=True,check_input=True,engine='auto',random_state=42)
#famd = famd.fit(vs)
# Nombre de composents:
p = mca.n_components
p1= pca.n_components
#print(p)

# Valeurs propres MCA et PCA
eigen = mca.eigenvalues_
eigen1 = pca.eigenvalues_
#eigen1=famd.eigenvalues_
eigendf = pd.DataFrame(eigen, columns = ["eigen"], index = ["Dim {}".format(i) for i in range(1, p+1,1)])
eigendf1 = pd.DataFrame(eigen1, columns = ["eigen"], index = ["Dim {}".format(i) for i in range(1, p1+1,1)])
inertia=mca.explained_inertia_
inertia1=pca.explained_inertia_
inertiadf = pd.DataFrame(inertia, columns = ["inertia"], index = ["Dim {}".format(i) for i in range(1, p+1,1)])
inertiadf1 = pd.DataFrame(inertia1, columns = ["inertia"], index = ["Dim {}".format(i) for i in range(1, p1+1,1)])

# Valeurs propres acm

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
print(eigendf)
print(eigendf1)
print(inertiadf)
print(inertiadf1)
ax = mca.plot_coordinates(
X=Qual,
ax=None,
figsize=(6, 6),
show_row_points=False,
row_points_size=10,
show_row_labels=False,
show_column_points=True,
column_points_size=10,
show_column_labels=True,
legend_n_cols=1)
plt.show()



# Coordonnées des individus
rowCoord = mca.row_coordinates(Qual)
rowCoord1 = pca.row_coordinates(Quanti)
rowCoord.columns = ['axe1','axe2','axe3','axe4']
rowCoord1.columns = ['axe5','axe6','axe7']
rowCoord = pd.concat([rowCoord, rowCoord1], axis = "columns")
rowCoord=rowCoord.dropna()# car ce n'est pas les mêmes valeurs manquantes entre les 2 dataframe

print(rowCoord.head())

#print(rowCoord.shape)


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

scaler = MinMaxScaler()

scaler.fit(rowCoord)
scaler.transform(rowCoord)

#methode des kmeans pour chaque choix de nbr de clusters
#choix de k
#acp et acm font ca
sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(rowCoord)
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)
plt.show()
#par la methode du coude on trouve 2 ou 3 a tester
#methode des kmeans

km = KMeans(n_clusters=4)
y_predicted = km.fit_predict(rowCoord)
#y_predicted1 = km.fit_predict(rowCoord1)
rowCoord['cluster']=y_predicted
#rowCoord1['cluster']=y_predicted1
#print(km.cluster_centers_) # centres des classes

#construire chaque dataframe et repre graphique
df1 = rowCoord[rowCoord.cluster==0]
df2 = rowCoord[rowCoord.cluster==1]
df3 = rowCoord[rowCoord.cluster==2]
df4 = rowCoord[rowCoord.cluster==3]



#represent graphique:
'''plt.scatter(df1.axe1,df1.axe2,color='green')
plt.scatter(df2.axe1,df2.axe2,color='red')
plt.scatter(df3.axe1,df3.axe2,color='black')
#plt.scatter(df4.age,df4['kilometrage'],color='yellow')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('age')
plt.ylabel('kilometrage')
plt.show()
'''
#print(df1.shape)#(17823, 3)
#print(df2.shape)#(17823, 3)
#print(df3.shape)#(27575, 3)
##print(df4.shape)###(2024, 3)





#librairies pour la CAH
 
 
 
'''
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
#générer la matrice des liens
Z = linkage(rowCoord,method='ward',metric='euclidean')
#affichage du dendrogramme
plt.title("CAH")
dendrogram(Z,labels=rowCoord.index,orientation='left',color_threshold=0)
plt.show()


#matérialisation des 4 classes (hauteur t = 7)
plt.title('CAH avec matérialisation des 4 classes')
dendrogram(Z,labels=Qual.index,orientation='left',color_threshold=7)
plt.show()
#découpage à la hauteur t = 7 ==> identifiants de 4 groupes obtenus
groupes_cah = fcluster(Z,t=7,criterion='distance')
print(groupes_cah)
#index triés des groupes
import numpy as np
idg = np.argsort(groupes_cah)
#affichage des observations et leurs groupes
print(pandas.DataFrame(Qual.index[idg],groupes_cah[idg]))

'''



#imputation par le mode

df1[["modele_com","boite_de_vitesse", "porte"]] = df1[["modele_com","boite_de_vitesse", "porte"]].fillna(df1.mode)
df2[["modele_com","boite_de_vitesse", "porte"]] = df2[["modele_com","boite_de_vitesse", "porte"]].fillna(df2.mode)
df3[["modele_com","boite_de_vitesse", "porte"]] = df3[["modele_com","boite_de_vitesse", "porte"]].fillna(df3.mode)
df4[["modele_com","boite_de_vitesse", "porte"]] = df4[["modele_com","boite_de_vitesse", "porte"]].fillna(df4.mode)

#imputation par le mode de la variable couleur
df1["couleur"].replace("autre",df1[u"couleur"].mode,inplace= True)
df2["couleur"].replace("autre",df2[u"couleur"].mode,inplace= True)
df3["couleur"].replace("autre",df3[u"couleur"].mode,inplace= True)
df4["couleur"].replace("autre",df4[u"couleur"].mode,inplace= True)


#imputation des variables quantitatives par la médiane:

# Convertion en float
df[["horsepower", "engine","puissance_fiscale"]] = df[["horsepower", "engine","puissance_fiscale"]].astype(float)
# Remplacer NAN en utilisant la valeur médiane
df1[["horsepower", "engine","puissance_fiscale"]] = df1[["horsepower", "engine","puissance_fiscale"]].fillna(df1[["horsepower", "engine","puissance_fiscale"]].median())
df2[["horsepower", "engine","puissance_fiscale"]] = df2[["horsepower", "engine","puissance_fiscale"]].fillna(df2[["horsepower", "engine","puissance_fiscale"]].median())
df3[["horsepower", "engine","puissance_fiscale"]] = df3[["horsepower", "engine","puissance_fiscale"]].fillna(df3[["horsepower", "engine","puissance_fiscale"]].median())
df4[["horsepower", "engine","puissance_fiscale"]] = df4[["horsepower", "engine","puissance_fiscale"]].fillna(df4[["horsepower", "engine","puissance_fiscale"]].median())


df= pd.concat([df1,df2,df3,df4], ignore_index=True)
#df.to_csv('/Users/famille//projetstat/.aws/automobile.csv')