# coding: utf-8
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy
import scipy
import sklearn
import matplotlib
import s3fs
#import nltk
# probleme cet import
df = pd.read_csv("/Users/famille//projetstat/.aws/automobile2.csv", error_bad_lines=False, index_col=0)

from numpy import unique
#### Extraction des variables qualitatives et quantitatives
vsQual = df[["modele_com", "energie", "boite_de_vitesse", "premiere_main", "departement", "porte"]]
# Suppression des variables département et porte et des valeurs manquantes
Qual = vsQual.drop(["departement", "porte"], axis="columns").dropna()
print(Qual.describe())
# print(Qual.isna().sum())

# ACM
from prince import MCA
mca = MCA(n_components=4, copy=True, check_input=True, engine='auto', random_state=1, benzecri=False)  # M-p puis >1/p
mca = mca.fit(Qual)


# Nombre de composents:
p = mca.n_components

# print(p)

# Valeurs propres MCA
eigen = mca.eigenvalues_
# eigen1=famd.eigenvalues_
eigendf = pd.DataFrame(eigen, columns=["eigen"], index=["Dim {}".format(i) for i in range(1, p + 1, 1)])
inertia = mca.explained_inertia_
inertiadf = pd.DataFrame(inertia, columns=["inertia"], index=["Dim {}".format(i) for i in range(1, p + 1, 1)])

# Valeurs propres acm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
print(eigendf)
print(inertiadf)
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
rowCoord.columns = ['axe1', 'axe2', 'axe3', 'axe4']
print(rowCoord.head())

# print(rowCoord.shape)
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# nuage de points
# indexNames= quanti[ quanti['prix_vente']>0.8*10**7].index
# Delete these row indexes from dataFrame
'''
quanti.drop(indexNames, inplace=True)
print(quanti)
plt.scatter(quanti['age'],quanti['prix_vente'])
plt.xlabel('Age')
plt.ylabel('prix_vente')
plt.show()
'''

# centrer et reduire
'''
scaler = MinMaxScaler()
scaler.fit(rowCoord)
scaler.transform(rowCoord)
'''
# methode des kmeans pour chaque choix de nbr de clusters

'''
# choix de k
# acp et acm font ca
sse = []
k_rng = range(1, 10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(rowCoord)
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)
plt.show()
# par la methode du coude on trouve 2 ou 3 a tester
# methode des kmeans

km = KMeans(n_clusters=4)
from sklearn.metrics import confusion_matrix

classe1 = km.fit(rowCoord)
classe = classe1.labels_
print(classe)
y_predicted = km.fit_predict(rowCoord)
# y_predicted1 = km.fit_predict(rowCoord1)
rowCoord['cluster'] = y_predicted
# rowCoord1['cluster']=y_predicted1
# print(km.cluster_centers_) # centres des classes


# construire chaque dataframe et repre graphique

df1 = rowCoord[rowCoord.cluster == 0]
df2 = rowCoord[rowCoord.cluster == 1]
df3 = rowCoord[rowCoord.cluster == 2]
df4 = rowCoord[rowCoord.cluster == 3]
'''
# represent graphique:
'''plt.scatter(df1.axe1,df1.axe2,color='green')
plt.scatter(df2.axe1,df2.axe2,color='red')
plt.scatter(df3.axe1,df3.axe2,color='black')
#plt.scatter(df4.age,df4['kilometrage'],color='yellow')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('age')
plt.ylabel('kilometrage')
plt.show()
'''
# print(df1.shape)#(17823, 3)
# print(df2.shape)#(17823, 3)
# print(df3.shape)#(27575, 3)
##print(df4.shape)###(2024, 3)


# librairies pour la CAH


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


# df.to_csv('/Users/famille//projetstat/.aws/automobile.csv')