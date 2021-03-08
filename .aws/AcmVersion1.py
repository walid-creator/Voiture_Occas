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
df = pd.read_csv("/Users/famille//projetstat/.aws/df_modelisation.csv", error_bad_lines=False, index_col=0)
df1 = pd.read_csv("/Users/famille//projetstat/.aws/automobile.csv", error_bad_lines=False, index_col=0)
print(df1.columns)
from numpy import unique
#### Extraction des variables qualitatives

Qual = df1[['finition_intens','finition_business', 'finition_limited', 'finition_zen',
             'finition_societe', 'finition_trend', 'finition_rs','finition_expression', 'autre_finition','modele_com']]

#print(Qual.describe())
from numpy import unique
Finition= unique(df['Finition'])
print(Finition)
# print(Qual.isna().sum())
Qual=pd.get_dummies(Qual)
#inertie des individus
n=Qual.shape[0]
p=Qual.shape[1]
ind_moy = numpy.sum(Qual.values,axis=0)/(n*p)
disto_ind = numpy.apply_along_axis(arr=Qual.values,axis=1,func1d=lambda x:numpy.sum(1/ind_moy*(x/p-ind_moy)**2))
#poids des obs.
poids_ind = numpy.ones(Qual.shape[0])/n
#inertie
inertie_ind = poids_ind*disto_ind
#afffichage
#print(pd.DataFrame(numpy.transpose([poids_ind,disto_ind,inertie_ind]),index=Qual.index))

# ACM
from prince import MCA
from fanalysis.ca import CA
mca = MCA(n_components=8, copy=True, check_input=True, engine='auto', random_state=1, benzecri=False)  # M-p puis >1/p
mca.fit(Qual)
mca = CA(row_labels=Qual.index,col_labels=Qual.columns)
mca.fit(Qual.values)
#propriétés
print(dir(mca))

#valeurs propres
print(mca.eig_)
print(pd.DataFrame(numpy.transpose(mca.eig_),index=range(1,14),columns=['Val.P','%inertie','%inertiecumulée']))
#print(pd.DataFrame(mca.row_coord_[:,:8],index=Qual.index,columns=['axe1','axe2','axe3','axe4','axe5','axe6','axe7','axe8']))

#coordonnées fact. des modalités
print(pd.DataFrame(mca.col_coord_[:,:8],index=Qual.columns,columns=['axe1','axe2','axe3','axe4','axe5','axe6','axe7','axe8']))

profil = numpy.apply_along_axis(arr=Qual.values,axis=1,func1d=lambda x:x/numpy.sum(x))
print(pd.DataFrame(profil,index=Qual.index,columns=Qual.columns))


#réduire les profils
profil = profil/numpy.std(profil,axis=0,ddof=0)
print(profil)


#pondération des modalités
somme_col = numpy.sum(Qual.values,axis=0)
pond_moda = (n-somme_col)/(n*p)
print(pond_moda)

# lancer une ACP
from fanalysis.pca import PCA

acp = PCA(std_unit=False, row_labels=Qual.index, col_labels=Qual.columns)
acp.fit(profil)

# valeurs propres
print(acp.eig_[0])

# coordonnées fact. des observations
print(pd.DataFrame(acp.row_coord_[:, :2], index=Qual.index, columns=['Fact.1', 'Fact.2']))


# Nombre de composents:
p = mca.n_components

#print(p)

# Valeurs propres acm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
'''
print(eigendf)
#print(inertiadf)

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
'''
# Coordonnées des individus
'''
rowCoord = mca.row_coordinates(Qual)
rowCoord.columns = ['axe1', 'axe2', 'axe3', 'axe4','axe5','axe6','axe7','axe8']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(rowCoord)
scaler.transform(rowCoord)
print(rowCoord.head())

from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# nuage de points
# indexNames= quanti[ quanti['prix_vente']>0.8*10**7].index
# Delete these row indexes from dataFrame

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


# choix de k
# acp et acm font ca
'''
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

'''
# construire chaque dataframe et repre graphique
'''
df1 = rowCoord[rowCoord.cluster == 0]
df2 = rowCoord[rowCoord.cluster == 1]
df3 = rowCoord[rowCoord.cluster == 2]
df4 = rowCoord[rowCoord.cluster == 3]
'''
# represent graphique:
'''
plt.scatter(df1.axe1,df1.axe2,color='green')
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
'''
df["Finition"] = df["Finition"].replace("Alize", "autre")
df["Finition"] = df["Finition"].replace("Authentique", "autre")
df["Finition"] = df["Finition"].replace("Billabong", "autre")
df["Finition"] = df["Finition"].replace("Campus", "autre")
df["Finition"] = df["Finition"].replace("Dynamique", "autre")
df["Finition"] = df["Finition"].replace("Exception", "autre")
df["Finition"] = df["Finition"].replace("Extreme", "autre")
df["Finition"] = df["Finition"].replace("GT", "autre")
df["Finition"] = df["Finition"].replace("Generation", "autre")
df["Finition"] = df["Finition"].replace("Initiale", "autre")
df["Finition"] = df["Finition"].replace("Life", "autre")
df["Finition"] = df["Finition"].replace("Luxe", "autre")
df["Finition"] = df["Finition"].replace("One", "autre")
df["Finition"] = df["Finition"].replace("Privilege", "autre")
df["Finition"] = df["Finition"].replace("RTA", "autre")
df["Finition"] = df["Finition"].replace("RTE", "autre")
df["Finition"] = df["Finition"].replace("RXE", "autre")
df["Finition"] = df["Finition"].replace("RXT", "autre")
df["Finition"] = df["Finition"].replace("SI", "autre")
df["Finition"] = df["Finition"].replace("Sport", "autre")
df["Finition"] = df["Finition"].replace("Tomtom", "autre")
df["Finition"] = df["Finition"].replace("Trend", "autre")
Qual = df[["Finitions", 'autre_finition']]
'''

# df.to_csv('/Users/famille//projetstat/.aws/automobile.csv')
