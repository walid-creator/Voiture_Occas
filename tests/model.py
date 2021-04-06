from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("/Users/famille//projetstat/.aws/df_modelisation.csv",error_bad_lines=False,index_col=0)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#suppression des prix>10^5
indexNames3 = df[ df["prix_vente"]  >10**5].index
df.drop(indexNames3, inplace=True)


#df=df.head(1000)
y=df["prix_vente"]


#correlation
'''
print(df[[u"kilometrage", u"prix_vente"]].corr())#il est bien negative mais faible -0.051
print(df[[u"puissance_fiscale", u"prix_vente"]].corr())#0.022
print(df[[u"horsepower", u"prix_vente"]].corr())#0.035
print(df[[u"age", u"prix_vente"]].corr())# -0.04
print(df[[u"prix_vente", u"engine"]].corr())# -0.01
print(df[[u"age", u"kilometrage",u"puissance_fiscale",u"horsepower","engine"]].corr())#forte corrélation:age&kilometrage=0.76 et puissance_fiscale&horsepower=0.81
'''

#suppression de var corr et prix de vente

df1=df.drop(["prix_vente","horsepower","modele_com","date_mec","options","departement","date_publication","date_depublication","age"],axis=1)
#transformation des valeurs de portes en str pour que ca soit traité comme une var quali
df1['porte'] = df1['porte'].astype(str)
df1=pd.get_dummies(df1[["energie","boite_de_vitesse","couleur","premiere_main","Finition","porte"]])



# Variables quantitatives
df2=df[["engine","age","horsepower",'axe 1', 'axe 2',
       'axe 3', 'axe 4', 'axe 5', 'axe 6', 'axe 7', 'axe 8', 'axe 9', 'axe 10',
       'axe 11', 'axe 12', 'axe 13']]



# Concaténation
X=pd.concat([df1,df2],axis=1)

#Normer les variables explicatives quanti
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X[["engine","age","horsepower",'axe 1', 'axe 2',
       'axe 3', 'axe 4', 'axe 5', 'axe 6', 'axe 7', 'axe 8', 'axe 9', 'axe 10',
       'axe 11', 'axe 12', 'axe 13']]=scaler.fit_transform(X[["engine","age","horsepower",'axe 1', 'axe 2',
       'axe 3', 'axe 4', 'axe 5', 'axe 6', 'axe 7', 'axe 8', 'axe 9', 'axe 10',
       'axe 11', 'axe 12', 'axe 13']])


#y=y.to_frame()
#y=scaler.fit_transform(y)
#y = pd.DataFrame(y,columns=['prix_vente'])

#spliter les données
import numpy as np
X_train,X_test,y_train,y_test=train_test_split(X,np.log(y),test_size=0.25,random_state=11)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)



from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()


#choix de k: nb du plus proche voisin par deux méthodes
#méthode1:


param_grid={'n_neighbors': np.arange(1,50),'metric': ['manhattan','euclidean']}
from sklearn.model_selection import GridSearchCV
knn = GridSearchCV(knn, param_grid, cv=5)
knn.fit(X_train,y_train)
print(knn.best_params_)




#entrainer le modèle
import math
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(metric= 'manhattan', n_neighbors= 6,weights='distance')# les prix ne sont pas uniformement distribués
knn.fit(X_train,y_train)
print(knn.score(X_test,y_test))
#prédiction
pred=knn.predict(X_test)

#erreure et score
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
error = math.sqrt(mean_squared_error(y_test,pred)) #calculate rmse
score= knn.score(X_test,y_test)
mape=mean_absolute_percentage_error(y_test,pred)




#erreure quadratique moyenne
print('RMSE value for k= ', 6 , 'is:', error)
#sans traitement=1578.0976223381065
#log=0.17273758688079002
#norm=0.4089707745255511 //c'est normal car on est oentre 0 et 1 et la différence au carré est de plus en plus en petite

#R^2
print('Score value for k= ', 6 , 'is:', score)
#sans traitement=0.8357542311811924
#log=0.8262774212423686
#norm=0.8357542311811924

#erreure en valeur absolue
print('MAPE value for k= ', 6 , 'is:', mape)
#sans traitement=12.423715479695412%
#log=1.3146505898190326%
#norm=128.825381 %





from sklearn.feature_selection import SequentialFeatureSelector
#selection des variables les plus impactantes avec l'algorithme forward
import numpy as np

#recherche par validation croisée du meilleur paramètre du nombre de variables à sélectionner
sfw = SequentialFeatureSelector(knn,direction='forward')
sfw.fit(X_train,y_train)
print(sfw.get_support())
print(sfw.transform(X).shape)
pos=[]
for i in range(45):
    if sfw.get_support()[i]==True:
        pos.append(i)
colnameB = X.columns[pos]
print(colnameB)

#selection des variables les plus impactantes avec l'algorithme backward
'''
sbw = SequentialFeatureSelector(knn, n_features_to_select= ,direction='backward')
sbw.fit(X_train,y_train)
print(sbw.get_support())
print(sbw.transform(X).shape)
pos=[]
for i in range(45):
    if sfw.get_support()[i]==True:
        pos.append(i)
colnamef = X.columns[pos]
'''



