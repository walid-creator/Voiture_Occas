from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("/Users/famille//projetstat/.aws/df_modelisation.csv",error_bad_lines=False,index_col=0)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#suppression des prix>10^5
indexNames3 = df[ df["prix_vente"]  >10**5].index
df.drop(indexNames3, inplace=True)
n=df.shape[0]

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
df2=df[["engine","kilometrage","horsepower",'axe 1', 'axe 2',
       'axe 3', 'axe 4', 'axe 5', 'axe 6', 'axe 7', 'axe 8', 'axe 9', 'axe 10',
       'axe 11', 'axe 12', 'axe 13']]



# Concaténation
X=pd.concat([df1,df2],axis=1)

#Normer les variables explicatives quanti
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X[["engine","kilometrage","horsepower",'axe 1', 'axe 2',
       'axe 3', 'axe 4', 'axe 5', 'axe 6', 'axe 7', 'axe 8', 'axe 9', 'axe 10',
       'axe 11', 'axe 12', 'axe 13']]=scaler.fit_transform(X[["engine","kilometrage","horsepower",'axe 1', 'axe 2',
       'axe 3', 'axe 4', 'axe 5', 'axe 6', 'axe 7', 'axe 8', 'axe 9', 'axe 10',
       'axe 11', 'axe 12', 'axe 13']])

#avec l'algo forward on sélectionnes les varaibles suivantes:
'''X=X[['energie_Essence', 'boite_de_vitesse_meca', 'couleur_blanc',
       'premiere_main_non', 'Finition_Societe', 'porte_3.0', 'engine',
       'kilometrage', 'horsepower', 'axe 1']]'''
#y=y.to_frame()
#y=scaler.fit_transform(y)
#y = pd.DataFrame(y,columns=['prix_vente'])

#spliter les données
import numpy as np
X_train,X_test,y_train,y_test=train_test_split(X,np.log(y),test_size=0.3,random_state=11)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)



from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()


#choix de k: nb du plus proche voisin par deux méthodes
#méthode1:


param_grid={'n_neighbors': np.arange(1,50),'metric': ['manhattan','euclidean'],'weights':['distance','uniform']}
from sklearn.model_selection import GridSearchCV
knn = GridSearchCV(knn, param_grid, cv=5)
knn.fit(X_train,y_train)
print(knn.best_params_)




#entrainer le modèle
import math
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(metric= 'euclidean', n_neighbors= 9,weights='distance')# les prix ne sont pas uniformement distribués
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
mape=mean_absolute_percentage_error(np.exp(y_test),np.exp(pred))




#erreure quadratique moyenne
print('RMSE value for k= ', 9 , 'is:', error)
#sans traitement=1578.0976223381065
#log=0.21847782477102562//dernier
#norm=0.4089707745255511 //c'est normal car on est oentre 0 et 1 et la différence au carré est de plus en plus en petite

#R^2
print('Score value for k= ', 9 , 'is:', score)
#sans traitement=0.8357542311811924
#log=0.7278853117792159//dernier
#norm=0.8357542311811924

#erreure en valeur absolue
print('MAPE value for k= ', 9 , 'is:', mape)
#sans traitement=12.423715479695412%
#log=0.016168859332416144//dernier
#norm=128.825381 %

#to sum up:
#RMSE value for k=  9 is: 0.21847782477102562
#Score value for k=  9 is: 0.7236276765367943
#MAPE value for k=  9 is: 0.16304824258851666



from sklearn.feature_selection import SequentialFeatureSelector
#selection des variables les plus impactantes avec l'algorithme forward

#recherche par validation croisée du meilleur paramètre du nombre de variables à sélectionner//pas réussi
import numpy as np
from sklearn.metrics import r2_score, make_scorer
#ftwo_scorer = make_scorer(fbeta_score, beta=2)
'''
param_grid1={'n_features_to_select': np.arange(1,2)}
from sklearn.model_selection import GridSearchCV
sfw = SequentialFeatureSelector(knn,direction='forward')
sfw = GridSearchCV(sfw, param_grid1,scoring=r2_score, cv=5)
sfw.fit(X_train,y_train)
print(sfw.best_params_)
'''

#lancement de l'algorithme
sfw = SequentialFeatureSelector(knn,direction='forward',n_features_to_select=10)
sfw.fit(X_train,y_train)
print(sfw.get_support())
print(sfw.transform(X).shape)
pos=[]
for i in range(45):
    if sfw.get_support()[i]==True:
        pos.append(i)
colnamef = X.columns[pos]
print(colnameB)
#RMSE value for k=  9 is: 0.15784476672070805
#Score value for k=  9 is: 0.8557419031104687
#MAPE value for k=  9 is: 0.11021248043273771
kselct=10
k=X.shape[1]
R2_ajsuté=1 - (1-score)*(n-1)/(n-k-1)#0.7276518660749811
R2_ajusté_forw= 1 - (1-score)*(n-1)/(n-kselct-1)#0.8580747705147141

print('R2_ajsuté value for k= ', 10 , 'is:', R2_ajsuté)
print('R2_ajusté_forw value for k= ', 10 , 'is:', R2_ajusté_forw)


#selection des variables les plus impactantes avec l'algorithme backward
'''
sbw = SequentialFeatureSelector(knn, n_features_to_select=10 ,direction='backward')
sbw.fit(X_train,y_train)
print(sbw.get_support())
print(sbw.transform(X).shape)
pos=[]
for i in range(45):
    if sfw.get_support()[i]==True:
        pos.append(i)
colnameb = X.columns[pos]
print(colnameb)
'''



