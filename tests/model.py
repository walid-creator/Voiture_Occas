from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("/Users/famille//projetstat/.aws/df_modelisation.csv",error_bad_lines=False,index_col=0)
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

df1=df.drop(["prix_vente","horsepower","modele_com","date_mec","options","departement","date_publication","date_depublication","kilometrage"],axis=1)
df1["premiere_main"].replace("oui",1,inplace= True)
df1["premiere_main"].replace("non",0,inplace= True)
df1=pd.get_dummies(df[["energie","boite_de_vitesse","porte","couleur","premiere_main","Finition"]])



# Variables quantitatives
df2=df[["engine","puissance_fiscale","age","horsepower",'axe 1', 'axe 2',
       'axe 3', 'axe 4', 'axe 5', 'axe 6', 'axe 7', 'axe 8', 'axe 9', 'axe 10',
       'axe 11', 'axe 12', 'axe 13']]

# Concaténation
X=pd.concat([df1,df2],axis=1)


#Normer les variables explicatives
scaler = MinMaxScaler(feature_range=(0, 1))
X= scaler.fit_transform(X)

#spliter les données
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=11)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

#faut-il traiter aussi la variable target
'''
y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.fit_transform(y_test)
y_train = pd.DataFrame(y_train_scaled)
y_test = pd.DataFrame(y_test_scaled)
y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.fit_transform(y_test)
'''


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()


#choix de k: nb du plus proche voisin par deux méthodes
#méthode1:
from sklearn.model_selection import validation_curve
train_score,val_score=validation_curve(knn,X_train,y_train,'n_neighbors',params,cv=5)
#méthode2: en utilisant grid search pour le choix de la métrique aussi
import numpy as np
param_grid={'n_neighbors': np.arange(6890,6905),'metric':['euclidean','manhattan']}
from sklearn.model_selection import GridSearchCV
knn = GridSearchCV(knn, param_grid, cv=5)
knn.fit(X_train,y_train)
print(knn.best_params_)
#{'metric': 'manhattan', 'n_neighbors': 6900}

#pourcentage de données utiles à récolter
'''
from sklearn.model_selection import learning_curve
N,train_score,val_score=learning_curve(knn, X_train, y_train, train_sizes=np.linspace(0.1,1.0,10),cv=5 )
print(N)
plt.plot(N,train_score.mean(axis=1),label="train")
plt.plot(N,val_score.mean(axis=1),label="validation")
plt.xlabel("trains_sizes")
plt.legend()
'''


#entrainer le modèle
import math
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(metric= 'manhattan', n_neighbors= 6900)
knn.fit(X_train,y_train)

#prédiction
pred=knn.predict(X_test)

#erreure et score
import math
from sklearn.metrics import mean_squared_error
error = math.sqrt(mean_squared_error(y_test,pred)) #calculate rmse
score= knn.score(X_test,y_test)
print('RMSE value for k= ', 6900 , 'is:', error)
print('score value for k= ', 6900 , 'is:', score)
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.plot(X_test,pred,c='r')

#algorithme backward:
def modeleFin(model,X,y):
       X=X
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=11)
       X_train = pd.DataFrame(X_train)
       X_test = pd.DataFrame(X_test)
       knn = KNeighborsRegressor(metric='manhattan', n_neighbors=6900)
       knn.fit(X_train, y_train)
       score = knn.score(X_test, y_test)


#cas de target qualitative
'''
#le modèle est estimé sur la valeur optimale du paramètre
knn = KNeighborsClassifier(n_neighbors= digit_knn.best_params_["n_neighbors"])
digit_knn=knn.fit(X_train, y_train)
# Estimation de l’erreur de prévision
1-digit_knn.score(X_test,y_test)
# Prévision
y_chap = digit_knn.predict(X_test)
# matrice de confusion
table=pd.crosstab(y_test,y_chap)
print(table)
matshow(table)
title("Matrice de Confusion")
colorbar()
show()
'''
