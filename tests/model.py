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
df1=df.drop(["prix_vente","age","horsepower"],axis=1)
df1=pd.get_dummies(df[["Finition","couleur","boite_de_vitesse","energie"]])
df1.head()



# Variables quantitatives
df2=df[["age","horsepower"]]

# Concaténation
X=pd.concat([df1,df2],axis=1)

#Preprocessing – Scaling the features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=11)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
X_train = pd.DataFrame(X_train_scaled)
X_test = pd.DataFrame(X_test_scaled)
y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.fit_transform(y_test)
#y_train = pd.DataFrame(y_train_scaled)
y_test = pd.DataFrame(y_test_scaled)
#y_train_scaled = scaler.fit_transform(y_train)
#y_test_scaled = scaler.fit_transform(y_test)
print(X_train.head())
print(X_test.head())

from sklearn.model_selection import GridSearchCV
params = {'n_neighbors':[9,10,11,12,13]}
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(5)
''''
model = GridSearchCV(knn, params, cv=5)
model.fit(X_train,y_train)
print(model.best_params_)
'''
import math
knn.fit(X_train,y_train)
# Estimation de l’erreur de prévision sur l’échantillon test
#fit the model
pred=knn.predict(X_test) #make prediction on test set
from sklearn.metrics import mean_squared_error
error = math.sqrt(mean_squared_error(y_test,pred)) #calculate rmse
print('RMSE value for k= ', 5 , 'is:', error)




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
#df.to_csv("/Users/famille//projetstat/.aws/df_modelisation.csv")