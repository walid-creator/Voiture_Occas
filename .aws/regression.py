import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
import statsmodels.api as sm


table = pd.read_csv('D:/Projet stat 2A/projet_stat/.aws/df_modelisation.csv')

y = table["prix_vente"].to_numpy()
x = table.drop(["prix_vente","date_mec","options","date_publication","date_depublication","departement"], axis = 1)
x= x.drop(["age","puissance_fiscale"], axis = 1)
x_quanti = x.drop(["modele_com","energie","boite_de_vitesse","couleur","premiere_main","Finition"], axis = 1)
x_quanti = x_quanti.iloc[0:len(x_quanti),1:21]
x_quanti = x_quanti.to_numpy()
x_quanti = normalize(x_quanti, axis=0, norm='max')

model = LinearRegression()
model.fit(x_quanti,y)

r_sq = model.score(x_quanti,y)
print('coefficient of determination:', r_sq)

### avec statsmodels

table = pd.read_csv('D:/Projet stat 2A/projet_stat/.aws/df_modelisation.csv')
table = table.loc[table["prix_vente"]<100000,:]

XTrain = table.drop(["prix_vente","date_mec","options","date_publication","date_depublication","departement"], axis = 1)
XTrain = XTrain.drop(["age","puissance_fiscale"], axis = 1)
XTrain = XTrain.drop(["modele_com","energie","boite_de_vitesse","couleur","premiere_main","Finition"], axis = 1)
XTrain = XTrain.iloc[0:len(XTrain),1:21]
XTrain = sm.add_constant(XTrain)

print(XTrain.head())

YTrain = table["prix_vente"]

reg = sm.OLS(YTrain,XTrain)
resReg = reg.fit()

print(resReg.summary())

### On enleve l'axe 1

XTrain = XTrain.drop(["axe 1","axe 4"], axis = 1)

reg = sm.OLS(YTrain,XTrain)
resReg = reg.fit()

print(resReg.summary())


model = LinearRegression()
model.fit(XTrain,YTrain)

r_sq = model.score(XTrain,YTrain)
print('coefficient of determination:', r_sq)




