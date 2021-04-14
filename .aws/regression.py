import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


table = pd.read_csv('D:/Projet stat 2A/projet_stat/.aws/df_modelisation.csv')

y = table["prix_vente"].to_numpy()
x = table.drop(["prix_vente","date_mec","options","date_publication","date_depublication","departement"], axis = 1)
x = x.drop(["age","puissance_fiscale"], axis = 1)
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
YTrain = np.log(YTrain)

xTrain, xTest, yTrain, yTest = train_test_split(XTrain, YTrain, test_size = 0.3, random_state = 5)

reg = sm.OLS(yTrain,xTrain)
resReg = reg.fit()

print(resReg.summary())

yPred = resReg.predict(xTest)

mape_reg = abs((np.exp(yTest) - np.exp(yPred))/np.exp(yTest))
n = len(mape_reg)
MAPE_reg = (mape_reg.sum())/n

ecart_type_mape = (mape_reg-MAPE_reg)**2
ecart_type_mape = np.sqrt(ecart_type_mape.sum()/n)

print(MAPE_reg,ecart_type_mape)



