import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV
from sklearn.model_selection import RepeatedKFold


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
XTrain = XTrain.drop(["modele_com","energie","boite_de_vitesse","couleur","premiere_main","Finition","porte"], axis = 1)
# XTrain = XTrain.drop(["age","horsepower"], axis = 1)
XTrain = XTrain.iloc[0:len(XTrain),1:21]
XTrain = sm.add_constant(XTrain)

qualcol = ["energie", "boite_de_vitesse","couleur", "premiere_main", "Finition","porte"]
qual = table[qualcol].astype(str)
tdc = pd.get_dummies(qual)

data = pd.concat([XTrain, tdc], axis = 1)

print(XTrain.head())

YTrain = table["prix_vente"]
YTrain = np.log(YTrain)

xTrain, xTest, yTrain, yTest = train_test_split(data, YTrain, test_size = 0.3, random_state = 5)

regOLS = sm.OLS(yTrain,xTrain)
resReg = regOLS.fit()

print(resReg.summary())

modelRidge = Lasso(alpha=0.01)
resRidge = modelRidge.fit(xTrain,yTrain)

print(resRidge.score(xTrain,yTrain))

### Indices échantillon test

yPred = resReg.predict(xTest)

mape_reg = abs((np.exp(yTest) - np.exp(yPred))/np.exp(yTest))
n = len(mape_reg)
MAPE_reg = (mape_reg.sum())/n
SE = (np.exp(yTest) - np.exp(yPred))**2
RMSE = np.sqrt((SE.sum())/n)

ecart_type_mape = (mape_reg-MAPE_reg)**2
ecart_type_mape = np.sqrt(ecart_type_mape.sum()/n)

print(MAPE_reg,ecart_type_mape,RMSE)
print(ecart_type_mape/MAPE_reg)

### Indices échantillon d'apprentissage


### Regression Lasso

table = pd.read_csv('D:/Projet stat 2A/projet_stat/.aws/df_modelisation.csv')

delete = ["options","date_mec","date_publication","date_depublication","departement"]
newdf = table.drop(delete, axis = 1)

target = newdf["prix_vente"]
features = newdf.drop("prix_vente", axis = 1)
logtarget = target.apply(lambda x : np.log(x))

# Subdivision
axe = ["axe {}".format(i) for i in range(1,14)]
qualcol = ["modele_com", "energie", "boite_de_vitesse","couleur", "premiere_main", "Finition"]
quantcol = ["horsepower","engine", "kilometrage", "puissance_fiscale", "age","porte"]
axedf = newdf[axe]
quant = newdf[quantcol]
# dummies variables
qual = newdf[qualcol].astype(str)
tdc = pd.get_dummies(qual)
# Concatenate
data = pd.concat([quant, axedf, tdc], axis = 1)

XTrain, XTest, yTrain, yTest = train_test_split(data, logtarget, test_size = 0.3, random_state = 5)

# Extraction
quanttrain = XTrain[quantcol]
quanttest = XTest[quantcol]

# Suppresion
axetdctrain = XTrain.drop(quantcol, axis = 1)
axetdctest = XTest.drop(quantcol, axis = 1)

# StandardScaler train set and test set
from sklearn.preprocessing import StandardScaler
stdSc = StandardScaler()
quantTrain = pd.DataFrame(stdSc.fit_transform(quanttrain), index = quanttrain.index,
                          columns = quanttrain.columns)
quantTest = pd.DataFrame(stdSc.transform(quanttest), index = quanttest.index,
                         columns = quanttest.columns)

# Concaténation
Xtrain = pd.concat([quantTrain,axetdctrain], axis = 1)
Xtest = pd.concat([quantTest, axetdctest], axis = 1)

from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

def score_model(estimators, X_train, y_train, X_test, y_test, **kwargs):
    """
        evaluation sur plusieurs critères
    """
    results = {}
    index = ["r2 train", "r2 test", "rmse train", "rmse test", "mape train", "mape test",
            "exp. var. score train", "exp. var. score test"]
    for model in estimators.values():
        y_train_pred = model.fit(X_train, y_train).predict(X_train)
        y_test_pred = model.fit(X_train, y_train).predict(X_test)
        r2_train = r2_score(np.exp(y_train), np.exp(y_train_pred))
        r2_test = r2_score(np.exp(y_test),np.exp(y_test_pred))
        rmse_train =np.sqrt(mean_squared_error(np.exp(y_train),np.exp(y_train_pred)))
        rmse_test = np.sqrt(mean_squared_error(np.exp(y_test),np.exp(y_test_pred)))
        mape_train = mean_absolute_percentage_error(np.exp(y_train),np.exp(y_train_pred))
        mape_test = mean_absolute_percentage_error(np.exp(y_test),np.exp(y_test_pred))
        evs_train = explained_variance_score(np.exp(y_train),np.exp(y_train_pred))
        evs_test = explained_variance_score(np.exp(y_test),np.exp(y_test_pred))
        results[model] = [r2_train, r2_test, rmse_train, rmse_test, mape_train, mape_test,
                         evs_train, evs_test]
    return pd.DataFrame(results, index = index)

from sklearn.linear_model import LinearRegression, Lasso, Ridge
models = {'OLS': LinearRegression(),
         'Lasso': Lasso(alpha = 0.00001),
         'Ridge': Ridge(),}
score_model(models, Xtrain, yTrain, Xtest, yTest)


