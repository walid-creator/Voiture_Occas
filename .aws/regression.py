import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV
from sklearn.model_selection import RepeatedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


### avec statsmodels

table = pd.read_csv('D:/Projet stat 2A/projet_stat/.aws/df_modelisation.csv')
table = table.loc[table["prix_vente"]<100000,:]

XTrain = table.drop(["prix_vente","date_mec","options","date_publication","date_depublication","departement"], axis = 1)
XTrain = XTrain.drop(["modele_com","energie","boite_de_vitesse","couleur","premiere_main","Finition","porte"], axis = 1)
XTrain = XTrain.iloc[0:len(XTrain),1:21]
XTrain = sm.add_constant(XTrain)

qualcol = ["energie", "boite_de_vitesse","couleur", "premiere_main", "Finition","porte"]
qual = table[qualcol].astype(str)
tdc = pd.get_dummies(qual)

data = pd.concat([XTrain, tdc], axis = 1)

# On rend le modèle identifiable en enlevant une indicatrice pour chaque variable catégorielle
# data = data.drop(["energie_Hybride essence électrique","boite_de_vitesse_auto","couleur_orange","premiere_main_oui","Finition_Expression","porte_0.0"], axis = 1)


print(XTrain.head())

YTrain = table["prix_vente"]
YTrain = np.log(YTrain)

xTrain, xTest, yTrain, yTest = train_test_split(data, YTrain, test_size = 0.3, random_state = 5)

plt.hist(yTest, edgecolor='black', bins=100 , density=False, color='red')
plt.show()

regOLS = sm.OLS(yTrain,xTrain)
resReg = regOLS.fit()


print(resReg.summary())

### Indices échantillon test

yPred_test = resReg.predict(xTest)

MAE_test = mean_absolute_error(np.exp(yPred_test),np.exp(yTest))
MAPE_test = mean_absolute_percentage_error(np.exp(yPred_test),np.exp(yTest))
MSE_test = mean_squared_error(np.exp(yPred_test),np.exp(yTest))
RMSE_test = np.sqrt(MSE_test)
ecart_type_test = (abs((np.exp(yTest) - np.exp(yPred_test))/np.exp(yTest))-MAPE_test)**2
ecart_type_test = np.sqrt(ecart_type_test.sum()/len(yTest))
ecart_type_test = (((np.exp(yTest) - np.exp(yPred_test))**2/np.exp(yTest))-MAPE_test)**2
ecart_type_test = np.sqrt(ecart_type_test.sum()/len(yTest))
CV_test = ecart_type_test / MAPE_test
moy_yTest = (np.exp(yTest).sum())/len(yTest)
R2_test = 1 - (((np.exp(yTest) - np.exp(yPred_test))**2).sum()) / (((np.exp(yTest) - moy_yTest)**2).sum())

print(MAE_test,MAPE_test,RMSE_test,ecart_type_test,CV_test,R2_test)

### Indices échantillon train

yPred_train = resReg.predict(xTrain)

MAE_train = mean_absolute_error(np.exp(yPred_train),np.exp(yTrain))
MAPE_train= mean_absolute_percentage_error(np.exp(yPred_train),np.exp(yTrain))
MSE_train = mean_squared_error(np.exp(yPred_train),np.exp(yTrain))
RMSE_train = np.sqrt(MSE_train)
ecart_type_train = (abs((np.exp(yTrain) - np.exp(yPred_train))/np.exp(yTrain))-MAPE_train)**2
ecart_type_train = np.sqrt(ecart_type_train.sum()/len(yTrain))
CV_train = ecart_type_train / MAPE_train
moy_yTrain = yTrain.sum() / len(yTrain)
R2_train = 1 - (((np.exp(yTrain) - np.exp(yPred_train))**2).sum()) / (((np.exp(yTrain) - moy_yTrain)**2).sum())

print(MAE_train,MAPE_train,RMSE_train,ecart_type_train,CV_train,R2_train)

### Parcours forward sur le CP_Mallows

X1Train = xTrain["const"]
X1Test = xTest["const"]
X2Train = xTrain.drop(["const"],axis = 1)
X2Test = xTest.drop(["const"],axis = 1)
n = X2Train.shape[1]
CP_Mallows = 10**10
j = 0
v = True
while v and j < n:
    print(j)
    CP_Mallows_j = 10**10
    meilleur_indice = 0
    m = n - j
    for i in range(m) :
        XTrain = X2Train.iloc[:,i]
        XTest = X2Test.iloc[:,i]
        X3Train = pd.concat([X1Train, XTrain], axis=1)
        X3Test = pd.concat([X1Test, XTest], axis=1)
        regOLS = sm.OLS(yTrain, X3Train)
        resReg = regOLS.fit()
        yPred = resReg.predict(X3Test)
        SSE = ((np.exp(yTest) - np.exp(yPred)) ** 2).sum()
        CP_Mallows_j_i = (SSE/MSE_test) - X3Train.shape[0] + 2*(X3Train.shape[1]+1)
        if CP_Mallows_j_i < CP_Mallows_j :
            meilleur_indice = i
            CP_Mallows_j = CP_Mallows_j_i
    if CP_Mallows_j < CP_Mallows :
        XjTrain = X2Train.iloc[:,meilleur_indice]
        XjTest = X2Test.iloc[:,meilleur_indice]
        X1Train = pd.concat([X1Train, XjTrain], axis=1)
        X1Test = pd.concat([X1Test, XjTest], axis=1)
        X2Train = X2Train.drop(columns = X2Train.columns[meilleur_indice])
        CP_Mallows = CP_Mallows_j
        j += 1
    else :
        v = False

regOLS = sm.OLS(yTrain, X1Train)
resReg = regOLS.fit()
print(resReg.summary())

yPred = resReg.predict(X1Test)

mape_reg = abs((np.exp(yTest) - np.exp(yPred))/np.exp(yTest))
n = len(mape_reg)
MAPE_reg = (mape_reg.sum())/n
SE = (np.exp(yTest) - np.exp(yPred))**2
MSE = (SE.sum())/n
RMSE = np.sqrt((SE.sum())/n)

ecart_type_mape = (mape_reg-MAPE_reg)**2
ecart_type_mape = np.sqrt(ecart_type_mape.sum()/n)

print(MAPE_reg,ecart_type_mape,RMSE)
print(ecart_type_mape/MAPE_reg)

### Parcours forward sur le MAPE

X1Train = xTrain["const"]
X1Test = xTest["const"]
X2Train = xTrain.drop(["const"],axis = 1)
X2Test = xTest.drop(["const"],axis = 1)
n = X2Train.shape[1]
CV = 10**10
j = 0
v = True
while v and j < n:
    print(j)
    CV_j = 10**10
    meilleur_indice = 0
    m = n - j
    for i in range(m) :
        XTrain = X2Train.iloc[:,i]
        XTest = X2Test.iloc[:,i]
        X3Train = pd.concat([X1Train, XTrain], axis=1)
        X3Test = pd.concat([X1Test, XTest], axis=1)
        regOLS = sm.OLS(yTrain, X3Train)
        resReg = regOLS.fit()
        yPred = resReg.predict(X3Test)
        mape_reg = abs((np.exp(yTest) - np.exp(yPred))/np.exp(yTest))
        MAPE_j_i = (mape_reg.sum()) / len(mape_reg)
        ecart_type_j_i = (mape_reg - MAPE_reg) ** 2
        ecart_type_j_i = np.sqrt(ecart_type_mape.sum() / n)
        CV_j_i = MAPE_j_i / ecart_type_j_i
        if CV_j_i < CV_j :
            meilleur_indice = i
            CV_j = CV_j_i
    print(CV_j)
    if CV_j < CV :
        XjTrain = X2Train.iloc[:,meilleur_indice]
        XjTest = X2Test.iloc[:,meilleur_indice]
        X1Train = pd.concat([X1Train, XjTrain], axis=1)
        X1Test = pd.concat([X1Test, XjTest], axis=1)
        X2Train = X2Train.drop(columns = X2Train.columns[meilleur_indice])
        CV = CV_j
        j += 1
    else :
        v = False

regOLS = sm.OLS(yTrain, X1Train)
resReg = regOLS.fit()
print(resReg.summary())

yPred = resReg.predict(X1Train)

MAPE = mean_absolute_percentage_error(np.exp(yPred),np.exp(yTrain))
MAE = mean_absolute_error(np.exp(yPred),np.exp(yTrain))
SE = (np.exp(yTrain) - np.exp(yPred))**2
MSE = (SE.sum())/n
RMSE = np.sqrt((SE.sum())/n)
moy_yTrain = yTrain.sum() / len(yTrain)
R2 = 1 - (((np.exp(yTrain) - np.exp(yPred))**2).sum()) / (((np.exp(yTrain) - moy_yTrain)**2).sum())
ecart_type = (((np.exp(yTrain) - np.exp(yPred))**2/np.exp(yTrain))-MAPE)**2
ecart_type = np.sqrt(ecart_type.sum()/len(yTrain))
CV = ecart_type / MAPE

print(MAPE,ecart_type,RMSE,MAE,R2,CV)
print(ecart_type_mape/MAPE_reg)

### Parcours backward sur le MAPE

X1Train = xTrain["const"]
X1Test = xTest["const"]
X2Train = xTrain
X2Test = xTest
n = X2Train.shape[1]
RMSE = 10**10
j = 0
v = True
while v and j < n:
    print(j)
    RMSE_j = 10**10
    meilleur_indice = 0
    m = n - j
    for i in range(m) :
        X3Train = X2Train.drop(columns = X2Train.columns[i])
        X3Test = X2Test.drop(columns = X2Test.columns[i])
        regOLS = sm.OLS(yTrain, X3Train)
        resReg = regOLS.fit()
        yPred = resReg.predict(X3Test)
        SE = (np.exp(yTest) - np.exp(yPred)) ** 2
        MSE = (SE.sum()) / len(SE)
        RMSE_j_i = np.sqrt(MSE)
        if RMSE_j_i < RMSE_j :
            meilleur_indice = i
            RMSE_j = RMSE_j_i
    print(RMSE_j)
    if RMSE_j < RMSE :
        XjTrain = X2Train.iloc[:,meilleur_indice]
        XjTest = X2Test.iloc[:,meilleur_indice]
        X1Train = pd.concat([X1Train, XjTrain], axis=1)
        X1Test = pd.concat([X1Test, XjTest], axis=1)
        X2Train = X2Train.drop(columns = X2Train.columns[meilleur_indice])
        X2Test = X2Test.drop(columns=X2Test.columns[meilleur_indice])
        RMSE = RMSE_j
        j += 1
    else :
        v = False

regOLS = sm.OLS(yTrain, X2Train)
resReg = regOLS.fit()
print(resReg.summary())

yPred = resReg.predict(X2Test)

mape_reg = abs((np.exp(yTest) - np.exp(yPred))/np.exp(yTest))
n = len(mape_reg)
MAPE_reg = (mape_reg.sum())/n
SE = (np.exp(yTest) - np.exp(yPred))**2
MSE = (SE.sum())/n
RMSE = np.sqrt((SE.sum())/n)

ecart_type_mape = (mape_reg-MAPE_reg)**2
ecart_type_mape = np.sqrt(ecart_type_mape.sum()/n)

print(MAPE_reg,RMSE)
print(ecart_type_mape/MAPE_reg)

### Resultat backward CV : [MAPE,RMSE,CV] = [0.09813903253502688,1507.4281732291759,1.1497504722912784]
### Resultat backward MAPE : [MAPE,RMSE,CV] = [0.09813903253502688,1507.4281732291759,1.1497504722912784]

### Regression Ridge

alphas = np.arange(0, 10, 0.01)
alphas = alphas[1:]

ridgecv = RidgeCV(alphas = alphas, scoring = 'neg_mean_absolute_percentage_error', normalize = True)
ridgecv.fit(xTrain, yTrain)
ridgecv.alpha_

ridge4 = Ridge(alpha = ridgecv.alpha_, normalize = True)
ridge4.fit(xTrain, yTrain)
mean_absolute_percentage_error(yTest, ridge4.predict(xTest))

ridge4 = Ridge(alpha = 0, normalize = False)
ridge4.fit(xTrain, yTrain)
mean_absolute_percentage_error(yTest, ridge4.predict(xTest))


