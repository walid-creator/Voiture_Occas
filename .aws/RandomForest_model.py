#!/usr/bin/env python
# coding: utf-8

# # Prévision

# In[1]:


# Repertoire de travail
from os import chdir

path = "C:\\Users\\duver\\Desktop\\2A_ENSAI\\SEMESTRE 2\\ProjetStatistiques\\codeprojet"
chdir(path)

# In[2]:


# Chargement du data set
import pandas as pd

df = pd.read_csv("aws_df_modelisation.csv", header=0, index_col=0)

# In[3]:


## Variables inutiles
axe = ["axe {}".format(i) for i in range(1, 14)]
delete = ["options", "date_mec", "date_publication", "date_depublication", "departement"]
newdf = df.drop(delete, axis=1)
newdf = newdf.drop(axe, axis=1)
newdf = newdf[newdf["prix_vente"] <= 100000]

# In[4]:


#
import seaborn as sns

quant = newdf[["prix_vente", "horsepower", "engine", "kilometrage", "puissance_fiscale", "age"]]
matrice_corr = quant.corr().round(2)
sns.heatmap(data=matrice_corr, annot=True)

# In[5]:


# Subdivision en target et features
target = newdf["prix_vente"]
# Extraction
import numpy as np

logtarget = np.log(target)

# In[6]:


# Subdivision
# axe = ["axe {}".format(i) for i in range(1,14)]
qualcolumns = ["energie", "boite_de_vitesse", "couleur", "premiere_main", "Finition", "porte"]
quantcolumns = ["horsepower", "engine", "kilometrage"]
# Train set
qual_var = newdf[qualcolumns].astype(str)
# axe_var = newdf[axe]
quant_var = newdf[quantcolumns]

# ### StandardScaler and dummies variables

# In[7]:


# StandardScaler train set and test set
from sklearn.preprocessing import StandardScaler

stdSc = StandardScaler()
quant_scale = pd.DataFrame(stdSc.fit_transform(quant_var), index=quant_var.index,
                           columns=quant_var.columns)
# codage en 0/1 des variables qualitatives
tdc = pd.get_dummies(qual_var)
# Concaténation
features = pd.concat([quant_scale, tdc], axis=1)
print(features.shape)

# In[8]:


# train set et test set
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(features, logtarget, test_size=0.3, random_state=5)

# #### Score model

# In[9]:


# Score model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score


# Calcul de l'écart type
def get_sd_error(y_true, y_pred):
    n = len(y_true)
    somme = 0.0
    mean = mean_absolute_percentage_error(y_true, y_pred)
    for i in range(n):
        somme += ((np.abs((y_true[i] - y_pred[i]) / y_true[i])) - mean) ** 2
    return (np.sqrt(somme / n))


# Evaluation
def evaluate(y_train, y_test, y_pred_train, y_pred_test):
    # Evaluation sur le train set
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    # Evaluation sur le test set
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    # Ecart type
    sd_train = get_sd_error(y_train.values, y_pred_train)
    sd_test = get_sd_error(y_test.values, y_pred_test)

    # Coefficient de variation
    cv_train = sd_train / mape_train
    cv_test = sd_test / mape_test

    # Affichage
    result_train = [rmse_train, mae_train, mape_train, r2_train, cv_train]
    result_test = [rmse_test, mae_test, mape_test, r2_test, cv_test]
    result = pd.DataFrame(np.transpose([result_train, result_test]),
                          columns=["train_set", "test_set"],
                          index=["RMSE", "MAE", "MAPE", "r2 score", "coef. var. (MAPE)"])
    return result


# ### Random Forest et Bagging

# In[10]:


# Importation de la classe de calcul
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

# Instanciation
model_tree = DecisionTreeRegressor(random_state=0, criterion="mse", ccp_alpha=1.4341231800350741e-05,
                                   min_samples_leaf=0.001)
model_bag = BaggingRegressor(model_tree, n_estimators=100)

# Entraînement
model_bag.fit(Xtrain, ytrain)
# Prédiction sur le bag
ypred_train_bag = np.exp(model_bag.predict(Xtrain))
ypred_test_bag = np.exp(model_bag.predict(Xtest))
# Evaluation
evaluate(np.exp(ytrain), np.exp(ytest), ypred_train_bag, ypred_test_bag)

# In[11]:


# Importance des variables
feature_imp = np.mean([tree.feature_importances_ for tree in model_bag.estimators_], axis=0)
bag_importance = pd.DataFrame(feature_imp, columns=["importance"],
                              index=Xtrain.columns)
sort_bag = bag_importance.sort_values(by="importance", ascending=False)
sort_bag.head(10)

# In[12]:


# Représentation graphique
from matplotlib import pyplot as plt

n_feat = Xtrain.shape[1]
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(range(1, n_feat + 1), np.cumsum(bag_importance.values))
ax.set_title("Cumsum importance feature with bagging model")
ax.set_xlabel("nb. features")
ax.set_ylabel("feature importance")
ax.grid()
plt.savefig("feat_bag.png")
plt.show()

# In[13]:


#  Random Forest
from sklearn.ensemble import RandomForestRegressor

# Instanciation
model_rfr = RandomForestRegressor(random_state=0, ccp_alpha=1.4341231800350741e-05,
                                  min_samples_leaf=0.001)
# Entraînement
model_rfr.fit(Xtrain, ytrain)
# Prédiction sur le bag
ypred_train_rfr = np.exp(model_rfr.predict(Xtrain))
ypred_test_rfr = np.exp(model_rfr.predict(Xtest))
# Evaluation
evaluate(np.exp(ytrain), np.exp(ytest), ypred_train_rfr, ypred_test_rfr)

# In[14]:


# Importance des variables
feat_import = model_rfr.feature_importances_
importance = pd.DataFrame(feat_import, columns=["importance"],
                          index=Xtrain.columns)
sort = importance.sort_values(by="importance", ascending=False)
sort.head(10)

# In[15]:


# Représentation graphique
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(range(1, n_feat + 1), np.cumsum(importance.values))
ax.set_title("Cumsum importance feature with random forest model")
ax.set_xlabel("nb. features")
ax.set_ylabel("feature importance")
ax.grid()
plt.savefig("feat_rfr.png")
plt.show()

# ### Ridge - Lasso régression

# In[16]:


# importation de la classe de calcul
from sklearn.linear_model import LinearRegression, Lasso, Ridge

# Instanciation
model_lr = LinearRegression()
model_rid = Ridge()
model_las = Lasso(alpha=0.00001)

# Entraînement
model_lr.fit(Xtrain, ytrain)
model_rid.fit(Xtrain, ytrain)
model_las.fit(Xtrain, ytrain)

# Prévision train set
ypred_train_lr = np.exp(model_lr.predict(Xtrain))
ypred_train_rid = np.exp(model_rid.predict(Xtrain))
ypred_train_las = np.exp(model_las.predict(Xtrain))

# Prévision test set
ypred_test_lr = np.exp(model_lr.predict(Xtest))
ypred_test_rid = np.exp(model_rid.predict(Xtest))
ypred_test_las = np.exp(model_las.predict(Xtest))

# In[17]:


# Evaluation modèle de régression
evaluate(np.exp(ytrain), np.exp(ytest), ypred_train_lr, ypred_test_lr)

# In[18]:


# Evaluation modèle de régression ridge
evaluate(np.exp(ytrain), np.exp(ytest), ypred_train_rid, ypred_test_rid)

# In[19]:


# Evaluation modèle de régression Lasso
evaluate(np.exp(ytrain), np.exp(ytest), ypred_train_las, ypred_test_las)

# In[20]:


# Sélection des variables
from sklearn.feature_selection import RFECV

rfe_select = RFECV(estimator=Ridge(), cv=5)
rfe_select.fit(Xtrain, ytrain)
# Nombre de variables sélectionnées
print(rfe_select.n_features_)

