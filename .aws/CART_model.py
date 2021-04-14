# Importation des packages.

from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import export_text
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingRegressor

# Chargement de la base de données:
df = pd.read_csv(".aws/df_modelisation.csv", error_bad_lines=False, index_col=0)

# On retire les outliers
for i in range(len(df)):
    if df["prix_vente"][i] > 100000:
        df = df.drop(df.index[i])
        df.reset_index(drop=True, inplace=True)

scaler = StandardScaler()
df[["kilometrage", "engine", "puissance_fiscale", "age"]] = scaler.fit_transform(df[["kilometrage", "engine", "puissance_fiscale", "age"]])

# Récupération du prix et des régresseurs pour construire l'arbre (pour l'instant uniquement regresseur quantitatifs car je n'ai pas réussi à ajouter des régresseurs qualitatifs).
#couleur a ajouter et remplacer horsepower
dummies = pd.get_dummies(df['energie']).rename(columns=lambda x: 'Category_' + str(x))
df = pd.concat([df, dummies], axis=1)
dummies = pd.get_dummies(df['boite_de_vitesse']).rename(columns=lambda x: 'Category_' + str(x))
df = pd.concat([df, dummies], axis=1)
dummies = pd.get_dummies(df['premiere_main']).rename(columns=lambda x: 'Category_' + str(x))
df = pd.concat([df, dummies], axis=1)
dummies = pd.get_dummies(df['Finition']).rename(columns=lambda x: 'Category_' + str(x))
df = pd.concat([df, dummies], axis=1)
dummies = pd.get_dummies(df['couleur']).rename(columns=lambda x: 'Category_' + str(x))
df = pd.concat([df, dummies], axis=1)
dummies = pd.get_dummies(df['porte']).rename(columns=lambda x: 'Category_' + str(x))
df = pd.concat([df, dummies], axis=1)

Prix_vente = df["prix_vente"]
Feature = df.drop(["prix_vente"], axis=1)
Feature = Feature.drop(["options", "porte", "modele_com", "date_mec", "date_publication", "date_depublication", "energie", "boite_de_vitesse", "couleur", "horsepower", "premiere_main", "departement", "Finition", "age"], axis=1)

Feature_bis = df[["kilometrage", "engine", "puissance_fiscale"]]
# Création de l'échantillon d'apprentissage et de l'échantillon de validation:

X_train, X_test, Y_train, Y_test = train_test_split(Feature_bis, np.log(Prix_vente), test_size=0.3, random_state = 0)

# Création de l'arbre maximal sur l'echantillon d'apprentissage en cherchant à minimiser la variance (critere mse) :
regressor = DecisionTreeRegressor(random_state=0, criterion="mse", min_samples_leaf = 0.001)
regressor.fit(X_train, Y_train)

#Visualisation de l'arbre :
tree_rules = export_text(regressor, feature_names=list(X_train.columns))
print(regressor.feature_importances_)

# Prédiction du prix sur l'échantillon de validation.
Y_pred = regressor.predict(X_test)
Y_pred_exp = np.exp(Y_pred)

score_1 = regressor.score(X_test, Y_test) # Les régresseurs expliquent 81 % de l'information présente dans le modèle.
print(score_1)

#MAPE % puis coeff de variation il faut < 20% sinon trop grand ecart entre obs et moyenne donc MAPE biaisé ==> moyenne pas significative/ RMSE
# tracer courbe entre y reel et y predit.

# Visualisation de la différence entre les vrais prix et ceux prédit par l'arbre de régression:

Y_true = np.exp(Y_test)
Y_true.reset_index(drop = True, inplace = True)

print(mean_absolute_percentage_error(Y_true, Y_pred_exp))
def MAPE(Y_True, Y_Pred):
    n = len(Y_true)
    somme = 0
    for i in range(n):
        somme += np.abs((Y_True[i] - Y_Pred[i])/Y_True[i])
    return(somme/n)
print(MAPE(Y_true,Y_pred_exp)) # MAPE égal à 0.12

print(mean_absolute_error(Y_true, Y_pred_exp))
def MAE(Y_True, Y_Pred):
    n = len(Y_true)
    somme = 0
    for i in range(n):
        somme += np.abs(Y_True[i] - Y_Pred[i])
    return(somme/n)
print(MAE(Y_true,Y_pred_exp)) # MAE égale a 1119

print(mean_squared_error(Y_true, Y_pred_exp, squared=False))
def RMSE(Y_True, Y_Pred):
    n = len(Y_true)
    somme = 0
    for i in range(n):
        somme += (Y_True[i] - Y_Pred[i])**2
    return(np.sqrt(somme/n))
print(RMSE(Y_true,Y_pred_exp)) #RMSE égale à 1570.

def get_sd_error(Y_True,Y_Pred):
    n = len(Y_Pred)
    somme = 0
    m_bar = mean_absolute_percentage_error(Y_true, Y_Pred)
    for i in range(n):
        somme += ((np.abs((Y_True[i] - Y_Pred[i])/Y_True[i])) - m_bar)**2
    return(np.sqrt(somme/n))
print(get_sd_error(Y_true, Y_pred_exp)) # écart type égal à 1100

print(get_sd_error(Y_true, Y_pred_exp)/mean_absolute_percentage_error(Y_true, Y_pred_exp)) # coeff de variation égal à 0.98
# ==================================================================================================================#

# Elagage de l'arbre maximal : visualisation de coût complexité.
clf = regressor
path = clf.cost_complexity_pruning_path(X_train, Y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Représentation graphique de l'impureté totale en fonction de la valeur alpha de coût-complexité minimal")

# Recherche du alpha optimal à la main sans limite par feuille :

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeRegressor(random_state=0, ccp_alpha = ccp_alpha, min_samples_leaf= 0.001)
    clf.fit(X_train, Y_train)
    clfs.append(clf)

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()

train_scores = [clf.score(X_train, Y_train) for clf in clfs]
test_scores = [clf.score(X_test, Y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("précision")
ax.set_title("Précision (X2) en fonction de alpha")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()

def find_max_alpha():
    n = len(ccp_alphas)
    max = 0
    index = 0
    for i in range(n):
        if test_scores[i] > max:
            index = i
            max = test_scores[i]
    return(index, ccp_alphas[index])
print(find_max_alpha())

# ==================================================================================================================#
# Recherche du alpha optimal avec limite par feuille:
print(test_scores[490])
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeRegressor(random_state=0, ccp_alpha = ccp_alpha, min_samples_split = 0.01)
    clf.fit(X_train, Y_train)
    clfs.append(clf)

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()

train_scores = [clf.score(X_train, Y_train) for clf in clfs]
test_scores = [clf.score(X_test, Y_test) for clf in clfs]


fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("précision")
ax.set_title("Précision (X2) en fonction de alpha")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()

# ==================================================================================================================#
# Recréation de l'arbre avec un alpha = 0.0001, et calcul

regressor_with_ccp = DecisionTreeRegressor(random_state=0, criterion="mse", ccp_alpha = 1.4341231800350741e-05, min_samples_leaf= 0.001)
regressor_with_ccp.fit(X_train, Y_train)
Y_pred_with_ccp = regressor_with_ccp.predict(X_test)

score_with_ccp = regressor_with_ccp.score(X_test, Y_test) # Les régresseurs expliquent 77 % de l'information présente dans le modèle.
print(score_with_ccp)

Y_pred_with_ccp_exp = np.exp(Y_pred_with_ccp)

print(mean_absolute_percentage_error(Y_true, Y_pred_with_ccp_exp))
def MAPE(Y_True, Y_Pred):
    n = len(Y_true)
    somme = 0
    for i in range(n):
        somme += np.abs((Y_True[i] - Y_Pred[i])/Y_True[i])
    return(somme/n)
print(MAPE(Y_true,Y_pred_with_ccp_exp)) # MAPE égal à 0.11

print(mean_absolute_error(Y_true, Y_pred_with_ccp_exp))
def MAE(Y_True, Y_Pred):
    n = len(Y_true)
    somme = 0
    for i in range(n):
        somme += np.abs(Y_True[i] - Y_Pred[i])
    return(somme/n)
print(MAE(Y_true,Y_pred_with_ccp_exp)) # MAE égal à 1101


print(mean_squared_error(Y_true, Y_pred_with_ccp_exp, squared=False))
def RMSE(Y_True, Y_Pred):
    n = len(Y_true)
    somme = 0
    for i in range(n):
        somme += (Y_True[i] - Y_Pred[i])**2
    return(np.sqrt(somme/n))
print(RMSE(Y_true,Y_pred_with_ccp_exp)) # RMSE égal à 1548

def get_sd_error(Y_True,Y_Pred):
    n = len(Y_Pred)
    somme = 0
    m_bar = mean_absolute_percentage_error(Y_true, Y_Pred)
    for i in range(n):
        somme += ((np.abs((Y_True[i] - Y_Pred[i])/Y_True[i])) - m_bar)**2
    return(np.sqrt(somme/n))
print(get_sd_error(Y_true, Y_pred_exp)) # écart type égal à 1100

print(get_sd_error(Y_true, Y_pred_exp)/mean_absolute_percentage_error(Y_true, Y_pred_with_ccp_exp)) # coeff de variation égal à 0.98
# ==================================================================================================================#
