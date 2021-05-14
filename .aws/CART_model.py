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
from sklearn.metrics import r2_score, make_scorer
from sklearn.feature_selection import SequentialFeatureSelector

import numpy as np
import statsmodels.api as sm
import pylab as py

# Chargement de la base de données:
df = pd.read_csv(".aws/df_modelisation.csv", error_bad_lines=False, index_col=0)

# On retire les 6 outliers
for i in range(len(df)):
    if df["prix_vente"][i] > 100000:
        df = df.drop(df.index[i])
        df.reset_index(drop=True, inplace=True)

# On réduit et on centre les variables explicatives numériques.
def Normalisation(df):
    scaler = StandardScaler()
    df[["kilometrage", "engine", "puissance_fiscale", "age"]] = scaler.fit_transform(df[["kilometrage", "engine", "puissance_fiscale", "age"]])
    return(df)
df = Normalisation(df)


#Création de variables indicatrices pour les variables catégorielles.

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

# Création de deux data frames : l'un possédant les prix des véhicules, l'autre comportant l'ensemble des régresseurs.
Prix_vente = df["prix_vente"]
Feature = df.drop(["prix_vente"], axis=1)
Feature = Feature.drop(["options", "porte","energie", "modele_com", "date_mec", "date_publication", "date_depublication", "boite_de_vitesse", "couleur", "puissance_fiscale", "premiere_main", "departement", "Finition"], axis=1)

# Création de l'échantillon d'apprentissage et de l'échantillon de validation:
X_train, X_test, Y_train, Y_test = train_test_split(Feature, np.log(Prix_vente), test_size=0.3, random_state = 1)

# Création de l'arbre maximal sur l'echantillon d'apprentissage en cherchant à minimiser la variance (critere mse) :
regressor = DecisionTreeRegressor(random_state=0, criterion="mse")
regressor.fit(X_train, Y_train)

#Visualisation de l'arbre :
tree_rules = export_text(regressor, feature_names=list(X_train.columns))
print(regressor.feature_importances_)

#Fonction calculant l'écart type du MAPE

def get_sd_error(Y_True,Y_Pred):
    n = len(Y_Pred)
    somme = 0
    m_bar = mean_absolute_percentage_error(Y_True, Y_Pred)
    for i in range(n):
        abs_diff = np.abs((Y_True[i] - Y_Pred[i])/Y_True[i])
        somme = somme + (abs_diff - m_bar)**2
    res = np.sqrt(somme/n)
    return(res)

#Fonction renvoyant tous les indicateurs (MAE, MAPE, RMSE, R2) pour chaque modèle.
def calculEstimateur(X_train, Y_train, X_test, Y_test, regressor):
    Y_predtrain = np.exp(regressor.predict(X_train))

    Y_predtest_exp = np.exp(regressor.predict(X_test))

    score_train_set = regressor.score(X_train, Y_train)
    score_test_set = regressor.score(X_test,Y_test)

    MAPE_train = mean_absolute_percentage_error(np.exp(Y_train), Y_predtrain)
    MAPE_test = mean_absolute_percentage_error(np.exp(Y_test), Y_predtest_exp)

    MAE_train = mean_absolute_error(np.exp(Y_train), Y_predtrain)
    MAE_test = mean_absolute_error(np.exp(Y_test), Y_predtest_exp)

    RMSE_train = mean_squared_error(np.exp(Y_train), Y_predtrain, squared= False)
    RMSE_test = mean_squared_error(np.exp(Y_test), Y_predtest_exp, squared= False)

    Y_train.reset_index(drop=True, inplace=True)
    Y_test.reset_index(drop=True, inplace=True)

    sd_error_train = get_sd_error(np.exp(Y_train), Y_predtrain)
    sd_error_test = get_sd_error(np.exp(Y_test), Y_predtest_exp)

    coeff_corr_train = sd_error_train/MAPE_train
    coeff_corr_test = sd_error_test/MAPE_test

    return(print(f"Modèle : {regressor} \n Le score de l'échantillon d'apprentissage est : {score_train_set}. \n Le score de l'échantillon test est : {score_test_set}. \n"
                 f"Le MAPE de l'échantillon d'apprentissage est : {MAPE_train}. \n Le MAPE de l'échantillon de test est : {MAPE_test}. \n"
                 f"Le RMSE de l'échantillon d'apprentissage est : {RMSE_train}. \n Le RMSE de l'échantillon de test est : {RMSE_test}. \n"
                 f"Le coefficient de variation de l'échantillon d'apprentissage est : {coeff_corr_train}. \n Le coefficient de variation de l'échantillon de test est : {coeff_corr_test}. \n"
                 f"Le MAE de l'échantillon d'apprentissage est : {MAE_train}. \n Le MAE de l'échantillon de test est : {MAE_test}."))

calculEstimateur(X_train,Y_train,X_test,Y_test, regressor)

# Vérification des résultats fournis par scikit:

def MAPE(Y_True, Y_Pred):
    n = len(Y_True)
    somme = 0
    for i in range(n):
        somme += np.abs((Y_True[i] - Y_Pred[i])/Y_True[i])
    return(somme/n)
def MAE(Y_True, Y_Pred):
    n = len(Y_True)
    somme = 0
    for i in range(n):
        somme += np.abs(Y_True[i] - Y_Pred[i])
    return(somme/n)
def RMSE(Y_True, Y_Pred):
    n = len(Y_True)
    somme = 0
    for i in range(n):
        somme += (Y_True[i] - Y_Pred[i])**2
    return(np.sqrt(somme/n))

#np.std(np.abs(Y_true - Y_pred_exp))

# ==================================================================================================================#

# Elagage de l'arbre maximal : visualisation de coût complexité.
clf = regressor
path = clf.cost_complexity_pruning_path(X_train, Y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("Valeur de alpha")
ax.set_ylabel("Impureté totale des feuilles")
ax.set_title("Représentation graphique de l'impureté totale \n en fonction de la valeur alpha de coût-complexité minimal")

# Recherche du alpha optimal:

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeRegressor(random_state=0, criterion= "mse", ccp_alpha = ccp_alpha, min_samples_leaf= 0.001)
    clf.fit(X_train, Y_train)
    clfs.append(clf)

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker='s', drawstyle="steps-post")
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

#def find_max_alpha():
#    n = len(ccp_alphas)
#    max = 0
#    index = 0
#    for i in range(n):
#        if test_scores[i] > max:
#            index = i
#            max = test_scores[i]
#    return(index, ccp_alphas[index])
#print(find_max_alpha())

# ==================================================================================================================#
# Recherche du alpha optimal sans limite par feuille:

ccp_to_test = [0, 0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009, 0.0001, 0.00011, 0.00012, 0.00013, 0.00014, 0.00015, 0.00016, 0.00017,
               0.00018, 0.00019, 0.0002, 0.00025, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0015, 0.002, 0.003]

clfs = []
for ccp_alpha in ccp_to_test:
    clf = DecisionTreeRegressor(random_state=0, criterion= "mse",ccp_alpha = ccp_alpha)
    clf.fit(X_train, Y_train)
    clfs.append(clf)

clfs = clfs[:-1]
ccp_to_test = ccp_to_test[:-1]

train_scores = [clf.score(X_train, Y_train) for clf in clfs]
test_scores = [clf.score(X_test, Y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("Alpha")
ax.set_ylabel("Valeur du score")
ax.set_title("Représentation graphique du score \n en fonction de alpha")
ax.plot(ccp_to_test, train_scores, marker='o', label="echantillon d'apprentissage",
        drawstyle="steps-post")
ax.plot(ccp_to_test, test_scores, marker='s', label="echantillon test",
        drawstyle="steps-post")
ax.legend()
plt.show()

# ==================================================================================================================#
# Recréation de l'arbre avec un alpha = 0.0004, et calcul des indicateurs :

regressor_with_ccp = DecisionTreeRegressor(random_state=0, criterion="mse", ccp_alpha = 0.00004)
regressor_with_ccp.fit(X_train, Y_train)

calculEstimateur(X_train, Y_train, X_test, Y_test, regressor_with_ccp)

# ==================================================================================================================#
# Tentative avec d'autres paramètres :
regressor_with_min_samples = DecisionTreeRegressor(random_state=0, criterion="mse",  min_samples_leaf = 0.001) #On limite la croissance de l'arbre en fixant un nbr minimal d'observations par noeud terminal
regressor_with_min_samples.fit(X_train, Y_train)

calculEstimateur(X_train, Y_train, X_test, Y_test, regressor_with_min_samples)

regressor_with_min_samples_pruned = DecisionTreeRegressor(random_state=0, criterion= "mse", min_samples_leaf= 0.001, ccp_alpha = 0.00001) # On élague l'arbre précédent.
regressor_with_min_samples_pruned.fit(X_train,Y_train)

calculEstimateur(X_train, Y_train, X_test, Y_test, regressor_with_min_samples_pruned) # Calcul des indicateurs.
# ==================================================================================================================#
regressor_with_median = DecisionTreeRegressor(random_state=0, criterion= "mae") # On utilise le MAE au lieu de la variance comme critère d'impureté.
regressor_with_median.fit(X_train,Y_train)

calculEstimateur(X_train, Y_train, X_test, Y_test, regressor_with_median) # Calcul des indicateurs.

#Elagage de l'arbre précédent :
clf = regressor_with_median
path = clf.cost_complexity_pruning_path(X_train, Y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("Valeur de alpha")
ax.set_ylabel("Impureté totale des feuilles")
ax.set_title("Représentation graphique de l'impureté totale en fonction de la valeur alpha de coût-complexité minimal")


ccp_to_test = [0, 0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009, 0.0001, 0.00011, 0.00012, 0.00013, 0.00014, 0.00015, 0.00016, 0.00017,
               0.00018, 0.00019, 0.0002, 0.00025, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0015, 0.002, 0.003]

clfs = []
for ccp_alpha in ccp_to_test:
    clf = DecisionTreeRegressor(random_state=0, criterion= "mae",ccp_alpha = ccp_alpha)
    clf.fit(X_train, Y_train)
    clfs.append(clf)

clfs = clfs[:-1]
ccp_to_test = ccp_to_test[:-1]

train_scores = [clf.score(X_train, Y_train) for clf in clfs]
test_scores = [clf.score(X_test, Y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("Coefficient de détermination")
ax.set_title("Représentation graphique du coefficient de détermination en fonction de alpha")
ax.plot(ccp_to_test, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_to_test, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()

#Recréation de l'arbre avec le paramère alpha optimal :
regressor_with_median_pruned = DecisionTreeRegressor(random_state=0, criterion= "mae", ccp_alpha = 0.00003)
regressor_with_median_pruned.fit(X_train,Y_train)

#Calcul des indicateurs pour l'arbre construit avec le MAE comme critère d'impureté
calculEstimateur(X_train, Y_train, X_test, Y_test, regressor_with_median_pruned)

# ==================================================================================================================#

# Représentation des densités estimées et des densités réelles.

import numpy as np
import seaborn as sns


Y_predtest_exp = np.exp(regressor_with_ccp.predict(X_test))
Y_true = np.exp(Y_test)

ax = sns.distplot(Y_predtest_exp, hist=False, kde=True, label='densité estimée')
[line.set_linestyle("--") for line in ax.lines]
bx = sns.distplot(Y_true, hist=False, kde=True, label='densité des vraies valeurs')
axes = plt.gca()
axes.set_xlabel('prix de vente des véhicules en EURO')
axes.set_ylabel('densité estimée')
axes.set_xlim(0, 40000)
plt.title("Comparaisons des densités des vraies données et des densités estimées")
plt.legend()
axes.set_facecolor('#E0E0E0')
plt.show()

