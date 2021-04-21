from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("/Users/famille//projetstat/.aws/df_modelisation.csv",error_bad_lines=False,index_col=0)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#suppression des prix>10^5
indexNames3 = df[ df["prix_vente"]  >10**5].index
#indexNames4 = df[ df["prix_vente"]  <5*10**3].index
df.drop(indexNames3, inplace=True)
#df.drop(indexNames4, inplace=True)
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

#avec l'algo forward on sélectionnes les varaibles suivantes: 10

X=X[['energie_Essence', 'boite_de_vitesse_meca', 'couleur_blanc',
       'premiere_main_non', 'Finition_Societe', 'porte_3.0', 'engine',
       'kilometrage', 'horsepower', 'axe 1']]



#spliter les données
import numpy as np
X_train,X_test,y_train,y_test=train_test_split(X,np.log(y),test_size=0.3,random_state=11)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train=y_train.tolist()
y_test=y_test.tolist()


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor()


#choix de k: nb du plus proche voisin par deux méthodes
#méthode1:

'''
param_grid={'n_neighbors': np.arange(1,50),'metric': ['manhattan','euclidean'],'weights':['distance','uniform']}
from sklearn.model_selection import GridSearchCV
knn = GridSearchCV(knn, param_grid, cv=5)
knn.fit(X_train,y_train)
print(knn.best_params_)
'''



#entrainer le modèle
import math
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(metric= 'euclidean', n_neighbors= 9,weights='distance')# les prix ne sont pas uniformement distribués
knn.fit(X_train,y_train)

#prédiction
pred=knn.predict(X_test)
pred_train=knn.predict(X_train)
#erreure et score
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
error = math.sqrt(mean_squared_error(np.exp(y_test),np.exp(pred))) #calculate rmse
error_train= math.sqrt(mean_squared_error(np.exp(y_train),np.exp(pred_train)))
score= r2_score(np.exp(y_test),np.exp(pred))
score_train= r2_score(np.exp(y_train),np.exp(pred_train))
mae=mean_absolute_error(np.exp(y_test),np.exp(pred))
mape=mean_absolute_percentage_error(np.exp(y_test),np.exp(pred))
mae_train= mean_absolute_error(np.exp(y_train),np.exp(pred_train))
mape_train=mean_absolute_percentage_error(np.exp(y_train),np.exp(pred_train))
def get_sd_error(Y_True,Y_Pred):
    n = len(Y_Pred)
    somme = 0
    m_bar = mean_absolute_percentage_error(Y_True, Y_Pred)
    for i in range(n):
        somme += ((np.abs(Y_True[i] - Y_Pred[i])/Y_True[i]) - m_bar)**2
    return(np.sqrt(somme/n))



ecart_type=get_sd_error(np.exp(y_test), np.exp(pred))
coef_var=get_sd_error(np.exp(y_test), np.exp(pred))/mape
ecart_type_train=get_sd_error(np.exp(y_train), np.exp(pred_train))
coef_var_train=get_sd_error(np.exp(y_train), np.exp(pred_train))/mape_train



#erreure quadratique moyenne
print('RMSE value for k= ', 9 , 'is:', error)
print('RMSE_train value for k= ', 9 , 'is:', error_train)
#R^2
print('Score value for k= ', 9 , 'is:', score)
print('Score_train value for k= ', 9 , 'is:', score_train)
#erreure en valeur absolue
print('MAE value for k= ', 9 , 'is:', mae)
print('MAE value_train for k= ', 9 , 'is:', mae_train)
print('Ecart type for k= ', 9 , 'is:', ecart_type)
print('Ecart_train type for k= ', 9 , 'is:', ecart_type_train)
print('Coeff de variation for k= ', 9 , 'is:', coef_var)
print('Coeff_train de variation for k= ', 9 , 'is:', coef_var_train)
print('MAPE for k= ', 9 , 'is:', mape)
print('MAPE_train for k= ', 9 , 'is:', mape_train)
#to sum up:
#RMSE value for k=  9 is: 2031.3787234678903
#Score value for k=  9 is: 0.7236276765367943
#Mape value for k=  9 is: 0.16304824258851666
#Ecart type for k=  9 is: 0.3187826817234875
#Coeff de variation for k=  9 is: 1.9551433162514753



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
'''
sfw = SequentialFeatureSelector(knn,direction='forward',n_features_to_select=10)
sfw.fit(X_train,y_train)
print(sfw.get_support())
print(sfw.transform(X).shape)
pos=[]
for i in range(45):
    if sfw.get_support()[i]==True:
        pos.append(i)
colnamef = X.columns[pos]
print(colnamef)
'''
#RMSE value for k=  9 is: 0.15784476672070805
#Score value for k=  9 is: 0.8557419031104687
#MAPE value for k=  9 is: 0.11021248043273771
kselct=10
k=X.shape[1]
R2_ajsuté=1 - (1-score)*(n-1)/(n-k-1)#0.7276518660749811
R2_ajusté_forw= 1 - (1-score)*(n-1)/(n-kselct-1)#0.8580747705147141

R2_ajsuté_train_forw= 1 - (1-score_train)*(n-1)/(n-kselct-1)
print('R2_ajsuté value for k= ', 9 , 'is:', R2_ajsuté)
print('R2_ajusté_forw value for k= ', 9 , 'is:', R2_ajusté_forw)


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

from sklearn.feature_selection import SelectFromModel
print(X.shape)
model = SelectFromModel(knn, prefit=True)
X_new = model.transform(X)
print(X_new.shape)
'''
RMSE value for k=  9 is: 1398.8461658165036
RMSE_train value for k=  9 is: 511.3966118923854
Score value for k=  9 is: 0.8698820273471887
Score_train value for k=  9 is: 0.9823598230746853
MAE value for k=  9 is: 1029.7700292408615
MAE value_train for k=  9 is: 178.95106997203484
Ecart type for k=  9 is: 0.20597822984701722
Ecart_train type for k=  9 is: 0.04000511675039793
Coeff de variation for k=  9 is: 1.815367938983363
Coeff_train de variation for k=  9 is: 2.795889507103137
MAPE for k=  9 is: 0.11346362653202333
MAPE_train for k=  9 is: 0.014308547118461712
R2_ajsuté value for k=  5 is: 0.869869633742905
R2_ajusté_forw value for k=  5 is: 0.869869633742905'''


#représenatation graphique pour la comparaison des modèles
import matplotlib.pyplot as plt
import seaborn as sns
axes = plt.gca()
x = ['KNN','RF','ArbreD','RL']
y0 = [1,2,3,4]
y1= [1.2,3.1,2.2,1]
y2= [2,1,2,1]
y3= [1,3,2,1]
a1, =plt.plot(x,y0,color="yellow",label=" r2")

a2, =plt.plot(x,y1,color="red")

a3, =plt.plot(x,y2,color="blue")

a4, =plt.plot(x,y3,color="black")
plt.legend([a1, a2, a3, a4,], ['r2','MAPE','CV','MAE'])
plt.title("Evaluation par différents critères de chaque modèle")
axes.set_xlabel('axe des x')
axes.set_ylabel('axe des y')
#plt.legend(y,y1)
plt.show()

#distribution des prix:
import numpy as np
import seaborn as sns
#densité
sns.distplot(y, kde=True)
plt.show()


