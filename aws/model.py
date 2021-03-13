from sklearn.model_selection import train_test_split
import pandas as pd
df = pd.read_csv("/Users/famille//projetstat/.aws/df_modelisation.csv",error_bad_lines=False,index_col=0)
#df=df.head(1000)
y=df["prix_vente"]
#correlation
print(df[[u"kilometrage", u"prix_vente"]].corr())#il est bien negative mais faible -0.051
print(df[[u"puissance_fiscale", u"prix_vente"]].corr())#0.022
print(df[[u"horsepower", u"prix_vente"]].corr())#0.035
print(df[[u"age", u"prix_vente"]].corr())# -0.04
print(df[[u"prix_vente", u"engine"]].corr())# -0.01
print(df[[u"age", u"kilometrage",u"puissance_fiscale",u"horsepower","engine"]].corr())#forte corrélation:age&kilometrage=0.76 et puissance_fiscale&horsepower=0.81

#suppression de var corr
df.drop(["prix_vente","modele_com","date_publication","date_depublication","date_mec","kilometrage","puissance_fiscale"],axis=1,inplace=True)

# Table des indicatrices
df["Finition"] = df["Finition"].replace("Alize", "autre")
df["Finition"] = df["Finition"].replace("Authentique", "autre")
df["Finition"] = df["Finition"].replace("Billabong", "autre")
df["Finition"] = df["Finition"].replace("Campus", "autre")
df["Finition"] = df["Finition"].replace("Dynamique", "autre")
df["Finition"] = df["Finition"].replace("Exception", "autre")
df["Finition"] = df["Finition"].replace("Extreme", "autre")
df["Finition"] = df["Finition"].replace("GT", "autre")
df["Finition"] = df["Finition"].replace("Generation", "autre")
df["Finition"] = df["Finition"].replace("Initiale", "autre")
df["Finition"] = df["Finition"].replace("Life", "autre")
df["Finition"] = df["Finition"].replace("Luxe", "autre")
df["Finition"] = df["Finition"].replace("One", "autre")
df["Finition"] = df["Finition"].replace("Privilege", "autre")
df["Finition"] = df["Finition"].replace("RTA", "autre")
df["Finition"] = df["Finition"].replace("RTE", "autre")
df["Finition"] = df["Finition"].replace("RXE", "autre")
df["Finition"] = df["Finition"].replace("RXT", "autre")
df["Finition"] = df["Finition"].replace("SI", "autre")
df["Finition"] = df["Finition"].replace("Sport", "autre")
df["Finition"] = df["Finition"].replace("Tomtom", "autre")
df["Finition"] = df["Finition"].replace("Trend", "autre")
df1=pd.get_dummies(df[["Finition","couleur","boite_de_vitesse","energie"]])



'''
# Variables quantitatives
df2=df[["age","horsepower"]]

# Concaténation
X=pd.concat([df1,df2],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=11)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=11)
digit_knn=knn.fit(X_train, y_train)

# Estimation de l’erreur de prévision sur l’échantillon test
print(1-digit_knn.score(X_test,y_test))
'''
from sklearn.model_selection import GridSearchCV
#grille de valeurs
'''
param=[{"n_neighbors":list(range(1,15))}]
knn= GridSearchCV(KNeighborsClassifier(),param,cv=5,n_jobs=-1)
digit_knn=knn.fit(X_train, y_train)
'''
# paramètre optimal
#print(digit_knn.best_params_["n_neighbors"])
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