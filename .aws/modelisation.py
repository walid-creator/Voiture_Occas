from sklearn.cross_validation import train_test_split
from lastVersion.py import df
X=df.drop(["prix_vente"],axis=1,inplace=True)
y=df["prix_vente"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=11)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
digit_knn=knn.fit(X_train, y_train)
# Estimation de l’erreur de prévision sur l’échantillon test
print(1-digit_knn.score(X_test,y_test))
# # Coefficients
print(titan_logit.coef_)
from sklearn.grid_search import GridSearchCV
#grille de valeurs
param=[{"n_neighbors":list(range(1,15))}]
knn= GridSearchCV(KNeighborsClassifier(),param,cv=5,n_jobs=-1)
digit_knn=knn.fit(X_train, y_train)
# paramètre optimal
print(digit_knn.best_params_["n_neighbors"])

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

