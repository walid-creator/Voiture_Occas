from sklearn.model_selection import train_test_split
from projet import df
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
X=df.drop(["prix_vente"],axis=1)
y=df["prix_vente"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
'''plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, alpha=0.8)
plt.title('Train set')
plt.subplot(122)
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1],c=y_test, alpha=0.8)
plt.title('Test set')
plt.show()'''
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
k = np.arange(1, 50)

train_score, val_score = validation_curve(model, X_train, y_train,
                                          'n_neighbors', k, cv=5)

plt.plot(k, val_score.mean(axis=1), label='validation')
plt.plot(k, train_score.mean(axis=1), label='train')

plt.ylabel('score')
plt.xlabel('n_neighbors')
plt.legend()