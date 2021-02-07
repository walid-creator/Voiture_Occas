# -*- coding: utf-8 -*-
"""validation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jWZNsk7lFy6uvxyG4hZKt6UmipnknybX
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from google.colab import drive
drive.mount('/content/drive')
ozone = pd.read_csv('/content/drive/My Drive/ColabNotebooks/data/ozone_long.csv', sep=";", decimal='.')
print(ozone.head())

reg = smf.ols('O3~T6+T12+Ne12+Ne15+Vx+O3v',data=ozone).fit()
reg.summary2()

"""# Valeurs aberrantes

Sur le graphique suivant sont représentés en ordonnée les résidus studentisés pour chaque individus. Les lignes en pointillés vertes représentent le seuil d'une lois de Student à un niveau de confiance de 95%.
"""

n = len(ozone)

from statsmodels.nonparametric.smoothers_lowess import lowess # pour le lissage
from statsmodels.stats.outliers_influence import OLSInfluence # pour récupérer les résidus studentisés
influ = OLSInfluence(reg)
x = np.linspace(1,n,n)
y = np.repeat(2,n)
fig, ax = plt.subplots()
ax.scatter(x,influ.resid_studentized_external)
ax.plot(x,y,linestyle='dashed',color='green')
ax.plot(x,-y,linestyle='dashed',color='green')
filtered = lowess(influ.resid_studentized_external,x)
ax.plot(filtered[:,0],filtered[:,1],color='black')
ax.set_xlabel("Individu")
ax.set_ylabel("Résidu studentisé par VC")
ax.set_title("Représentation des résidus en fonction des individus") #p.54

"""Il est normal que certanes valeurs dépassent ce seuil ponctuellement, en fait, en moyenne 5% des individus sont à l'extérieur de la bande matérialisée par les lignes en pointillés vertes.

La courbe en noire est un lissage des valeurs des résidus studentisés. On observe une légère courbure autour du jour 600 mais rien de dramatique.

Les observations suspectes correspondent au jour 432, 618 et 671 et il faut regarder plus précisément ces observations pour comprendre pourquoi ces points sont aberrants. Une valeur aberrante n'est pas nécessairement une observation que l'on doit écarter de l'échantillon. C'est surtout le cas s'il s'agit d'un point levier (voir plus bas).
"""

ozone.iloc[np.where(np.abs(influ.resid_studentized_external) > 3)]

"""# Analyse de la normalité

Ci-dessous est représenté le graphe quantile-quantile des quantiles empriques en ordonné par rapport aux quantiles théorique d'une loi $\mathcal N(0,1)$. Le choix de la version studentisé des résidus est nécessaire pour que chaque valeur soit normalisée et que la comparaison avec les quantiles empiriques fassent sens.
"""

fig = sm.qqplot(influ.resid_studentized_external,line='45') #p.55

"""On retrouve ici les 3 observations aberrantes de la section précédente : ce sont les 3 points les plus éloignés de la première bisectrice. Cependant, l'hypothèse de normalité reste très raisonnable.

# Homoscédasticité

À gauche sont représentés les résidus studentisés contre les valeurs ajustées de la variable réponse. On retrouve ces 3 observations un peu en dehors de la norme. Cepedant, on observe pas de structure particulière.

On observe un phénomène très simillaire sur le graphique de droite représentant la valeur absolue des résidus studentisés en fonction de la valeur ajustée de la variable réponse. Le lissage permet de mieux objectiver le fait qu'il n'y ait pas de structure.
"""

fig, (ax1,ax2) = plt.subplots(1,2)
fig.subplots_adjust(wspace=0.5)
filtered = lowess(abs(influ.resid_studentized_external),reg.fittedvalues)
ax1.scatter(reg.fittedvalues,influ.resid_studentized_external)
ax2.scatter(reg.fittedvalues,np.abs(influ.resid_studentized_external)) #p.55
ax2.plot(filtered[:,0],filtered[:,1],color = 'black')

ax1.set_xlabel("Valeur ajustée Y")
ax1.set_ylabel("Résidu studentisé par VC")
ax1.set_title("Résidus studentisés \n versus valeurs ajustées")

ax2.set_xlabel("Valeur ajustée Y")
ax2.set_ylabel("Résidus studentisés en valeur abs.")
ax2.set_title("Résidus studentisés versus \n valeurs ajustées en valeur abs.")

"""# Analyse de la structure des résidus

Il a déjà été procédé à une première analyse de la structure des résidus dans la section précédente. On peut affiner cette analyse en représentant le graphique des résidus studentisés en fonction de chaque variables explicatives.
"""

# analyse de la structure des résidus (residus versus variables explicatives)
fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,figsize=(6,8))
fig.subplots_adjust(wspace=1,hspace=1)
# residus vs T6
filtered_T6 = lowess(influ.resid_studentized_external,ozone['T6'])
ax1.scatter(ozone['T6'],influ.resid_studentized_external)
ax1.plot(filtered_T6[:,0],filtered_T6[:,1],color = 'black')
ax1.set_xlabel("T6")
ax1.set_ylabel("Résidus stud.")
ax1.set_title("Résidus versus T6")
# residus vs T12
filtered_T12 = lowess(influ.resid_studentized_external,ozone['T12'])
ax2.scatter(ozone['T12'],influ.resid_studentized_external)
ax2.plot(filtered_T12[:,0],filtered_T12[:,1],color = 'black')
ax2.set_xlabel("T12")
ax2.set_ylabel("Résidus stud.")
ax2.set_title("Résidus versus T12")
# residus versus Ne12
filtered_Ne12 = lowess(influ.resid_studentized_external,ozone['Ne12'])
ax3.scatter(ozone['Ne12'],influ.resid_studentized_external)
ax3.plot(filtered_Ne12[:,0],filtered_Ne12[:,1],color = 'black')
ax3.set_xlabel("Ne12")
ax3.set_ylabel("Résidus stud.")
ax3.set_title("Résidus versus Ne12")
# residus versus Ne15
filtered_Ne15 = lowess(influ.resid_studentized_external,ozone['Ne15'])
ax4.scatter(ozone['Ne15'],influ.resid_studentized_external)
ax4.plot(filtered_Ne15[:,0],filtered_Ne15[:,1],color = 'black')
ax4.set_xlabel("Ne15")
ax4.set_ylabel("Résidus stud.")
ax4.set_title("Résidus versus Ne15")
# residus versus Vx
filtered_Vx = lowess(influ.resid_studentized_external,ozone['Vx'])
ax5.scatter(ozone['Vx'],influ.resid_studentized_external)
ax5.plot(filtered_Vx[:,0],filtered_Vx[:,1],color = 'black')
ax5.set_xlabel("Vx")
ax5.set_ylabel("Résidus stud.")
ax5.set_title("Résidus versus Vx")
# residus versus O3v
filtered_O3v = lowess(influ.resid_studentized_external,ozone['O3v'])
ax6.scatter(ozone['O3v'],influ.resid_studentized_external)
ax6.plot(filtered_O3v[:,0],filtered_O3v[:,1],color = 'black')
ax6.set_xlabel("O3v")
ax6.set_ylabel("Résidus stud.")
ax6.set_title("Résidus versus O3v")

"""Les graphiques pour les variables ${\rm T6}, {\rm Ne12}, {\rm Ne15}$ n'exhibe pas de structure particulière si bien que l'on peut supposer que l'hypothèse linéaire en ces variables est raisonnable ainsi que l'hypothèse d'homoscédasticité.

En revanche pour les variables ${\rm Vx}, {\rm O3v}$ et plus particulièrement ${\rm T12}$ on observe un changement de régime matérialisé par une brisure de la courbe de lissage. Cela peut être dû à une hétéroscédasticité inhérente au phénomène physique ou à une erreur de modèle. Typiquement, on pourrait tenter de décomposer ${\rm T12}$ en deux variables qui modéliserait le lien avec la température à midi dans chacun des régime inférieure ou supérieure à 20°C. Ceci peut-être fait à l'aide d'indicatrice.

# Distance de Cook, DFFITS et points leviers
"""

yok = np.repeat(0.1,n)
cd, pval = influ.cooks_distance
fig, ax = plt.subplots()
ax.scatter(x,cd)
ax.plot(x,yok,linestyle='dashed',color='green')
ax.set_xlabel("Individu")
ax.set_ylabel("distance de Cooks")
ax.set_title("Distance de Cooks") #p.54

obs, dffits = influ.dffits
print(dffits)
yok = np.repeat(dffits,n)
yok_r = np.repeat(1.5*dffits,n)
fig, ax = plt.subplots()
ax.scatter(x,abs(obs))
ax.plot(x,yok,linestyle='dashed',color='green')
ax.plot(x,yok_r,linestyle='dashed',color='green')
ax.set_xlabel("Individu")
ax.set_ylabel("Écart de Welsh-Kuh")
ax.set_title("Mesure DFFITS") #p.54

hii = influ.hat_diag_factor
yhw = np.repeat(2*7/n,n) # Hoaglin-Welsh
yvw = np.repeat(3*7/n,n) # Velleman-Welsh
yh = np.repeat(0.5,n) # Huber
fig, ax = plt.subplots()
ax.scatter(x,hii)
ax.plot(x,yhw,linestyle='dashed',color='green',label='Hoaglin-Welsh')
ax.plot(x,yvw,linestyle='dashed',color='blue',label='Velleman-Welsh')
#ax.plot(x,yh,linestyle='dashed',color='black',label='Huber')
ax.set_xlabel("Individu")
ax.set_ylabel("hii")
ax.set_title("Points leviers") #p.54
ax.legend()

"""# Sélection de variables

Contrairement à `R`, la librairie `statsmodel` n'a pas de procédure toute faite pour la sélection de variables. On définit donc une fonction pour l'implémenter. Voici un code pour la procédure `forward` (en exercice, vous pouvez tenter d'écrire la procédure `backward`) basée sur le $R^2$ ajusté (équivalent à la minimisation de la variance).

En exercice, vous pouvez écrire une fonction équivalente pour les $C_p$ de Mallow, l'AIC et le BIC.
"""
"""
import statsmodels.formula.api as smf

def forward_selected(data, response, verbose = True):
    #Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            if verbose:
                print('Ajout de {:30} le R2 ajusté courrant est {:.6}'.format(best_candidate, best_new_score))
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model
"""
"""Un exemple d'application sur les données d'ozones.

model = forward_selected(ozone, 'O3')
print(model.model.formula)
"""