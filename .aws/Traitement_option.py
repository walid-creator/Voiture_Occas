import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import unicodedata

### Importation des données

df = pd.read_csv("s3://projet-stat-ensai/Lacentrale.csv",sep =';', dtype=str)
print(df.shape)
print(type(df))
print(df.columns)
print(df.describe())

### Fonction d'uniformisation des mots (on enlève les accents, on met en miniscule...)

def uniform(word) :
    u = unicodedata.normalize('NFKD', word).encode('ASCII', 'ignore')
    u2 = u.decode('ASCII')
    u3 = u2.lower()
    return(u3)


### Fonction renvoyant une liste avec l'ensemble des options présentes dans la variable 'options'
### et une liste de liste contenant les options par voiture

def liste_options() :
    m = df.shape[0]
    options = []
    op_par_voit = []
    for i in range(m) :
        print(i)
        if type(df['options'][i]) == str :
            words = word_tokenize(df['options'][i])     # on tokenize l'information de df['options'][i]
            l = []
            n = len(words)
            i = 0
            while i < n :    # on va parcourir la liste de mot words et on va regrouper les mots en groupe de mots formant les options en considérant que ces groupes de mots sont séparés par des points virgules
                w = ''
                while i < n and words[i] != '``' and words[i] != "''" and words != '"''"' and words[i] != ';' :
                    if len(w) > 0 :
                        w += ' '
                    w += words[i]
                    i += 1
                if w != '' :
                    w = uniform(w)
                    l.append(w)
                i += 1
            options += l
            op_par_voit.append(l)
        else :
            op_par_voit.append([])
    return(options,op_par_voit)

options,op_par_voit = liste_options()


### Calcul des fréquences d'appartition de chaque options

fdist = FreqDist(options)


### Recupération des options présentes dans au moins p% des voitures

def options_rec(p) :
    new_options = []
    for elt in fdist :
        if fdist[elt] > df.shape[0]*(p/100) :
            new_options.append(elt)
    return(new_options)


options2 = options_rec(10)     # on récupère les options présentes dans au moins 10% des véhicules de la base
len(options2)                  # on garde ainsi 96 options différentes


### Ajout de nouvelles variables indicatrices à la table pour chaque options, valant 1 si l'option est présente dans le véhicule et 0 sinon

decompte = 0
for elt in options2 :
    decompte += 1
    print(decompte)
    df[elt] = 52513*[0]
    m = df.shape[0]
    for i in range(m):
        if elt in op_par_voit[i]:
            df[elt][i] = 1


### Création d'une nouvelle variable indicatrice "autre_options" valant 1 si le véhicule considéré a au moins
### 1 option n'appartenant pas au 96 options retenues

df["autre_options"] = 52513*[0]
for i in range(df.shape[0]) :
    print(i)
    if op_par_voit[i] != [] :
        n = len(op_par_voit[i])
        autre = False
        j = 0
        while not(autre) and j < n :
            autre = (op_par_voit[i][j] not in options2)
            j += 1
        if autre :
            df["autre_options"][i] = 1


### ACP sur les variables options retenues

from prince import PCA

base_acp = df.iloc[:,23:]      # on sélectionne les 97 variables indicatrices des options
pca = PCA(n_components =97)
pca = pca.fit(base_acp)

p = pca.n_components
eigen = pca.eigenvalues_
inertia = pca.explained_inertia_

eigendf = pd.DataFrame(eigen, columns = ["eigen"], index = ["Dim {}".format(i) for i in range(1, p+1,1)])
print(eigendf)

inertiadf = pd.DataFrame(inertia, columns = ["inertia"], index = ["Dim {}".format(i) for i in range(1, p+1,1)])
print(inertiadf)

inertiadf["inertia"][1:30]
eigendf["eigen"][1:30]

#Règle de Kaiser : 13 axes retenus car 13 axes pour lesquels la valeur propre est supérieur à 1


### Création des axes avec les coordonnées

rowCoord = pca.row_coordinates(base_acp)
table = pd.read_csv('D:/Projet stat 2A/projet_stat/.aws/df_modelisation.csv')      # récupération du tableau de données modifiés dans les autres fichiers
for i in range(13) :
    table["axe %s" %(i+1)] = rowCoord[i]
table.to_csv('D:/Projet stat 2A/projet_stat/.aws/df_modelisation.csv', index = False)


