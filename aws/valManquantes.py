## Chargement du repertoire de travail
from os import chdir
chdir("C:/Users/duver/Desktop/codeprojet")

# Chargement du dataset
from pandas import read_excel
df = read_excel("data.xlsx", sheet_name = 0, header =0, index_col = 0)

### Extraction des variables quantitatives
vsQuant = df[["horsepower", "engine", "kilometrage",
                "puissance_fiscale","prix_vente"]]

# Convertion en float
vsQuant = vsQuant.astype(float)

# Remplacer NAN en utilisant la valeur médiane
vsQuant = vsQuant.fillna(vsQuant.median())

##### NUAGE DE POINTS
from plotnine import *

## Nuage de points : relation horsepower et prix_vente
gph1 = (ggplot(vsQuant) +
        geom_point(aes(x = "horsepower",y = "prix_vente"))+
        labs(title = ("Nuage de points")))
##
gph2 = (ggplot(vsQuant) +
        geom_point(aes(x = "engine",y = "prix_vente"))+
        labs(title = ("Nuage de points")))

##
gph3 = (ggplot(vsQuant) +
        geom_point(aes(x = "kilometrage",y = "prix_vente"))+
        labs(title = ("Nuage de points")))

##
gph4 = (ggplot(vsQuant) +
        geom_point(aes(x = "puissance_fiscale",y = "prix_vente"))+
        labs(title = ("Nuage de points")))

## Affichage
"""
print(gph1)
print(gph2)
print(gph3)
print(gph4)
"""

#### Extraction des variables qualitative
vsQual = df[["modele_com","couleur","energie","boite_de_vitesse",
               "premiere_main","departement", "porte"]]

## Uniformisation des variables
vsQual.boite_de_vitesse = vsQual.boite_de_vitesse.replace(["mÃ©canique", "mécanique"], "meca")
vsQual.boite_de_vitesse = vsQual.boite_de_vitesse.replace("automatique", "auto")

from numpy import unique
modalite = unique(vsQual.boite_de_vitesse) # 2 modalités

# Recodage de la variable energie

vsQual.loc[vsQual["energie"] == "Bicarburation essence GPL","energie"] = "begpl"
vsQual.loc[vsQual["energie"] == "Bicarburation essence bioéthanol","energie"] = "bebeth"
vsQual.loc[vsQual["energie"] == "Biocarburant","energie"] = "biocar"
vsQual.loc[vsQual["energie"] == "Diesel","energie"] = "diesel"
vsQual.loc[vsQual["energie"] == "Electrique","energie"] = "electrik"
vsQual.loc[vsQual["energie"] == "Essence","energie"] = "essence"
vsQual.loc[vsQual["energie"] == "Hybride essence électrique","energie"] = "heelect"
modalite = unique(vsQual.energie) # 6 modalités

### Recodage de la variable modele_com
vsQual.loc[vsQual["modele_com"] == "nan","modele_com"] = "CLIO 4"

from pandas import concat
data = concat([vsQuant, vsQual.energie], axis = "columns")

### Box plot
## Boxplot
gph1 = (ggplot(data) +
        geom_boxplot(aes(x = "energie", y = "prix_vente"))+
        labs(title = ("Boxplot")))
gph2 = (ggplot(data) +
        geom_boxplot(aes(x = "energie", y = "horsepower"))+
        labs(title = ("Boxplot")))
gph3 = (ggplot(data) +
        geom_boxplot(aes(x = "energie", y = "engine"))+
        labs(title = ("Boxplot")))
gph4 = (ggplot(data) +
        geom_boxplot(aes(x = "energie", y = "kilometrage"))+
        labs(title = ("Boxplot")))
gph5 = (ggplot(data) +
        geom_boxplot(aes(x = "energie", y = "puissance_fiscale"))+
        labs(title = ("Boxplot")))

print(gph1)
print(gph2)
print(gph3)
print(gph4)
print(gph5)


###
print(data.sort_values(by=['prix_vente']))

###
print(vsQual.isna().sum())

##
vsQual = vsQual.fillna(vsQual.mode)
print(vsQual.isna().sum())

#### Gestion des dates
from pandas import to_datetime
date = df[["date_mec", "date_publication", "date_depublication"]]
date["date_mec"] = to_datetime(date["date_mec"]).dt.tz_localize(None)
date["date_publication"] = to_datetime(date["date_publication"]).dt.tz_localize(None)
date["date_depublication"] = to_datetime(date["date_depublication"]).dt.tz_localize(None)
vsQuant["age"] = (date["date_publication"]  - date["date_mec"]).dt.days

##### ENCODAGE

from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
print(encoder.fit_transform(vsQual))
