#Importation des packages.
import string
import regex as re
import pandas as pd
from pandas import to_datetime

#Importation du data frame.
df = pd.read_csv("s3://projet-stat-ensai/Lacentrale.csv",sep=';', dtype=str)

#Suppression de certaines variables.
df.drop([u'reference', u'url', u'user_id'],axis=1,inplace=True)
df.drop([u'type', u'marque',u'modele'], axis=1, inplace=True)

#creation de la variable age

df["date_mec"] = to_datetime(df["date_mec"])
df["date_publication"] = to_datetime(df["date_publication"]).dt.tz_localize(None)
df["date_depublication"] = to_datetime(df["date_depublication"]).dt.tz_localize(None)
df["age"] = (df["date_depublication"]  - df["date_mec"]).dt.days

#Uniformisation de la variable prix.
df[u'prix_vente']=pd.to_numeric(df[u'prix_vente'], downcast='integer')

#Uniformisation de la variable boite de vitesse.
df["boite_de_vitesse"] = df["boite_de_vitesse"].replace(["mÃ©canique", "mécanique"], "meca")
df["boite_de_vitesse"] = df["boite_de_vitesse"].replace("automatique", "auto")

for i in range(df.shape[0]):
    if type(df[u'couleur'][i])==str:
        df[u'couleur'][i]=df[u'couleur'][i].lower()
#print(df.loc[df[u'couleur']=="blnache",:])
df.iloc[44695,15]="blanc"

df[u"couleur"].replace("blnache", "blanc",inplace= True)
df[u"couleur"].replace("noiir", "noir",inplace= True)
df[u"couleur"].replace("blanche", "blanc",inplace= True)
df[u"couleur"].replace("noire", "noir",inplace= True)
df[u"couleur"].replace("grise", "gris",inplace= True)
df[u"couleur"].replace("roue flamme", "rouge flamme",inplace= True)
df[u"couleur"].replace("blnache", "blanc",inplace= True)
df[u"couleur"].replace("*lanc", "blanc",inplace= True)

#valeur manquantes de couleur:
df[u"couleur"] = df[u"couleur"].fillna("autre")
df[u"couleur"].replace("2017el50360", "autre",inplace= True)
df[u"couleur"].replace("+", "autre",inplace= True)
df[u"couleur"].replace("eqb", "autre",inplace= True)
df[u"couleur"].replace("eehbrunral8007", "autre",inplace= True)
df[u"couleur"].replace(".", "autre",inplace= True)
df[u"couleur"].replace("inc", "autre",inplace= True)
df[u"couleur"].replace("inc.", "autre",inplace= True)
df[u"couleur"].replace("inconn", "autre",inplace= True)
df[u"couleur"].replace("inconnu", "autre",inplace= True)
df[u"couleur"].replace("inconnue", "autre",inplace= True)
df[u"couleur"].replace("inconu", "autre",inplace= True)
df[u"couleur"].replace("kpn", "autre",inplace= True)
df[u"couleur"].replace("n.c.", "autre",inplace= True)
df[u"couleur"].replace("n/a", "autre",inplace= True)
df[u"couleur"].replace("nc", "autre",inplace= True)
df[u"couleur"].replace("neutre.", "autre",inplace= True)
df[u"couleur"].replace("xx", "autre",inplace= True)
df[u"couleur"].replace("rqh", "autre",inplace= True)

df[u"couleur"].replace("non codifiee", "autre",inplace= True)
df[u"couleur"].replace("non codifie", "autre",inplace= True)

df[u"couleur"].replace("non renseigne", "autre",inplace= True)
df[u"couleur"].replace('autre / non affect\xc3\xa9', "autre",inplace= True)
for j in range(df.shape[0]):
    if type(df[u'couleur'][j])==str:
        if df[u'couleur'][j][0:2] == "n1":
            df[u'couleur'][j]='autre'

#détection des couleurs au milieu de la chaine de caractère

for j in df.index.tolist():
    if "noir" in df[u'couleur'][j].strip(".").split():
        df[u'couleur'][j]='noir'
    elif "rouge" in df[u'couleur'][j].strip(".").split():
        df[u'couleur'][j]='rouge'
    elif "jaune" in df[u'couleur'][j].strip(".").split():
        df[u'couleur'][j]='jaune'
    elif "gris" in df[u'couleur'][j].strip(".").split():
        df[u'couleur'][j]='gris'
    elif "blanc" in df[u'couleur'][j].strip(".").split():
        df[u'couleur'][j]='blanc'
    elif "rouge" in df[u'couleur'][j].strip(".").split():
        df[u'couleur'][j]='rouge'
    elif "bleu" in df[u'couleur'][j].strip(".").split():
        df[u'couleur'][j]='bleu'


df[u"couleur"].replace('1602832gris fonce', "gris",inplace= True)
df['couleur'] = df['couleur'].str.replace(u"é", "e")
df['couleur'] = df['couleur'].str.replace(u"É", "e")
df[u"couleur"].replace('9458blanc gl', "blanc",inplace= True)
df[u"couleur"].replace('orance valencia', "orange",inplace= True)
df[u"couleur"].replace('nior etoile', "noir",inplace= True)
df[u"couleur"].replace('black', "noir",inplace= True)
df[u"couleur"].replace('lanc glacier', "blanc",inplace= True)
for j in df.index:
    if df[u'couleur'][j][0:4]=="gris":
        df[u'couleur'][j]="gris"
    elif  df[u'couleur'][j][0:3]=="noi":
        df[u'couleur'][j] = "noir"
    elif df[u'couleur'][j][0:5]=="blanc":
        df[u'couleur'][j] = "blanc"
    elif df[u'couleur'][j][0:5]=="beige":
        df[u'couleur'][j] = "beige"
    elif df[u'couleur'][j][0:4]=="bleu":
        df[u'couleur'][j] = "bleu"
    elif df[u'couleur'][j][0:6]=="orange":
        df[u'couleur'][j] = "orange"
    elif df[u'couleur'][j][0:2]=="or":
        df[u'couleur'][j] = "or"
    elif df[u'couleur'][j][0:6]=="marron":
        df[u'couleur'][j] = "marron"
    elif df[u'couleur'][j][0:6]=="violet":
        df[u'couleur'][j] = "violet"
    elif df[u'couleur'][j][0:4]=="vert":
        df[u'couleur'][j] = "vert"
    elif df[u'couleur'][j][0:4]=="roug":
        df[u'couleur'][j] = "rouge"
    elif df[u'couleur'][j][0:5]=="jaune":
        df[u'couleur'][j] = "jaune"
    elif df[u'couleur'][j][0:6]=="platin":
        df[u'couleur'][j] = "platine"
    elif df[u'couleur'][j][0:4]=="tita":
        df[u'couleur'][j] = "titane"

#unifomisation de couleur:

eff = df["couleur"].value_counts()
pourcent = df["couleur"].value_counts(normalize = True)
eff = pd.concat([eff, pourcent], axis = "columns")
pourcent_col=pd.DataFrame(list(pourcent.items()),columns=['couleur','pourcentage'])
couleurs = pourcent_col[ pourcent_col["pourcentage"]<0.01].couleur
for nom in couleurs:
    df["couleur"].replace(nom, "autre",inplace=True)


#Traitement des valeurs manquantes :
nan = float('nan')

#Récupération de la colonne version:
Version = df["version"]

def Recuperation_generation_voiture():
    n = len(df)
    L = []
    for i in range(n):
            if re.match(r"I\s", Version[i]) != None:
                L.append("CLIO")
            elif re.match(r"II\s", Version[i]) != None:
                L.append("CLIO 2")
            elif re.match(r"III\s", Version[i]) != None:
                L.append("CLIO 3")
            elif re.match(r"IV\s", Version[i]) != None:
                L.append("CLIO 4")
            elif re.match(r"V\s", Version[i]) != None:
                L.append("CLIO 5")
            else:
                L.append(nan)
    return(L)

L = Recuperation_generation_voiture() #Il reste 10 erreurs dans la base : les variables modèles et version n'indiquent pas la même génération pour la voiture.

#Ajout de la colonne génération au data frame:
df["generation"] = pd.Series(L, index = df.index)

#Création d'une fonction permettant de compléter les valeurs manquantes de la variable modele_com à partir de la variable
# generation construite en exploitant la variable version.

def Fill_modele_com():
    n = len(df)
    list_of_missing_model_com = df["modele_com"].isna()
    list_of_missing_generation = df["generation"].isna()
    for i in range(n):
        if list_of_missing_model_com[i] == True and list_of_missing_generation[i] == False:
            df["modele_com"][i] = df["generation"][i]
    return(True)

Fill_modele_com()

#Création d'une fonction permettant d'uniformiser les modalités de la variable modele_com:

def Uniformisation_modele_com():
    n = len(df)
    Indice_modele_invalide = []
    for i in range(n):
        if type(df["modele_com"][i]) == str:
            if "CLIO 1" in df["modele_com"][i] or "CLIO SOCIETE" in df["modele_com"][i]:
                df["modele_com"][i] = "CLIO"
            elif "CLIO 2" in df["modele_com"][i] or "WILLIAMS" in df["modele_com"][i]:
                df["modele_com"][i] = "CLIO 2"
            elif "CLIO 3" in df["modele_com"][i]:
                df["modele_com"][i] = "CLIO 3"
            elif "CLIO 4" in df["modele_com"][i]:
                df["modele_com"][i] = "CLIO 4"
            elif "CLIO 5" in df["modele_com"][i]:
                df["modele_com"][i] = "CLIO 5"
            elif "ZOE" in df["modele_com"][i]:
                Indice_modele_invalide.append(i)
            elif "GRAND" in df["modele_com"][i]:
                Indice_modele_invalide.append(i)
            elif "CAPTUR" in df["modele_com"][i]:
                Indice_modele_invalide.append(i)
            elif "MEGANE" in df["modele_com"][i]:
                Indice_modele_invalide.append(i)
            elif "TWINGO" in df["modele_com"][i]:
                Indice_modele_invalide.append(i)
    return(Indice_modele_invalide)

Liste_Modele_Invalide = Uniformisation_modele_com()

#Suppression des modalités n'étant pas des RENAULT CLIO et réindexation des observation du dataframe:
# On retire 7 modalités du data frame : 52513 observations ===> 52506 observations
df = df.drop(Liste_Modele_Invalide)
df.reset_index(drop=True, inplace=True)

#Récupération de la liste des index des valeurs manquantes et création d'un dataframe ne comportant que les observations ayant des valeurs manquantes dans la variable modele_com.

index_of_missing_values_model_com = df[df['modele_com'].isnull()].index.tolist()
df_missing_values_modele_com = df.loc[index_of_missing_values_model_com,]

#Création de plusieurs foncions permettant de rechercher les index des observations sans valeurs manquantes pour modele_com ayant une version similaire
# aux observations ayant des valeurs manquantes pour modele_com.

def search_index_MEDIANAV():
    n = len(df)
    L_medianav = []
    for i in range(n):
        if "MEDIANAV" in df["version"][i]:
            if type(df["modele_com"][i]) == str:
                L_medianav.append(i)
    return(L_medianav)

def search_index_AIR():
    n = len(df)
    L_AIR = []
    for i in range(n):
        if "1.5" in df["version"][i] and "DCI" in df["version"][i] and "AIR" in df["version"][i]:
            if type(df["modele_com"][i]) == str:
                L_AIR.append(i)
    return(L_AIR)

def search_index_ENERGY():
    n = len(df)
    L_energy = []
    for i in range(n):
        if "1.5" in df["version"][i] and "ENERGY" in df["version"][i] and "DCI" in df["version"][i]:
            if type(df["modele_com"][i]) == str:
                L_energy.append(i)
    return(L_energy)

#Création des data frames associées à chacune des listes renvoyées par la fonction ci-dessus, et visualisation de la génération
#des véhicules ayant des versions similaires aux véhicules dont la génération est manquante.

L_medianav = search_index_MEDIANAV()
df_version_medianav = df.loc[L_medianav,]
df_version_medianav["modele_com"].value_counts() #Tous les véhicules comportant un systeme MEDIANAV sont des CLIO 4.

L_AIR = search_index_AIR()
df_version_AIR= df.loc[L_AIR,]
df_version_AIR["modele_com"].value_counts() #Le mode est CLIO 4.

L_ENERGY = search_index_ENERGY()
df_version_ENERGY = df.loc[L_ENERGY,]
df_version_ENERGY["modele_com"].value_counts() #15296 véhicules des versions comportant les mots 1.5, ENERGY et DCI sont des CLIOS 4, 1 véhicule comportant ces trois mots est une CLIO 3.


# Création d'une fonction permettant d'imputer les valeurs manquantes de modele_com à partir des observations présentant une version similaire.
def Imputation_modele_com():
    n = len(df)
    for i in index_of_missing_values_model_com:
        if "MEDIANAV" in df["version"][i]:
            df["modele_com"][i] = "CLIO 4"
        elif "MEDIA" in df["version"][i]:
            df["modele_com"][i] = "CLIO 4"
        elif "1.5" and "DCI" and "AIR" in df["version"][i]:
            df["modele_com"][i] = "CLIO 4"
        elif "1.5" and "ENERGY" and "DCI" in df["version"][i]:
            df["modele_com"][i] = "CLIO 4"

Imputation_modele_com()

#Mise à jour du data frame comportant les valeurs manquantes restantes de modele_com.
index_of_missing_values_model_com = df[df['modele_com'].isnull()].index.tolist()
df_missing_values_modele_com = df.loc[index_of_missing_values_model_com,]

# Imputation manuelle des 16 dernières valeurs manquantes en cherchant la génération des véhicules ayant une version similaires à ceux-ci
# (J'ai modifié la fonction au fur et à mesure, toutes les recherches ne sont pas visibles ici et la fonction écrite ci-dessous est surtout présente pour illustrer la démarche).
def search_quick():
    n = len(df)
    L_END = []
    for i in range(n):
        if "SOCIETE" in df["version"][i] and "90" in df["version"][i] and "TCE" in df["version"][i] and "ENERGY" in df["version"][i]:
            if type(df["modele_com"][i]) == str:
                L_END.append(i)
    return(L_END)
L_END = search_quick()
df_version_EDITION= df.loc[L_END,]
df_version_EDITION["modele_com"].value_counts()

#Imputation des 16 dernières valeurs manquantes à partir des recherches effectuées ci-dessus.
def Fin_imputation_modele_com():
    df["modele_com"][14301] = "CLIO 2"
    df["modele_com"][50411] = "CLIO 2"
    df["modele_com"][11117] = "CLIO 4"
    df["modele_com"][11539] = "CLIO 4"
    df["modele_com"][19966] = "CLIO 4"
    df["modele_com"][16702] = "CLIO 4"
    df["modele_com"][17903] = "CLIO 4"
    df["modele_com"][10449] = "CLIO 4"
    df["modele_com"][22999] = "CLIO 4"
    df["modele_com"][20861] = "CLIO 5"
    df["modele_com"][27024] = "CLIO 5"
    df["modele_com"][48021] = "CLIO 4"
    df["modele_com"][35305] = "CLIO 4"
    df["modele_com"][28189] = "CLIO 3"
    df["modele_com"][4220] = "CLIO 2"
    df["modele_com"][37468] = "CLIO 4"
    return("Terminé")
Fin_imputation_modele_com()
#Uniformisation de la variable energie.

def Uniformisation_energie():
    n = len(df)
    for i in range(n):
            if "GPL" in df["energie"][i]:
                df["energie"][i] = "Essence"
            elif "Bicarburation" in df["energie"][i]:
                df["energie"][i] = "Essence"
            elif "Biocarburant" in df["energie"][i]:
                df["energie"][i] = "Diesel"
    return()

Uniformisation_energie()


#Récupération cylindrée
def Recuperation_cylindree():
    n = len(df)
    L = []
    for i in range(n):
        if "0.9" in Version[i]:
            L.append("0.9")
        elif "1.0" in Version[i]:
            L.append("1.0")
        elif "1.2" in Version[i]:
            L.append("1.2")
        elif "1.3" in Version[i]:
            L.append("1.3")
        elif "1.4" in Version[i]:
            L.append("1.4")
        elif "1.5" in Version[i]:
            L.append("1.5")
        elif "1.6" in Version[i]:
            L.append("1.6")
        elif "1.9" in Version[i]:
            L.append("1.9")
        elif "2.0" in Version[i]:
            L.append("2.0")
        elif "9.3" in Version[i]:
            L.append("9.3")
        else:
            L.append(nan)
    return(L)

L_Cyl = Recuperation_cylindree()
df["Cylindree"] = pd.Series(L_Cyl, index = df.index)

def Fill_engine():
    n = len(df)
    list_of_missing_engine = df["engine"].isna()
    list_of_missing_cylindree = df["Cylindree"].isna()
    for i in range(n):
        if list_of_missing_engine[i] == True and list_of_missing_cylindree[i] == False:
            df["engine"][i] = df["Cylindree"][i]
    return("Terminé")

Fill_engine()



#Regroupement des observations selon les variables modele_com et energie :
df["engine"] = df.groupby(['modele_com','energie'])["engine"].transform(lambda x: x.fillna(x.mode()[0]))

#Récupération Horse-Power:

def Recuperation_Horse_Power():
    n = len(df)
    L = []
    for i in range(n):
        if "90" in Version[i]:
            L.append("90")
        elif "75" in Version[i]:
            L.append("75")
        elif "100" in Version[i]:
            L.append("100")
        elif "85" in Version[i]:
            L.append("85")
        elif "120" in Version[i]:
            L.append("120")
        elif "130" in Version[i]:
            L.append("130")
        elif "115" in Version[i]:
            L.append("115")
        elif "110" in Version[i]:
            L.append("110")
        elif "70" in Version[i]:
            L.append("70")
        elif "200" in Version[i]:
            L.append("200")
        elif "65" in Version[i]:
            L.append("65")
        elif "220" in Version[i]:
            L.append("220")
        elif "60" in Version[i]:
            L.append("60")
        elif "105" in Version[i]:
            L.append("105")
        elif "95" in Version[i]:
            L.append("95")
        elif "80" in Version[i]:
            L.append("80")
        elif "140" in Version[i]:
            L.append("140")
        elif "203" in Version[i]:
            L.append("203")
        elif "128" in Version[i]:
            L.append("128")
        elif "255" in Version[i]:
            L.append("255")
        elif "182" in Version[i]:
            L.append("182")
        elif "230" in Version[i]:
            L.append("230")
        elif "55" in Version[i]:
            L.append("68")
        elif "172" in Version[i]:
            L.append("172")
        elif "201" in Version[i]:
            L.append("201")
        elif "12" in Version[i]:
            L.append("12")
        elif "16" in Version[i]:
            L.append("16")
        elif "88" in Version[i]:
            L.append("88")
        else:
            L.append(nan)
    return (L)

L_horsepower = Recuperation_Horse_Power()
df["Horsepower_version"] = pd.Series(L_horsepower, index = df.index)

def Fill_horsepower():
    n = len(df)
    list_of_missing_horsepower = df["horsepower"].isna()
    list_of_missing_Horsepower_version = df["Horsepower_version"].isna()
    for i in range(n):
        if list_of_missing_horsepower[i] == True and list_of_missing_Horsepower_version[i] == False:
            df["horsepower"][i] = df["Horsepower_version"][i]
    return("Terminé")

Fill_horsepower()

#Regroupement des observations selon les variables modele_com et energie.
df["horsepower"] = df.groupby(['modele_com','energie'])["horsepower"].transform(lambda x: x.fillna(x.mode()[0]))

#Imputation des trois valeurs manquantes de puissance_fiscale.

df["puissance_fiscale"] = df.groupby(['modele_com','energie'])["puissance_fiscale"].transform(lambda x: x.fillna(x.mode()[0]))

#Récupération des finitions.
def Recuperation_finition():
    n = len(df)
    L = []
    for i in range(n):
        if "LIFE" in Version[i]:
            L.append("Life")
        elif "TREND" in Version[i]:
            L.append("Trend")
        elif "ZEN" in Version[i]:
            L.append("Zen")
        elif "LIMITED" in Version[i]:
            L.append("Limited")
        elif "BUSINESS" in Version[i]:
            L.append("Business")
        elif "INTENS" in Version[i]:
            L.append("Intens")
        elif "PRIVILEGE" in Version[i]:
            L.append("Privilege")
        elif "AUTHENTIQUE" in Version[i]:
            L.append("Authentique")
        elif "SOCIETE" in Version[i]:
            L.append("Societe")
        elif "ALIZE" in Version[i]:
            L.append("Alize")
        elif "INITIALE" in Version[i]:
            L.append("Initiale")
        elif "TOMTOM" in Version[i]:
            L.append("Tomtom")
        elif "AIR" in Version[i]:
            L.append("Societe")
        elif "CAMPUS" in Version[i]:
            L.append("Campus")
        elif "EXPRESSION" in Version[i]:
            L.append("Expression")
        elif "GENERATION" in Version[i]:
            L.append("Generation")
        elif "EXCEPTION" in Version[i]:
            L.append("Exception")
        elif "RS" in Version[i]:
            L.append("RS")
        elif "RS LINE" in Version[i]:
            L.append("RS LINE")
        elif "DYNAMIQUE" in Version[i]:
            L.append("Dynamique")
        elif "GT" in Version[i]:
            L.append("GT")
        elif "SPORT" in Version[i]:
            L.append("Sport")
        elif "ONE" in Version[i]:
            L.append("One")
        elif "EXTREME" in Version[i]:
            L.append("Extreme")
        elif "BILLABONG" in Version[i]:
            L.append("Billabong")
        elif "LUXE" in Version[i]:
            L.append("Luxe")
        elif "RTA" in Version[i]:
            L.append("RTA")
        elif "RTE" in Version[i]:
            L.append("RTE")
        elif "RXE" in Version[i]:
            L.append("RXE")
        elif "RXT" in Version[i]:
            L.append("RXT")
        elif "SI" in Version[i]:
            L.append("SI")
        elif "BASE" in Version[i]:
            L.append("Base")
        else:
            L.append(nan)
    return(L)

L_Finition = Recuperation_finition()

df["Finition"] = pd.Series(L_Finition, index = df.index)

#Imputation des valeurs manquantes de la variable Finition.
df["Finition"] = df.groupby(['modele_com','energie'])["Finition"].transform(lambda x: x.fillna(x.mode()[0]))

#Regroupement modalités finitions :

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

#Suppression des variables créés afin d'exploiter la variable version.
df.drop([u'Horsepower_version', u'Cylindree', u'generation'],axis=1,inplace=True)
df.drop([u'version'],axis=1,inplace=True)

#Imputation des valeurs manquantes de porte.
df["porte"] = df.groupby(['modele_com','energie'])["porte"].transform(lambda x: x.fillna(x.mode()[0]))

df.to_csv(".aws/df_modelisation.csv")