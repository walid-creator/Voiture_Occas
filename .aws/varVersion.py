#Importation des packages:
import string
import regex as re
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import unicodedata

#Importation de la base de données:
df = pd.read_csv("s3://projet-stat-ensai/Lacentrale.csv",sep=';', dtype=str)
df = pd.read_csv(".aws/df_modelisation.csv",error_bad_lines=False,index_col=0)

#Traitement des valeurs manquantes :
nan = float('nan')

print(df["version"].isna().value_counts()) #Pas de valeurs manquantes.

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

#Vérification des valeurs manquantes dans chacune des colonnes :

print(df["modele_com"].isna().value_counts()) #Avant 998 valeurs manquantes dans modele_com.
print(df["generation"].isna().value_counts()) #Avec 1386 valeurs manquantes dans generation.

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

#Visualisation des modalités de la variable modele_com :
df['modele_com'].value_counts()

#Recomptage des valeurs manquantes de la variable modele_com :
print(df["modele_com"].isna().value_counts()) #Après avoir complété modele_com grâce à la variable version  il n'y a plus que 256 valeurs manquantes.

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

#Visualisation des valeurs manquantes après imputation :
df["modele_com"].isna().value_counts() # il reste 16 valeurs manquantes après avoir rajouté les véhicules MEDIANAV

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

#Visualisation des valeurs manquantes et des modalités de modele_com :
print(df["modele_com"].isna().value_counts()) # Il n'y a plus de valeur manquante dans la variable modele_com.
print(df["modele_com"].value_counts())

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

print(df["energie"].value_counts())

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

#Visualisation des valeurs manquantes dans engine:
print(df["engine"].isna().value_counts()) #Il y a 889 valeurs manquantes dans engine.
print(df["Cylindree"].isna().value_counts()) # Il y a 897 valeurs manquantes dans cylindree.

def Fill_engine():
    n = len(df)
    list_of_missing_engine = df["engine"].isna()
    list_of_missing_cylindree = df["Cylindree"].isna()
    for i in range(n):
        if list_of_missing_engine[i] == True and list_of_missing_cylindree[i] == False:
            df["engine"][i] = df["Cylindree"][i]
    return("Terminé")

Fill_engine()

df["engine"].isna().value_counts() #Il reste 177 valeurs manquantes.

#Imputation des valeurs manquantes de engine à partir de modele_com et energie.

#Regroupement des observations selon les variables modele_com et energie :
df["engine"] = df.groupby(['modele_com','energie'])["engine"].transform(lambda x: x.fillna(x.mode()[0]))

#Revisualisation des valeurs manquantes de engine :
print(df["engine"].isna().value_counts()) #Il n'y a plus de valeurs manquantes pour la variable engine.

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

#Visualisation des valeurs manquantes :
df["horsepower"].isna().value_counts() #Il y a 750 valeurs manquantes.
df["Horsepower_version"].value_counts() #Il y a 376 valeurs manquantes.

def Fill_horsepower():
    n = len(df)
    list_of_missing_horsepower = df["horsepower"].isna()
    list_of_missing_Horsepower_version = df["Horsepower_version"].isna()
    for i in range(n):
        if list_of_missing_horsepower[i] == True and list_of_missing_Horsepower_version[i] == False:
            df["horsepower"][i] = df["Horsepower_version"][i]
    return("Terminé")

Fill_horsepower()

#Revisualisation des valeurs manquantes de horsepower.
df["horsepower"].isna().value_counts() #Il reste 99 valeurs manquantes.

#Regroupement des observations selon les variables modele_com et energie.
df["horsepower"] = df.groupby(['modele_com','energie'])["horsepower"].transform(lambda x: x.fillna(x.mode()[0]))

#Revisualisation des valeurs manquantes de horsepower :
print(df["horsepower"].isna().value_counts()) #Il n'y a plus de valeurs manquantes pour la variable horsepower.

#Imputation des trois valeurs manquantes de puissance_fiscale.

df["puissance_fiscale"] = df.groupby(['modele_com','energie'])["puissance_fiscale"].transform(lambda x: x.fillna(x.mode()[0]))

df["puissance_fiscale"].isna().value_counts()

#Ajout d'une colonne type de moteur à la base de données:
def Recuperation_type_de_moteur():
    n = len(df)
    L = []
    for i in range(n):
        if "DCI" in Version[i]:
            L.append("DCI ")
        elif "TCE" in Version[i]:
            L.append("TCE ")
        else:
            L.append(nan)
    return(L)

L_Moteur = Recuperation_type_de_moteur()
df["type_moteur"] = pd.Series(L_Moteur, index = df.index)

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
df["Finition"].isna().value_counts() #1157 valeurs manquantes dans la variable Finition.

#Imputation des valeurs manquantes de la variable Finition.
df["Finition"] = df.groupby(['modele_com','energie'])["Finition"].transform(lambda x: x.fillna(x.mode()[0]))
df["Finition"].isna().value_counts()
df["Finition"].value_counts()

def Recuperer_modalite_fintion():
    Finition_Intens = []
    Finition_Business = []
    Finition_Limited = []
    Finition_Zen = []
    Finition_Societe = []
    Finition_Trend = []
    Finition_RS = []
    Finition_Expression = []
    Autre_finition = []
    n = len(df)
    for i in range(n):
            if df["Finition"][i] == "Intens":
                Finition_Intens.append(1)
                Finition_Business.append(0)
                Finition_Limited.append(0)
                Finition_Zen.append(0)
                Finition_Societe.append(0)
                Finition_Trend.append(0)
                Finition_RS.append(0)
                Finition_Expression.append(0)
                Autre_finition.append(0)
            elif df["Finition"][i] == "Business":
                Finition_Intens.append(0)
                Finition_Business.append(1)
                Finition_Limited.append(0)
                Finition_Zen.append(0)
                Finition_Societe.append(0)
                Finition_Trend.append(0)
                Finition_RS.append(0)
                Finition_Expression.append(0)
                Autre_finition.append(0)
            elif df["Finition"][i] == "Limited":
                Finition_Intens.append(0)
                Finition_Business.append(0)
                Finition_Limited.append(1)
                Finition_Zen.append(0)
                Finition_Societe.append(0)
                Finition_Trend.append(0)
                Finition_RS.append(0)
                Finition_Expression.append(0)
                Autre_finition.append(0)
            elif df["Finition"][i] == "Zen":
                Finition_Intens.append(0)
                Finition_Business.append(0)
                Finition_Limited.append(0)
                Finition_Zen.append(1)
                Finition_Societe.append(0)
                Finition_Trend.append(0)
                Finition_RS.append(0)
                Finition_Expression.append(0)
                Autre_finition.append(0)
            elif df["Finition"][i] == "Societe":
                Finition_Intens.append(0)
                Finition_Business.append(0)
                Finition_Limited.append(0)
                Finition_Zen.append(0)
                Finition_Societe.append(1)
                Finition_Trend.append(0)
                Finition_RS.append(0)
                Finition_Expression.append(0)
                Autre_finition.append(0)
            elif df["Finition"][i] == "Trend":
                Finition_Intens.append(0)
                Finition_Business.append(0)
                Finition_Limited.append(0)
                Finition_Zen.append(0)
                Finition_Societe.append(0)
                Finition_Trend.append(1)
                Finition_RS.append(0)
                Finition_Expression.append(0)
                Autre_finition.append(0)
            elif df["Finition"][i] == "RS":
                Finition_Intens.append(0)
                Finition_Business.append(0)
                Finition_Limited.append(0)
                Finition_Zen.append(0)
                Finition_Societe.append(0)
                Finition_Trend.append(0)
                Finition_RS.append(1)
                Finition_Expression.append(0)
                Autre_finition.append(0)
            elif df["Finition"][i] == "Expression":
                Finition_Intens.append(0)
                Finition_Business.append(0)
                Finition_Limited.append(0)
                Finition_Zen.append(0)
                Finition_Societe.append(0)
                Finition_Trend.append(0)
                Finition_RS.append(0)
                Finition_Expression.append(1)
                Autre_finition.append(0)
            else:
                Finition_Intens.append(0)
                Finition_Business.append(0)
                Finition_Limited.append(0)
                Finition_Zen.append(0)
                Finition_Societe.append(0)
                Finition_Trend.append(0)
                Finition_RS.append(0)
                Finition_Expression.append(0)
                Autre_finition.append(1)
    return(Finition_Intens, Finition_Business, Finition_Limited, Finition_Zen, Finition_Societe, Finition_Trend, Finition_RS, Finition_Expression, Autre_finition)

Finition_Intens, Finition_Business, Finition_Limited, Finition_Zen, Finition_Societe, Finition_Trend, Finition_RS, Finition_Expression, Autre_finition = Recuperer_modalite_fintion()

df["finition_intens"] = pd.Series(Finition_Intens, index = df.index)
df["finition_business"] = pd.Series(Finition_Business, index = df.index)
df["finition_limited"] = pd.Series(Finition_Limited, index = df.index)
df["finition_zen"] = pd.Series(Finition_Zen, index = df.index)
df["finition_societe"] = pd.Series(Finition_Societe, index = df.index)
df["finition_trend"] = pd.Series(Finition_Trend, index = df.index)
df["finition_rs"] = pd.Series(Finition_RS, index = df.index)
df["finition_expression"] = pd.Series(Finition_Expression, index = df.index)
df["autre_finition"] = pd.Series(Autre_finition, index = df.index)


#Ajout des variables de version au data frame.
df.to_csv(".aws/automobile.csv")

df.to_csv(".aws/automobile2.csv")