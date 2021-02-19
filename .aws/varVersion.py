#Importation des packages:

import string
import regex as re
import pandas as pd

#Importation de la base de données:
#df = pd.read_csv("s3://projet-stat-ensai/Lacentrale.csv",sep=';', dtype=str)
df = pd.read_csv(".aws/automobile.csv",error_bad_lines=False,index_col=0)
#Traitement de la variable version

#Traitement des valeurs manquantes :
df['version'] = df['version'].fillna("")

#Récupération de la colonne version:
Version = df1["version"]

def Recuperation_generation_voiture():
    n = len(Version)
    L = []
    for i in range(n):
            if "I " in Version[i] and "II" not in Version[i] and "V" not in Version[i]:
                L.append("I")
            elif "II" in Version[i] and "III" not in Version[i]:
                L.append("II")
            elif "III" in Version[i]:
                L.append("III")
            elif "IV" in Version[i]:
                L.append("IV")
            elif "V" in Version[i] and "IV" not in Version[i]:
                L.append("V")
            else:
                L.append("")
    return(L)
L = Recuperation_generation_voiture()

#Ajout de la colonne génération au data frame:
df["generation"] = pd.Series(L, index = df.index)

#Ajout d'une colonne type de moteur à la base de données:

def Recuperation_type_de_moteur():
    n = len(Version)
    L = []
    for i in range(n):
        if "DCI" in Version[i]:
            L.append("DCI")
        elif "TCE" in Version[i]:
            L.append("TCE")
        else:
            L.append(" ")
    return(L)

L_Moteur = Recuperation_type_de_moteur()
df["type_moteur"] = pd.Series(L_Moteur, index = df.index)


def Recuperation_finition():
    n = len(Version)
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
            L.append("Air")
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
        elif "DYNAMIQUE" in Version[i]:
            L.append("Dynamique")
        elif "GT" in Version[i]:
            L.append("GT")
        elif "Sport" in Version[i]:
            L.append("Sport")
        elif "ONE" in Version[i]:
            L.append("One")
        else:
            L.append(" ")
    return(L)

L_Finition = Recuperation_finition()
df["Finition"] = pd.Series(L_Finition, index = df.index)

def Recuperation_cylindree():
    n = len(Version)
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
        else:
            L.append(" ")
    return(L)

L_Cyl = Recuperation_cylindree()
df["Cylindree"] = pd.Series(L_Cyl, index = df.index)

#Ajout des variables de version au data frame traité par Walid.

df.to_csv(".aws/automobile.csv")
