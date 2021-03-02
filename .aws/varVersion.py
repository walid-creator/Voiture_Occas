#Importation des packages:

import string
import regex as re
import pandas as pd

#Importation de la base de données:
df = pd.read_csv("s3://projet-stat-ensai/Lacentrale.csv",sep=';', dtype=str)
df = pd.read_csv(".aws/automobile.csv",error_bad_lines=False,index_col=0)
#Traitement de la variable version

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

def Uniformisation_modele_com():
    n = len(df)
    Indice_modele_invalide = []
    for i in range(n):
        if type(df["modele_com"][i]) == str:
            if "CLIO 1" in df["modele_com"][i] :
                df["modele_com"][i] = "CLIO 1"
            elif "CLIO 2" in df["modele_com"][i] or "WILLIAMS" in df["modele_com"][i] or "SOCIETE" in df["modele_com"][i]:
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
df = df.drop(Liste_Modele_Invalide)
df.reset_index(drop=True, inplace=True)

df['modele_com'].value_counts()
#Recomptage des valeurs manquantes de la variable modele_com :

print(df["modele_com"].isna().value_counts()) #Après il n'y a plus que 256 valeurs manquantes.

#Les 256 valeurs manquantes restantes sont celles étant initialement manquantes dans modele_com et qui ne sont pas présente
#dans la variable version. Il faudra les imputer à l'aide de GROUP BY.


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

#Visualisation des valeurs manquantes:
df["type_moteur"].isna().value_counts()

Liste_finition = ["Life", "Trend", "Zen", "Limited", "Business", "Intens", "Initiale Paris", "RS Line",
                  "Authentique", "Expression", "Dynamique", "Exception", "GT", "Initiale", "Privilege", "Dynamique", "RS",
                  "Initiale", "Playstation 2", "Extreme", "Billabong", "Campus", "Sport", "Luxe Privilege", "Societe" ]

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
df["Finition"].isna().value_counts() #1157 valeurs manquantes mais on ne retiendra pour la suite que les modalités dont
                                     # la fréquence est supérieure à 10% de la base donc ça ne posera pas de problèmes.
seuil = len(df)/10

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

df["energie"].value_counts()

df["puissance_fiscale"].isna().value_counts()
df["Finition"] = df.groupby(["energie"]).transform(lambda x: x.fillna(x.mode()))

df["modele_com"].isna().value_counts()

X = df.groupby(["modele_com"]).describe()
X = df["Finition"].value_counts()

Liste_modalite_finition_a_conserver = ["Intens", "Business", "Limited", "Zen", "Societe", "Trend", "RS", "Expression"]

df["Finition"].isna().value_counts()

def Recuperer_modalite_fintion():
    Finition_Intens = []
    Finition_Business = []
    Finition_Limited = []
    Finition_Zen = []
    Finition_Societe = []
    Autre_finition = []
    n = len(df)
    for i in range(n):
        if type(df["Finition"][i]) == str:
            if df["Finition"][i] == "Intens":
                Finition_Intens.append(1)
                Finition_Business.append(0)
                Finition_Limited.append(0)
                Finition_Zen.append(0)
                Finition_Societe.append(0)
                Autre_finition.append(0)
            elif df["Finition"][i] == "Business":
                Finition_Intens.append(0)
                Finition_Business.append(1)
                Finition_Limited.append(0)
                Finition_Zen.append(0)
                Finition_Societe.append(0)
                Autre_finition.append(0)
            elif df["Finition"][i] == "Limited":
                Finition_Intens.append(0)
                Finition_Business.append(0)
                Finition_Limited.append(1)
                Finition_Zen.append(0)
                Finition_Societe.append(0)
                Autre_finition.append(0)
            elif df["Finition"][i] == "Zen":
                Finition_Intens.append(0)
                Finition_Business.append(0)
                Finition_Limited.append(0)
                Finition_Zen.append(1)
                Finition_Societe.append(0)
                Autre_finition.append(0)
            elif df["Finition"][i] == "Societe":
                Finition_Intens.append(0)
                Finition_Business.append(0)
                Finition_Limited.append(0)
                Finition_Zen.append(0)
                Finition_Societe.append(1)
                Autre_finition.append(0)
            else:
                Finition_Intens.append(0)
                Finition_Business.append(0)
                Finition_Limited.append(0)
                Finition_Zen.append(0)
                Finition_Societe.append(0)
                Autre_finition.append(1)
        else:
            Finition_Intens.append(nan)
            Finition_Business.append(nan)
            Finition_Limited.append(nan)
            Finition_Zen.append(nan)
            Finition_Societe.append(nan)
            Autre_finition.append(nan)
    return(Finition_Intens, Finition_Business, Finition_Limited, Finition_Zen, Finition_Societe, Autre_finition)

Finition_Intens, Finition_Business, Finition_Limited, Finition_Zen, Finition_Societe, Autre_finition = Recuperer_modalite_fintion()

df["finition_intens"] = pd.Series(Finition_Intens, index = df.index)
df["finition_business"] = pd.Series(Finition_Business, index = df.index)
df["finition_limited"] = pd.Series(Finition_Limited, index = df.index)
df["finition_zen"] = pd.Series(Finition_Zen, index = df.index)
df["finition_societe"] = pd.Series(Finition_Societe, index = df.index)
df["autre_finition"] = pd.Series(Autre_finition, index = df.index)

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
df["Horsepower_version"].isna().value_counts() #Il y a 897 valeurs manquantes.

def Fill_horsepower():
    n = len(df)
    list_of_missing_horsepower = df["horsepower"].isna()
    list_of_missing_Horsepower_version = df["Horsepower_version"].isna()
    for i in range(n):
        if list_of_missing_horsepower[i] == True and list_of_missing_Horsepower_version[i] == False:
            df["horsepower"][i] = df["Horsepower_version"][i]
    return("Terminé")

Fill_horsepower()

df["horsepower"].isna().value_counts() #Il reste 99 valeurs manquantes.


#Ajout des variables de version au data frame.
df.to_csv(".aws/automobile.csv")

df.to_csv(".aws/automobile2.csv")