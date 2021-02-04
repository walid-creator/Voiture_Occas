import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy
import scipy
import sklearn
import matplotlib
#import nltk
# #probleme cet import
#df = pd.read_csv("s3://projet-stat-ensai/Lacentrale.csv",sep=';', dtype=str)

#hellow world !

df = pd.read_csv('/Users/famille//projetstat/.aws/automobile.csv',error_bad_lines=False)
print(df.head(10))
#print(df.iloc[:,15:21].head(10))
#u'reference', u'url', u'user_id': variables d'indentification,
# u'type', u'marque', u'modele' : n'ont qu'une seule modalite
#variable quali: u'modele_com', u'departement'(a discretiser), u'couleur',u'energie', u'porte'(nbr portes)
#variable quanti: u'horsepower', puissance_fiscale,u'kilometrage', 'prix_vente'
#format date: u'date_mec'
#version et options: faire du text mining
#print(df.shape)
#d= df[[u'date_mec', u'date_depublication']]

#d= df[[u'modele_com', u'date_mec', u'horsepower', u'engine', u'kilometrage',
 #      u'energie', u'boite_de_vitesse', u'porte', u'options', u'couleur',
  #     u'premiere_main', u'version', u'departement', u'puissance_fiscale',
   #    u'date_publication', u'date_depublication', u'prix_vente']]



#print(df.describe())
#################description des variables:##########################
#type:PRO
#marque: RENAULT
#modele: CLIO
#date_mec: la date de mise en circulation
#horsepower: nombre de chevaux d'une voiture, 1 Ch = 0,736 kW
#engine: La puissance du moteur en kilowatt
#kilometrage: Le nombre de kilometres parcouru par la voiture
#energie: essence ou diesel
#boite_de_vitesse: automatique ou manuel
#porte: le nombre de portes
#options: GPS,
#couleur: ...
#premiere_main:...
#version:
#departement: ...
#puissance_fiscale: la puissance maximale du moteur en terme de kilowatt/heure,
#date_publication: ...
#date_depublication: ...
#prix_vente: en euro
######################traitement###########################################

#suppression des variables d'id:
#df.drop([u'reference', u'url', u'user_id'],axis=1,inplace=True)
'''#print(df)
#print(df)
#transformation de la variable date en anciennete et suppression des variables,date_publication
#x=df[u'date_mec'][0][0]+ df[u'date_mec'][0][1] + df[u'date_mec'][0][2]+ df[u'date_mec'][0][3]
#y=df[u'date_mec'][1][0]+ df[u'date_mec'][1][1] + df[u'date_mec'][1][2]+ df[u'date_mec'][1][3]
#d= df[[u'date_mec', u'date_depublication']]
#print(d)



AnneeMiseEnVente=[]
AnneeDepublication=[]
MoisMiseEnVente=[]
MoisDepublication=[]
JourMiseEnVente=[]
JourDepublication=[]

for i in range(df.shape[0]):
    AnneeMiseEnVente.append(df[u'date_mec'][i][0]+ df[u'date_mec'][i][1] + df[u'date_mec'][i][2]+ df[u'date_mec'][i][3])
    AnneeDepublication.append(df[u'date_depublication'][i][0]+ df[u'date_depublication'][i][1] + df[u'date_depublication'][i][2]+ df[u'date_depublication'][i][3])
    if df[u'date_mec'][i][6]=='-':
        MoisMiseEnVente.append(df[u'date_mec'][i][5])
        JourMiseEnVente.append(df[u'date_mec'][i][7] + df[u'date_mec'][i][8])
    else:
        MoisMiseEnVente.append(df[u'date_mec'][i][5]+ df[u'date_mec'][i][6])
        JourMiseEnVente.append(df[u'date_mec'][i][8] + df[u'date_mec'][i][9])
    MoisDepublication.append(df[u'date_depublication'][i][5] + df[u'date_depublication'][i][6])
    JourDepublication.append(df[u'date_depublication'][i][8] + df[u'date_depublication'][i][9])
#transformation en des entiers
AnneeMiseEnVente=pd.to_numeric(AnneeMiseEnVente, downcast='integer')
AnneeDepublication=pd.to_numeric(AnneeDepublication, downcast='integer')
MoisMiseEnVente=pd.to_numeric(MoisMiseEnVente, downcast='integer')
MoisDepublication=pd.to_numeric(MoisDepublication,downcast='integer')
JourMiseEnVente=pd.to_numeric(JourMiseEnVente, downcast='integer')
JourDepublication=pd.to_numeric(JourDepublication, downcast='integer')


df['AnneeMiseEnVente']=AnneeMiseEnVente
df['AnneeDepublication']=AnneeDepublication
df['MoisMiseEnVente']= MoisMiseEnVente
df['MoisDepublication']= MoisDepublication
df['JourMiseEnVente']= JourMiseEnVente
df['JourDepublication']= JourDepublication
AgeAnnee=[]
AgeMois=[]
AgeJour=[]
for i in range(df.shape[0]):
    if df['MoisDepublication'][i]-df['MoisMiseEnVente'][i]>0:
        AgeAnnee.append(df['AnneeDepublication'][i]-df['AnneeMiseEnVente'][i])
        if df['JourDepublication'][i]-df['JourMiseEnVente'][i]>=0:
            AgeMois.append(df['MoisDepublication'][i]-df['MoisMiseEnVente'][i])
            AgeJour.append(df['JourDepublication'][i]-df['JourMiseEnVente'][i])
        else:
            AgeMois.append(df['MoisDepublication'][i] - df['MoisMiseEnVente'][i]-1)
            AgeJour.append(30+df['JourDepublication'][i]-df['JourMiseEnVente'][i])

    elif df['MoisDepublication'][i]-df['MoisMiseEnVente'][i]==0:
        if df['JourDepublication'][i]-df['JourMiseEnVente'][i]>=0:
            AgeAnnee.append(df['AnneeDepublication'][i] - df['AnneeMiseEnVente'][i])
            AgeMois.append(0)
            AgeJour.append(df['JourDepublication'][i]-df['JourMiseEnVente'][i])
        else:
            AgeAnneeappend(df['AnneeDepublication'][i] - df['AnneeMiseEnVente'][i]-1)
            AgeMois.append(11)
            AgeJour.append(df['JourDepublication'][i]-df['JourMiseEnVente'][i]+30)

    else:
       AgeAnnee.append(df['AnneeDepublication'][i] - df['AnneeMiseEnVente'][i]- 1)
       if df['JourDepublication'][i]-df['JourMiseEnVente'][i]>=0:
           AgeMois.append( df['MoisDepublication'][i] - df['MoisMiseEnVente'][i]+12)
           AgeJour.append( df['JourDepublication'][i]-df['JourMiseEnVente'][i])
       else:
           AgeMois.append( df['MoisDepublication'][i] - df['MoisMiseEnVente'][i]+11)
           AgeJour.append( df['JourDepublication'][i] - df['JourMiseEnVente'][i]+30)
df['AgeJour']=AgeJour
df['AgeMois']=AgeMois
df['AgeAnnee']=AgeAnnee
'''
#supression des variables avec une seule modalite et des variables de dates
#df.drop([u'type', u'marque',u'modele'], axis=1, inplace=True)
##df.drop([u'date_mec', u'date_depublication','AnneeMiseEnVente', 'AnneeDepublication','MoisMiseEnVente','MoisDepublication','JourDepublication','JourMiseEnVente'], axis=1, inplace=True)

#la supression prend un temps enorme et le nombre de calcul..


#regarder les variables pertinentes qui peuvent etre relies au prix et faire les stats du chi2 et des boxplots
#comme kilometrage et boite a vitesse, energie , horsepower et age
#regarder comment imputer les valeurs manquantes
#Detection des valeurs manquantes:

#df[u'horsepower']=pd.to_numeric(df[u'horsepower'], downcast='integer')

#sans:u'prix_vente',u'horsepower',u'departement'(a discretiser),u'energie', u'kilometrage'
#avec: u'couleur'(Autre/non affecte,N/a,RQH, OR) (1,1,1),u'puissance_fiscale'(NaN)(3),u'porte'(NaN)(4),u'horsepower'(NaN)(750)

#uniformiser la variable couleur:
#



'''
for i in range(df.shape[0]):
    if type(df[u'couleur'][i])==str:
        df[u'couleur'][i]=df[u'couleur'][i].lower()
#print(df.loc[df[u'couleur']=="blnache",:])
#df.iloc[44695,15]="blanc"
df[u"couleur"].replace("blnache", "blanc",inplace= True)
df[u"couleur"].replace("noiir", "noir",inplace= True)
df[u"couleur"].replace("blanche", "blanc",inplace= True)
df[u"couleur"].replace("noire", "noir",inplace= True)
df[u"couleur"].replace("grise", "gris",inplace= True)
df[u"couleur"].replace("roue flamme", "rouge flamme",inplace= True)
#imputation par le mode
for j in range(df.shape[0]):
    if type(df[u'couleur'][j])!=str or df[u'couleur'][j]=="non renseigne" or df[u'couleur'][j][0:2]=="n1":
        df[u'couleur'][j]="gris"

'''
'''
#print(df.loc[df[u'couleur'][0:5]=="blanc",:])
MissingData=df[u'couleur'].value_counts(dropna=False)
print(MissingData)


#reste a regrader s'il y'a des valeurs dans les couleurs dans le couleurs sinon tout uniformiser et creer un git pour que ca soit plus facile
print(df.shape[1])
#print(df.iloc[44695,22])
print(df['prix_vente'].astype('float').mean(axis=0))
print(df['prix_vente'].astype('float').max(axis=0))
print(df['prix_vente'].astype('float').min(axis=0))
print(df['prix_vente'].astype('float').median(axis=0))
print(df['prix_vente'].astype('float').std(axis=0))# ecrart type au sens statistique n-1 et saute eventuellement les na
#la moyenne et la mediane sont proches
#un ecrat type trop grand de 55622
print(df['prix_vente'].describe())


#
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#import numpy as np
#df[u'prix_vente']=pd.to_numeric(df[u'prix_vente'], downcast='integer')
#bins = np.linspace(min(df[u'prix_vente']), max(df[u"prix_vente"]), 4)
#group_names = ['Low', 'Medium', 'High']
#df[u'prix_vente'] = pd.cut(df[u'prix_vente'], bins, labels=group_names, include_lowest=True )

#plt.bar(group_names, df[u'prix_vente'].value_counts())

# set x/y labels and plot title
#plt.xlabel(u"prix_vente")
#plt.ylabel(u"count")
#plt.title(u"prix_vente")
x=[1,2,3,4,5]
y=[1,2,3,4,5]
plt.plot(x,y)


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

#print(df.head(10))
#df.to_csv("automobile.csv", index=False)

print(df[[u"kilometrage", u"prix_vente"]].corr())#il est bien negative mais faible 0.051
print(df[[u"puissance_fiscale", u"prix_vente"]].corr())#0.022
print(df[[u"horsepower", u"prix_vente"]].corr())#0.035
print(df[[u"AgeAnnee", u"prix_vente"]].corr())# 0.048
from scipy import stats
#pearson_coef, p_value = stats.pearsonr(df[u"AgeAnnee"], df[u"prix_vente"])
#print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
#print(df['couleur'].value_counts().to_frame())
sns.boxplot(x=u"premiere_main", y=u"prix_vente", data=df)
plt.show()

#df.drop([u'reference', u'url', u'user_id'],axis=0,inplace=True)
'''