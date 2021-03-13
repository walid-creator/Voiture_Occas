#découpage en classe des variables quantitatives:
'''
Quanti=df[[u'horsepower',u'engine',"age"]]
a= (max(df["age"])-min(df["age"]))/3
s1= min(df["age"])+a #10 ANS
s2=s1+a#20ans
print(s1, s2)
s3=730
s4=1825

def intersection_2_listes(L1,L2):
    L=[]
    for i in range(len(L1)):
        if L1[i] in L2:
            L.append(L1[i])
    return L
indexe1= Quanti[ Quanti["age"]<s3]
indexe21= Quanti[ s1<=Quanti["age"]]
indexe22=Quanti[Quanti["age"]<s4]
indexe3= Quanti[ Quanti["age"]>=s4]
indexe1=list(indexe1)
indexe21=list(indexe21)
indexe22=list(indexe22)
indexe2= intersection_2_listes(indexe21,indexe22)
indexe3=list(indexe3)
print(indexe3)
for i in indexe1:
    Quanti["age"][i]="presque_neuve"
for i in indexe2:
    Quanti["age"][i]="ancienne"
    for i in indexe3:
        Quanti["age"][i] = "très_ancienne"
'''