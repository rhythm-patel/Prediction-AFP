import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

dataset = pd.read_csv("train.csv")
x = dataset.iloc[:,1]
y = dataset.iloc[:,2]

s = {'A':1,'R':2,'N':3,'D':4,'C':5,'E':6,'Q':7,'G':8,'H':9,'I':10,'L':11,'K':12,'M':13,'F':14,'P':15,'S':16,'T':17,'W':18,'Y':19,'V':20}

listOfPercentages = list()

for rows in y:
    ls = [0]*20
    for a in rows:
        ls[s[a]-1] += 1
    for i in range(20):
        ls[i] =  ls[i]*100/len(rows)
    listOfPercentages.append(ls)

for l in range(len(x)):
    listOfPercentages[l].append(x[l])

np.savetxt('output.csv',listOfPercentages,delimiter=",")
# print(listOfPercentages)

dataset = pd.read_csv("output.csv")

x = dataset.iloc[:,20]
for rows in x:
    print(rows)
