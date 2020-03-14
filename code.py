import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

dataset = pd.read_csv("train.csv")
testset = pd.read_csv("test.csv")
sample = pd.read_csv("sample.csv")

x = dataset.iloc[:,1]
y = dataset.iloc[:,2]
testSet = testset.iloc[:,1]
sampleset = sample.iloc[:,0]

s = {'A':0,'R':1,'N':2,'D':3,'C':4,'E':5,'Q':6,'G':7,'H':8,'I':9,'L':10,'K':11,'M':12,'F':13,'P':14,'S':15,'T':16,'W':17,'Y':18,'V':19}

listOfPercentages = list()
listOfPercentagesOut = list()

for rows in y:
    ls = [0]*20
    for a in rows:
        index = s[a]
        ls[index] += 1
    for i in range(20):
        lenRows = len(rows)
        ls[i] =  ls[i]/lenRows
    listOfPercentages.append(ls)

# for l in range(len(x)):
#     listOfPercentages[l].append(x[l])

for rows in testSet:
    ls = [0]*20
    for a in rows:
        ls[s[a]-1] += 1
    for i in range(20):
        ls[i] = ls[i]/len(rows)
    listOfPercentagesOut.append(ls)



# np.savetxt('output.csv',listOfPercentages,delimiter=",")
# print(listOfPercentages)

# dataset = pd.read_csv("output.csv")

# x = dataset.iloc[:,20]
# for rows in x:
#     print(rows)

model = SVC(gamma=30, kernel = "rbf")

model.fit(listOfPercentages,np.ravel(x))
predictedOut = model.predict(listOfPercentagesOut)

# count = 0
# for rows in range(len(x)):
#     if predictedOut[rows] == x[rows]:
#         count += 1

# print(count/len(x))


# print(len(out))
print(len(predictedOut))
output = [["ID","Label"]]
for elements in range(len(predictedOut)):
    o = []
    o.append(sampleset[elements])
    o.append(predictedOut[elements])
    output.append(o)

with open('submission.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(output)



