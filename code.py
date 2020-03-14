import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

s = {'A':1,'R':2,'N':3,'D':4,'C':5,'E':6,'Q':7,'G':8,'H':9,'I':10,'L':11,'K':12,'M':13,'F':14,'P':15,'S':16,'T':17,'W':18,'Y':19,'V':20}

def findPercentageInSeries(ser):

    percentages=[]

    for row in ser:
        temp = [0]*20
        lenRow = len(row)

        # find frequency
        for letter in row:
            index = s[letter] - 1
            temp[index] += 1

        # find percentage
        for i in range(20):
            temp[i] = temp[i]/lenRow

        percentages.append(temp)

    return percentages


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

x_train_peptide = train.iloc[:,2] # peptide sequence of train dataset
y_train = train.iloc[:,1] # 1/-1 of train dataset
x_test_peptide = test.iloc[:,1] # peptide sequence of test dataset
IDs = test.iloc[:,0] # IDs from the test dataset

x_train = findPercentageInSeries(x_train_peptide)
x_test = findPercentageInSeries(x_test_peptide)

# np.savetxt('output.csv',listOfPercentages,delimiter=",")
# print(listOfPercentages)

# train = pd.read_csv("output.csv")

# x = train.iloc[:,20]
# for rows in x:
#     print(rows)

model = SVC(gamma=30, kernel = "rbf") # or ML model

model.fit(x_train,np.ravel(y_train)) # fit the model by x & y of train
y_test = model.predict(x_test) # predict the y of test based on our model

# count = 0
# for rows in range(len(x)):
#     if y_test[rows] == x[rows]:
#         count += 1

# print(count/len(x))


# print(len(out))
# print(len(y_test))

output = [["ID","Label"]]
for i in range(len(y_test)):
    temp = []
    temp.append(IDs[i]) #adds the IDs
    temp.append(y_test[i]) #adds the predicted y values i.e 1/-1
    output.append(temp)

with open('submission.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(output)