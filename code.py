import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

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

def fitModel(X_train,y_train,gamma):
    model = SVC(gamma=gamma, kernel = "rbf") # our ML model
    model.fit(X_train,y_train) # fit the model by x & y of train
    return model

    # clf = RandomForestClassifier(max_depth=100, random_state=0)
    # clf.fit(X_train,y_train)
    # return clf

    # clf = tree.DecisionTreeClassifier()
    # clf = clf.fit(X_train, y_train)
    # return clf

    # clf = tree.DecisionTreeRegressor()
    # clf = clf.fit(X_train, y_train)
    # return clf



def findAccuracy(gamma):

    trainDataset = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    x_train_peptide = trainDataset.iloc[:,2] # peptide sequence of train dataset
    
    X = findPercentageInSeries(x_train_peptide)
    y = trainDataset.iloc[:,1] # 1/-1 of train dataset

    noOfSplits = 5

    kf = KFold(n_splits=noOfSplits,random_state=False,shuffle=False)
    trainAcc = []
    testAcc = []

    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X = np.asarray(X) # convert list to numpy array
        y = np.asarray(y) # convert list to numpy array
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = fitModel(X_train,y_train,gamma)

        trainAccuracy = model.score(X_train,y_train) # first predicts X_train to y_train' & then compares with y_train
        testAccuracy = model.score(X_test,y_test)  # first predicts X_test to y_test' & then compares with y_test

        trainAcc.append(trainAccuracy)
        testAcc.append(testAccuracy)
        # rmse = sqrt(mean_squared_error(y_test, y_pred))
        # print (rmse)

    avgTrainAccuracy = sum(trainAcc)/noOfSplits
    avgTestAccuracy = sum(testAcc)/noOfSplits
    print ("Avg Train Accuracy: ",avgTrainAccuracy)
    print ("Avg Test Accuracy: ",avgTestAccuracy)
    return (avgTrainAccuracy,avgTestAccuracy)

def predict(gamma):

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    x_train_peptide = train.iloc[:,2] # peptide sequence of train dataset
    y_train = train.iloc[:,1] # 1/-1 of train dataset
    x_test_peptide = test.iloc[:,1] # peptide sequence of test dataset
    IDs = test.iloc[:,0] # IDs from the test dataset

    X_train = findPercentageInSeries(x_train_peptide)
    X_test = findPercentageInSeries(x_test_peptide)

    y_train = np.asarray(y_train) # convert list to numpy array
    # y_train = np.ravel(y_train) # reshape 2D array to 1D array

    model = fitModel(X_train,y_train,gamma)
    y_test = model.predict(X_test) # predict the y of test based on our model

    output = [["ID","Label"]]
    for i in range(len(y_test)):
        temp = []
        temp.append(IDs[i]) #adds the IDs
        temp.append(int(y_test[i])) #adds the predicted y values i.e 1/-1
        output.append(temp)

    with open('submission.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(output)

def findOptimalGamma():
    val = []
    gammaArr = []

    for gamma in range(10,150):

        print ("Gamma: ",gamma)
        gammaArr.append(gamma)

        predict(gamma)
        temp = findAccuracy(gamma)
        val.append(temp)

    train = []
    test = []
    for i in val:
        train.append(i[0])
        test.append(i[1])

    plt.plot(gammaArr,train,gammaArr,test)
    plt.show()

gamma = 200

predict(gamma)
x = findAccuracy(gamma)
# findOptimalGamma()