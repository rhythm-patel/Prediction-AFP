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
from sklearn.model_selection import validation_curve

s = {'A':1,'R':2,'N':3,'D':4,'C':5,'E':6,'Q':7,'G':8,'H':9,'I':10,'L':11,'K':12,'M':13,'F':14,'P':15,'S':16,'T':17,'W':18,'Y':19,'V':20}
s2 = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'E': 6, 'Q': 7, 'G': 8, 'H': 9, 'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20, 'AA': 21, 'AR': 22, 'AN': 23, 'AD': 24, 'AC': 25, 'AE': 26, 'AQ': 27, 'AG': 28, 'AH': 29, 'AI': 30, 'AL': 31, 'AK': 32, 'AM': 33, 'AF': 34, 'AP': 35, 'AS': 36, 'AT': 37, 'AW': 38, 'AY': 39, 'AV': 40, 'RA': 41, 'RR': 42, 'RN': 43, 'RD': 44, 'RC': 45, 'RE': 46, 'RQ': 47, 'RG': 48, 'RH': 49, 'RI': 50, 'RL': 51, 'RK': 52, 'RM': 53, 'RF': 54, 'RP': 55, 'RS': 56, 'RT': 57, 'RW': 58, 'RY': 59, 'RV': 60, 'NA': 61, 'NR': 62, 'NN': 63, 'ND': 64, 'NC': 65, 'NE': 66, 'NQ': 67, 'NG': 68, 'NH': 69, 'NI': 70, 'NL': 71, 'NK': 72, 'NM': 73, 'NF': 74, 'NP': 75, 'NS': 76, 'NT': 77, 'NW': 78, 'NY': 79, 'NV': 80, 'DA': 81, 'DR': 82, 'DN': 83, 'DD': 84, 'DC': 85, 'DE': 86, 'DQ': 87, 'DG': 88, 'DH': 89, 'DI': 90, 'DL': 91, 'DK': 92, 'DM': 93, 'DF': 94, 'DP': 95, 'DS': 96, 'DT': 97, 'DW': 98, 'DY': 99, 'DV': 100, 'CA': 101, 'CR': 102, 'CN': 103, 'CD': 104, 'CC': 105, 'CE': 106, 'CQ': 107, 'CG': 108, 'CH': 109, 'CI': 110, 'CL': 111, 'CK': 112, 'CM': 113, 'CF': 114, 'CP': 115, 'CS': 116, 'CT': 117, 'CW': 118, 'CY': 119, 'CV': 120, 'EA': 121, 'ER': 122, 'EN': 123, 'ED': 124, 'EC': 125, 'EE': 126, 'EQ': 127, 'EG': 128, 'EH': 129, 'EI': 130, 'EL': 131, 'EK': 132, 'EM': 133, 'EF': 134, 'EP': 135, 'ES': 136, 'ET': 137, 'EW': 138, 'EY': 139, 'EV': 140, 'QA': 141, 'QR': 142, 'QN': 143, 'QD': 144, 'QC': 145, 'QE': 146, 'QQ': 147, 'QG': 148, 'QH': 149, 'QI': 150, 'QL': 151, 'QK': 152, 'QM': 153, 'QF': 154, 'QP': 155, 'QS': 156, 'QT': 157, 'QW': 158, 'QY': 159, 'QV': 160, 'GA': 161, 'GR': 162, 'GN': 163, 'GD': 164, 'GC': 165, 'GE': 166, 'GQ': 167, 'GG': 168, 'GH': 169, 'GI': 170, 'GL': 171, 'GK': 172, 'GM': 173, 'GF': 174, 'GP': 175, 'GS': 176, 'GT': 177, 'GW': 178, 'GY': 179, 'GV': 180, 'HA': 181, 'HR': 182, 'HN': 183, 'HD': 184, 'HC': 185, 'HE': 186, 'HQ': 187, 'HG': 188, 'HH': 189, 'HI': 190, 'HL': 191, 'HK': 192, 'HM': 193, 'HF': 194, 'HP': 195, 'HS': 196, 'HT': 197, 'HW': 198, 'HY': 199, 'HV': 200, 'IA': 201, 'IR': 202, 'IN': 203, 'ID': 204, 'IC': 205, 'IE': 206, 'IQ': 207, 'IG': 208, 'IH': 209, 'II': 210, 'IL': 211, 'IK': 212, 'IM': 213, 'IF': 214, 'IP': 215, 'IS': 216, 'IT': 217, 'IW': 218, 'IY': 219, 'IV': 220, 'LA': 221, 'LR': 222, 'LN': 223, 'LD': 224, 'LC': 225, 'LE': 226, 'LQ': 227, 'LG': 228, 'LH': 229, 'LI': 230, 'LL': 231, 'LK': 232, 'LM': 233, 'LF': 234, 'LP': 235, 'LS': 236, 'LT': 237, 'LW': 238, 'LY': 239, 'LV': 240, 'KA': 241, 'KR': 242, 'KN': 243, 'KD': 244, 'KC': 245, 'KE': 246, 'KQ': 247, 'KG': 248, 'KH': 249, 'KI': 250, 'KL': 251, 'KK': 252, 'KM': 253, 'KF': 254, 'KP': 255, 'KS': 256, 'KT': 257, 'KW': 258, 'KY': 259, 'KV': 260, 'MA': 261, 'MR': 262, 'MN': 263, 'MD': 264, 'MC': 265, 'ME': 266, 'MQ': 267, 'MG': 268, 'MH': 269, 'MI': 270, 'ML': 271, 'MK': 272, 'MM': 273, 'MF': 274, 'MP': 275, 'MS': 276, 'MT': 277, 'MW': 278, 'MY': 279, 'MV': 280, 'FA': 281, 'FR': 282, 'FN': 283, 'FD': 284, 'FC': 285, 'FE': 286, 'FQ': 287, 'FG': 288, 'FH': 289, 'FI': 290, 'FL': 291, 'FK': 292, 'FM': 293, 'FF': 294, 'FP': 295, 'FS': 296, 'FT': 297, 'FW': 298, 'FY': 299, 'FV': 300, 'PA': 301, 'PR': 302, 'PN': 303, 'PD': 304, 'PC': 305, 'PE': 306, 'PQ': 307, 'PG': 308, 'PH': 309, 'PI': 310, 'PL': 311, 'PK': 312, 'PM': 313, 'PF': 314, 'PP': 315, 'PS': 316, 'PT': 317, 'PW': 318, 'PY': 319, 'PV': 320, 'SA': 321, 'SR': 322, 'SN': 323, 'SD': 324, 'SC': 325, 'SE': 326, 'SQ': 327, 'SG': 328, 'SH': 329, 'SI': 330, 'SL': 331, 'SK': 332, 'SM': 333, 'SF': 334, 'SP': 335, 'SS': 336, 'ST': 337, 'SW': 338, 'SY': 339, 'SV': 340, 'TA': 341, 'TR': 342, 'TN': 343, 'TD': 344, 'TC': 345, 'TE': 346, 'TQ': 347, 'TG': 348, 'TH': 349, 'TI': 350, 'TL': 351, 'TK': 352, 'TM': 353, 'TF': 354, 'TP': 355, 'TS': 356, 'TT': 357, 'TW': 358, 'TY': 359, 'TV': 360, 'WA': 361, 'WR': 362, 'WN': 363, 'WD': 364, 'WC': 365, 'WE': 366, 'WQ': 367, 'WG': 368, 'WH': 369, 'WI': 370, 'WL': 371, 'WK': 372, 'WM': 373, 'WF': 374, 'WP': 375, 'WS': 376, 'WT': 377, 'WW': 378, 'WY': 379, 'WV': 380, 'YA': 381, 'YR': 382, 'YN': 383, 'YD': 384, 'YC': 385, 'YE': 386, 'YQ': 387, 'YG': 388, 'YH': 389, 'YI': 390, 'YL': 391, 'YK': 392, 'YM': 393, 'YF': 394, 'YP': 395, 'YS': 396, 'YT': 397, 'YW': 398, 'YY': 399, 'YV': 400, 'VA': 401, 'VR': 402, 'VN': 403, 'VD': 404, 'VC': 405, 'VE': 406, 'VQ': 407, 'VG': 408, 'VH': 409, 'VI': 410, 'VL': 411, 'VK': 412, 'VM': 413, 'VF': 414, 'VP': 415, 'VS': 416, 'VT': 417, 'VW': 418, 'VY': 419, 'VV': 420}
counter = 420
for i in s:
	for j in s:
		for k in s:
			counter += 1
			s2[i+j+k] = counter


def findPercentageInSeries(ser):

	percentages=[]

	for row in ser:
		temp = [0]*8420
		lenRow = len(row)

		# find frequency
		for letter in range(len(row)):
			index = s2[row[letter]] - 1
			temp[index] += 1
			if(letter < len(row)-1):
				index = s2[row[letter:letter+2]] - 1
				temp[index] += 1
			if(letter < len(row) - 2):
				index = s2[row[letter:letter+3]] -  1
				temp[index] += 1			

		# find percentage
		for i in range(8420):
			temp[i] = temp[i]/lenRow

		percentages.append(temp)

	return percentages

def fitModel(X_train,y_train,gamma):
	# model = SVC(gamma = gamma,kernel = "rbf") # our ML model
	# model.fit(X_train,y_train) # fit the model by x & y of train
	# return model

	clf = RandomForestClassifier(n_estimators = gamma)
	clf.fit(X_train,y_train)
	return clf

	# clf = tree.DecisionTreeClassifier()
	# clf = clf.fit(X_train, y_train)
	# return clf

	# clf = tree.DecisionTreeRegressor()
	# clf = clf.fit(X_train, y_train)
	# return clf

def optimalParameters():

    trainDataset = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    x_train_peptide = trainDataset.iloc[:,2] # peptide sequence of train dataset
    
    X = findPercentageInSeries(x_train_peptide)
    y = trainDataset.iloc[:,1] # 1/-1 of train dataset


    # Create range of values for parameter
    param_range = np.arange(1, 4001, 400)

    # Calculate accuracy on training and test set using range of parameter values
    train_scores, test_scores = validation_curve(RandomForestClassifier(), 
                                                X, 
                                                y, 
                                                param_name="n_estimators", # add parameter to test here e.g. max_depth, n_estimators, min_sample_split, etc.
                                                param_range=param_range,
                                                cv=3, 
                                                scoring="accuracy", 
                                                )

    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot mean accuracy scores for training and test sets
    plt.plot(param_range, train_mean, label="Training score", color="navy")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="red")

    # Plot accurancy bands for training and test sets
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="lightskyblue")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="mistyrose")

    # Create plot
    plt.title("Validation Curve With Random Forest")
    plt.xlabel("Number Of Trees")
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show()

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

	print("Output in submission.csv")

def findOptimalGamma(start,end):
	val = []
	gammaArr = []

	for gamma in range(start,end,10):

		print ("Gamma: ",gamma)
		gammaArr.append(gamma)

		# predict(gamma)
		temp = findAccuracy(gamma)
		val.append(temp)

	train = []
	test = []
	for i in val:
		train.append(i[0])
		test.append(i[1])

	plt.plot(gammaArr,train,gammaArr,test)
	plt.show()

gamma = 700

predict(gamma)
x = findAccuracy(gamma)
# optimalParameters()