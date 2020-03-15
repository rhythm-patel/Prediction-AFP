import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Bidirectional

s = {'A':1,'R':2,'N':3,'D':4,'C':5,'E':6,'Q':7,'G':8,'H':9,'I':10,'L':11,'K':12,'M':13,'F':14,'P':15,'S':16,'T':17,'W':18,'Y':19,'V':20}

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

x_train_peptide = train.iloc[:,2] # peptide sequence of train dataset
y_train = train.iloc[:,1] # 1/-1 of train dataset
x_test_peptide = test.iloc[:,1] # peptide sequence of test dataset
IDs = test.iloc[:,0] # IDs from the test dataset

x_train = []
for peptide in x_train_peptide:
    vector = []
    for letter in peptide:
        vector.append(s[letter])
    x_train.append(vector)

x_test = []
for peptide in x_test_peptide:
    vector = []
    for letter in peptide:
        vector.append(s[letter])
    x_test.append(vector)

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
y_train = np.asarray(y_train)
for i in range(len(y_train)):
    if y_train[i]==-1:
        y_train[i] = 0

max_words = max([len(x) for x in x_train])

max_sequence_length = min(average_length, median_length)
x_train = sequence.pad_sequences(x_train, maxlen=max_words, padding='post', truncating='post')
x_test = sequence.pad_sequences(x_test, maxlen=max_words, padding='post', truncating='post')

hidden_size = 32

model = Sequential()
model.add(Embedding(max_words, hidden_size))
model.add(Bidirectional(LSTM(hidden_size, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(hidden_size, activation='tanh', dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 10

model.fit(x_train, y_train, epochs=epochs, shuffle=True)
y_test = model.predict(x_train)
y_actual_test = []
corr, tot = 0, 0
for i in range(len(y_test)):
    tot+=1
    y_actual_test.append(round(y_test[i][0]))
    if y_actual_test[i]==y_train[i]:
        corr+=1
print(corr*1.0/tot)