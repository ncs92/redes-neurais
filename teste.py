from sklearn import svm
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

import numpy as np
import csv
from sklearn.metrics import classification_report, confusion_matrix
np.random.seed(7) #fixar a semente pra sempre obter os mesmos resultados
#N sei se funciona perfeitamente com a train_test_split
from tensorflow.python.client import device_lib
from sklearn.svm import SVC
reader = csv.reader(open('OnlineNewsPopularity.csv','r'), delimiter=',')

rows = np.array(list(reader))
labels = rows[1, 1:-1]
X = rows[1:-1, 1:-1] 
Ya = rows[1:-1, -1] #ultima coluna
#print(rows)
Y = []

index = 0

for y in Ya:
    if(int(y) > 1200):
        Y.insert(index, True)
        
    else:
        Y.insert(index, False)
    
    index += 1
#print(rows)
print("WOOOOO2")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)

svclassifier = SVC(kernel='linear', verbose=True)
print("WOOOOO2")

svclassifier.fit(X_train, Y_train)
print("WOOOOO")
y_pred = svclassifier.predict(X_test)

print(confusion_matrix(Y_test,y_pred))
print(classification_report(Y_test,y_pred))