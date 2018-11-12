from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

import numpy as np
import csv

#fixar a semente pra sempre obter os mesmos resultados
#N sei se funciona perfeitamente com a train_test_split
np.random.seed(7) 

from tensorflow.python.client import device_lib
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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
#X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.2, stratify=Y_train)

########################### TESTE ################################

model = Sequential()
qnt_entradas = len(labels)
model.add(Dense(128, input_dim=qnt_entradas, init='uniform', activation='relu'))

model.add(Dense(128, init='uniform', activation='relu'))
model.add(Dense(128, init='uniform', activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #adam = descida do gradiente
model.fit(X_train, Y_train, nb_epoch=50, batch_size=10)
scores = model.evaluate(X_test, Y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print(qnt_entradas)
#train_test_split()
#print(labels)