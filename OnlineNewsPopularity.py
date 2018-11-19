import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
# from sklearn.model.sclection import train_test_split

# Generate dummy data

# x_traib, x_test, y_train, y_test  = train_test_split(x,y, test_size=0.2 , stratify = True)

x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))

model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)



equacao do quadro (05/11)
taxa = 0,1 => 5.0
td = 1-((taxa)/n) => 0,99995