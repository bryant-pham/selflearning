'''
Trains a simple NN using activation weights from the previously trained
autoencoder to create a new feature representation.
'''

from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.datasets import mnist
from keras.optimizers import RMSprop
import numpy as np
import keras

autoencoder = load_model('autoencoder.h5')
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Let last 20k examples be "labeled data" and train model on it
x_train_unlabeled = x_train[:40000]
x_train_labeled = x_train[40000:]
y_train_labeled = y_train[40000:]

model = Sequential()

# Let first layer be the autoencoder's first layer activation weights
model.add(Dense(32, input_shape=(784,), weights=autoencoder.layers[1].get_weights(), activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

# Let last layer be classifier
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
history = model.fit(x_train_labeled, y_train_labeled, batch_size=256, epochs=100, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
