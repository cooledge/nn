from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
import pdb

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def fixy_shape(data):
  return data.reshape( data.shape[0], 784 )

x_train = fixy_shape(x_train)
y_train = to_categorical(y_train)
x_test = fixy_shape(x_test)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Dense(32, input_dim=28*28))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=32)

scores = model.evaluate(x_test, y_test)
print("Large CNN error: %.2f%%" % (100-scores[1]*100))

