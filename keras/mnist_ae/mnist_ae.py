from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential
from keras.regularizers import l1, l2
import pdb

encoding_dim = 32

def get_autoencoder_original():
  input_img = Input(shape=(784,))
  encoded = Dense(encoding_dim, activation='relu')(input_img)
  decoded = Dense(784, activation='sigmoid')(encoded)
  return Model(input_img, decoded)

def get_autoencoder_fully_connected():
  model = Sequential()
  # encoder
  model.add(Dense(encoding_dim, activation='relu', input_shape=(784,)))
  # decoder
  model.add(Dense(784, activation='sigmoid'))
  return model

def get_autoencoder_deep_fc():
  model = Sequential()

  # Encoder
  model.add(Dense(128, activation='relu', input_shape=(784,)))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(32, activation='relu'))

  model.add(Dense(64, activation='relu'))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(784, activation='sigmoid'))
  return model

def get_autoencoder_conv():
  model = Sequential()

  pdb.set_trace()
  model.add(Conv2D(16, (3,3), activation='relu', padding='same', input_shape=(28,28,1)))
  model.add(MaxPooling2D(2,2, padding='same'))
  model.add(Conv2D(8, (3,3), activation='relu', padding='same'))
  model.add(MaxPooling2D(2,2, padding='same'))
  model.add(Conv2D(8, (3,3), activation='relu', padding='same'))
  model.add(MaxPooling2D(2,2, padding='same'))
  
  model.add(Conv2D(8, (3,3), activation='relu', padding='same'))
  model.add(UpSampling2D((2,2)))
  model.add(Conv2D(8, (3,3), activation='relu', padding='same'))
  model.add(UpSampling2D((2,2)))
  model.add(Conv2D(16, (3,3), activation='relu'))
  model.add(UpSampling2D((2,2)))
  model.add(Conv2D(1, (3,3), activation='sigmoid', padding='same'))
   
  return model 

autoencoder = get_autoencoder_conv()
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

pdb.set_trace()
shape = autoencoder.output.shape.as_list()[1:]
x_train = np.reshape(x_train, tuple([len(x_train)]+shape))
x_test = np.reshape(x_test, tuple([len(x_test)] + shape))

#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# encode and decode some digits
# note that we take them from the *test* set
#encoded_imgs = encoder.predict(x_test)
#decoded_imgs = decoder.predict(encoded_imgs)
decoded_imgs = autoencoder.predict(x_test)

# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

