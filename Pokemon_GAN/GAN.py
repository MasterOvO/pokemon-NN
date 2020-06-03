import numpy as np
from keras.layers import Dense, Conv2DTranspose, Reshape, BatchNormalization, LeakyReLU, Conv2D, Dropout, Flatten
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import os
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist


# Build model


def _generator():
    model = Sequential()
    model.add(Dense(7 * 7 * 256, use_bias = False, input_shape = (100, )))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)

    model.add(Conv2DTranspose(128, (5, 5) , strides = (1, 1), padding = "same", use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="same", use_bias=False))
    assert model.output_shape == (None, 14, 14, 32)
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(4, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation = "sigmoid"))
    assert model.output_shape == (None, 28, 28, 4)

    return model

gen = _generator()
noise = np.random.rand(1, 100)
img = gen.predict(noise)
print(img.shape)

def _discriminator():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), strides= (2,2), padding = "same", input_shape= (28, 28, 4)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(32, activation= "relu"))
    model.add(Dense(1, activation= "sigmoid"))
    model.trainable = True

    opt = Adam(lr=0.001, beta_1=0.5)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

img = tf.convert_to_tensor(img)
discri = _discriminator()
result = discri(img)
print(result)

def combined_model():
    model = Sequential()
    model.add(gen)
    model.add(discri)
    discri.trainable = False

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

com = combined_model()
com.summary()

def _train(x, epochs = 5000, batch = 128, save_interval = 200):
    for count in range(epochs):
        random_index = np.random.randint(0, len(x) - batch / 2)
        legit_images = x[random_index: random_index + int(batch / 2)].reshape(int(batch / 2), 28, 28, 4)

        noise_input = np.random.normal(0, 1, (int(batch / 2), 100))
        fake_images = gen.predict(noise_input)

        x_combined_batch = np.concatenate((legit_images, fake_images))
        y_combined_batch = np.concatenate((np.ones((int(batch / 2), 1)), np.zeros((int(batch / 2), 1))))

        d_loss = discri.train_on_batch(x_combined_batch, y_combined_batch)

        noise = np.random.normal(0, 1, (int(batch), 100))
        y_mislabled = np.ones((batch, 1))

        g_loss = com.train_on_batch(noise, y_mislabled)

        print('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (count, d_loss[0], g_loss[0]))

        if count % save_interval == 0:
            with open(f"Generated_img\\{count}.png", "wb") as f:
                noise1 = np.random.rand(1, 100)
                print("generated img")
                pic = gen.predict(noise1)[0,:,:,:]
                img1 = Image.fromarray((pic * 255).astype("uint8"), "RGBA")
                img1.save(f)
            with open(f"Generated_img\\{count}_demo.png", "wb") as f:
                img2 = Image.fromarray((x[random_index] *255).astype("uint8"), "RGBA")
                img2.save(f)

# Use gen1-gen3 pokemon
# Prepare our input
data = []

for pic in os.listdir("Data_GAN"):
    file = f"Data_GAN\\{pic}"
    pokm_img = Image.open(file)
    pokm_img = pokm_img.resize((28, 28))
    pokm_arr = np.array(pokm_img)
    pokm_arr = pokm_arr /255.0
    data.append(pokm_arr)

data = np.array(data)
_train(data)

"""
# this is the mnist training set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train/ 255.0
_train(x_train)
"""