'''
Main script

Inspired by https://www.kaggle.com/shivamb/how-autoencoders-work-intro-and-usecases/
'''

from keras.layers import Dense, Input, Conv2D, LSTM, MaxPool2D, UpSampling2D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

import matplotlib.pyplot as plt
from keras.models import Model, Sequential
# from imgaug import augmenters
from random import randint
import pandas as pd
import numpy as np

import os, sys
from PIL import Image

path = "C:/Calle/Programmering/Python/Autoencoder/data/"
model_path = "C:/Calle/Programmering/Python/Autoencoder/models/"
fname = "fashion-mnist_train.csv"

def read_and_split_data(path, fname):
    # Read dataset
    df = pd.read_csv(path + "fashion-mnist_train.csv")
    train_x = df[list(df.columns)[1:]].values
    train_y = df["label"].values

    # Normalize predictors
    train_x = train_x / 255

    # create train/val datasets
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)

    # reshape inputs
    train_x = train_x.reshape(-1, 784)
    val_x = val_x.reshape(-1, 784)

    return train_x, val_x, train_y, val_y

def create_model(input_shape, layer_sizes, latent_size):
    # Create model using functional API
    model = Sequential()

    # add encoding architecture
    for i, layer_size in enumerate(layer_sizes):
        if i==0:
            model.add(Dense(layer_size, input_shape=(input_shape,)))
        else:
            model.add(Dense(layer_size, activation="relu"))

    # latent view
    model.add(Dense(latent_size, activation="sigmoid"))

    # add decoding architecture
    for layer_size in layer_sizes[::-1]:
        model.add(Dense(layer_size, activation="relu"))

    # add output layer
    model.add(Dense(input_shape))

    return model

def plot_results(val_x, preds):
     
    f, ax = plt.subplots(2,5)
    f.set_size_inches(80, 40)
    for i in range(5):
        ax[0, i].imshow(val_x[i].reshape(28, 28))

    for i in range(5):
        ax[1, i].imshow(preds[i].reshape(28, 28))

    plt.show()


train_x, val_x, train_y, val_y = read_and_split_data(path, fname)

model = create_model(train_x.shape[1], [1500, 1000, 500], 10)
print(model.summary())

model.compile(loss="mse", optimizer="adam")
early_stopping = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, verbose=1, mode="auto")

model.fit(train_x, train_x, epochs=20, batch_size=2048, validation_data=(val_x, val_x), callbacks=[early_stopping])

preds = model.predict(val_x)

model.save(model_path + "model_simple.h5")
plot_results(val_x, preds)
