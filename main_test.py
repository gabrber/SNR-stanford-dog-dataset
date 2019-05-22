from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from keras.layers import Flatten, Dense, Dropout
from keras.utils import np_utils
from keras_preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras.models import Model

from keras.preprocessing import image

# VGG16
from keras.applications import VGG16, vgg16
from keras import models
from keras import layers
from keras import optimizers

# Helper libraries
import numpy as np
from os import listdir
from shutil import copyfile

from keras.optimizers import adam

import sklearn

from sklearn.model_selection import train_test_split
import cv2
import os

import matplotlib.pyplot as plt

def split_data(data_dir, train_dir, valid_dir, test_dir):
    print("[INFO] organizing data...")
    for img_dir_name in listdir(data_dir):
        label = img_dir_name.split("-")[1]
        if not os.path.exists(train_dir + "\\" + label):
            os.makedirs(train_dir + "\\" + label)
        if not os.path.exists(valid_dir + "\\" + label):
            os.makedirs(valid_dir + "\\" + label)
        if not os.path.exists(test_dir + "\\" + label):
            os.makedirs(test_dir + "\\" + label)
        img_dir = data_dir + "\\" + img_dir_name
        all = len(listdir(img_dir))
        train_count = int(all * 0.5)
        valid_count = int(all * 0.2)
        count = 0
        for img in listdir(img_dir):
            if (count < train_count):
                copyfile(img_dir + "\\" + img, train_dir + "\\" + label + "\\" + img)
                count=count+1
            elif (count < (train_count + valid_count)):
                copyfile(img_dir + "\\" + img, valid_dir + "\\" + label + "\\" + img)
                count=count+1
            else:
                copyfile(img_dir + "\\" + img, test_dir + "\\" + label + "\\" + img)

# prepare an image to be vgg16 friendly
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    pImg = vgg16.preprocess_input(img_array_expanded)
    return pImg

def prepare_model_zad1():
    # Freeze the layers except the last 4 layers (fully-connected classifier)
    x = vgg_conv.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(120, activation="softmax")(x)

    model_final = Model(input=vgg_conv.input, output=predictions)
    model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                        metrics=["accuracy"])

    for layer in model_final.layers[:-5]:
        layer.trainable = False

    return model_final

def predict_image(img):
    img_predict = vgg_conv.predict(img)
    # convert the probabilities to class labels
    label = vgg16.decode_predictions(img_predict)
    # retrieve the most likely result, e.g. highest probability
    print(label)
    label = label[0][0]
    # print the classification
    print('%s (%.2f%%)' % (label[1], label[2] * 100))


def training_data(data_dir, X,Y, image_size=224):
    print("[INFO] preparing training data...")
    for img in listdir(data_dir):
        label = img.split("-")[0]
        img_path = data_dir + "\\" + img
        #img_read = prepare_image(img_path)
        img_read = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_read = cv2.resize(img_read, (image_size, image_size))
        X.append(np.array(img_read))
        #train_X.append(img_read)
        Y.append(label)

def transform_labels(Y, num_classes=120):
    print("[INFO] transforming labels...")
    le = LabelEncoder()
    labels = le.fit_transform(Y)
    labels = keras.utils.to_categorical(labels, num_classes) #Converts a class vector (integers) to binary class matrix.
    return labels

def transform_img_data(X):
    print("[INFO] transforming images...")
    data = np.array(X)
    data = data / 255  # 0 to 1 scaling
    return data

def compile_model(model):
    print("[INFO] compiling model...")
    opt = adam(lr=0.001, decay=1e-6)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model

def train_model(model, train_data, valid_data):
    print("[INFO] training on training set...")
    # Train net:
    #history = model.fit(train_X, train_Y, epochs=epochs, batch_size=batch, validation_data=([valid_X], [valid_Y]))
    history = model.fit_generator(
        train_data,
        samples_per_epoch = 10264,
        epochs = 50,
        validation_data = valid_data,
        nb_val_samples = 4072)
    plot_history(history)

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

def prepare_img_generator(train_data_dir, img_height, img_width, batch_size):
    datagen = ImageDataGenerator(rescale = 1./255)
    prepared_data = datagen.flow_from_directory(train_data_dir,
        target_size = (img_height, img_width),
        batch_size = batch_size,
        class_mode = "categorical")
    return(prepared_data)

if __name__ == '__main__':

    image_size = 224
    num_classes = 120 #dog breeds
    batch = 100

    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    model = prepare_model_zad1()

    # Check the trainable status of the individual layers
    for layer in model.layers:
        print(layer, layer.trainable)

    #Uncomment if data is raw
    #split_data("dataset_raw\\Images","dataset_test\\training","dataset_test\\validation","dataset_test\\test")
    train_data = prepare_img_generator("dataset_test\\training", image_size, image_size, batch)
    valid_data = prepare_img_generator("dataset_test\\validation", image_size, image_size, batch)

    model_comp = compile_model(model)
    train_model(model_comp, train_data, valid_data)






