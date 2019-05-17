from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from keras.utils import np_utils
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

import matplotlib.pyplot as plt

# prepare an image to be vgg16 friendly
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    pImg = vgg16.preprocess_input(img_array_expanded)
    return pImg

def prepare_model_zad1():
    # Freeze the layers except the last 4 layers (fully-connected classifier)
    for layer in vgg_conv.layers[:-4]:
        layer.trainable = False

    # Check the trainable status of the individual layers
    for layer in vgg_conv.layers:
        print(layer, layer.trainable)

def predict_image(img):
    img_predict = vgg_conv.predict(img)
    # convert the probabilities to class labels
    label = vgg16.decode_predictions(img_predict)
    # retrieve the most likely result, e.g. highest probability
    print(label)
    label = label[0][0]
    # print the classification
    print('%s (%.2f%%)' % (label[1], label[2] * 100))


def split_data(data_dir, train_dir, valid_dir, test_dir):
    for img_dir_name in listdir(data_dir):
        label = img_dir_name.split("-")[1]
        img_dir = data_dir + "\\" + img_dir_name
        all = len(listdir(img_dir))
        train_count = int(all * 0.7)
        valid_count = int(all * 0.2)
        count = 0
        for img in listdir(img_dir):
            if (count < train_count):
                copyfile(img_dir + "\\" + img, train_dir + "\\" + label + "-" + img)
                count=count+1
            elif (count < (train_count + valid_count)):
                copyfile(img_dir + "\\" + img, valid_dir + "\\" + label + "-" + img)
                count=count+1
            else:
                copyfile(img_dir + "\\" + img, test_dir + "\\" + label + "-" + img)

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

def compile_and_train(model,train_X,train_Y,valid_X,valid_Y, epochs=128, batch=96):
    print("[INFO] compiling model...")
    opt = adam(lr=0.001, decay=1e-6)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
    model.summary()

    print("[INFO] training on training set...")
    # Train net:
    #history = model.fit(train_X, train_Y, epochs=epochs, batch_size=batch, validation_data=([valid_X], [valid_Y]))
#    history = model.fit_generator(IMDB_WIKI(train_X, train_Y), epochs=epochs, steps_per_epoch=batch, validation_steps=batch, validation_data=IMDB_WIKI(valid_X,valid_Y))
#    plot_history(history)

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

def IMDB_WIKI(X_samples, y_samples, batch_size=100):
    batch_size = len(X_samples) / batch_size
    X_batches = np.split(X_samples, batch_size)
    y_batches = np.split(y_samples, batch_size)
    for b in range(X_batches):
        x = np.array(map(transform_img_data, X_batches[b]))
        y = np.array(y_batches[b])
        yield x, y

if __name__ == '__main__':

    image_size = 224
    num_classes = 120 #dog breeds

    vgg_conv = VGG16(weights='imagenet', include_top=True, input_shape=(image_size, image_size, 3))
    prepare_model_zad1()

    #Uncomment if data is raw
    #split_data("dataset_raw\\Images","dataset\\training","dataset\\validation","dataset\\test")

    train_X = [] # images
    train_Y = [] # images' label
    training_data("dataset\\training",train_X,train_Y,image_size)
    #data = transform_img_data(train_X)
    labels = transform_labels(train_Y)

    valid_X = [] # images
    valid_Y = [] # images' label
    training_data("dataset\\validation",valid_X,valid_Y,image_size)
    #data_valid = transform_img_data(valid_X)
    labels_valid = transform_labels(valid_Y)

# TO FIX
    
    #compile_and_train(vgg_conv, data, labels, data_valid, labels_valid)
    compile_and_train(vgg_conv, train_X, labels, valid_X, labels_valid)

    # history = model.fit_generator(
    #     augs_gen.flow(x_train, y_train, batch_size=16),
    #     validation_data=(x_test, y_test),
    #     validation_steps=1000,
    #     steps_per_epoch=1000,
    #     epochs=20,
    #     verbose=1,
    #     callbacks=callbacks
    # )





