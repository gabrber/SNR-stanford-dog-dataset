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

def prepare_directories(train_dir, valid_dir, test_dir):
    print("[INFO] preparing directories...")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)


def split_data(data_dir, train_dir, valid_dir, test_dir, nb_train = 0.6, nb_valid = 0.2):
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
        train_count = int(all * nb_train)
        valid_count = int(all * nb_valid)
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

if __name__ == '__main__':

    prepare_directories("dataset\\train","dataset\\valid","dataset\\test")
    split_data("dataset_raw\\Images","dataset\\train","dataset\\valid","dataset\\test")






