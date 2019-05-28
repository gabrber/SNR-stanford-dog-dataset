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
from xml.dom import minidom


def get_bbox(img, data_dir, img_name):
    annotation_name = img_name.split('.')[0]
    mydoc = minidom.parse('dataset_raw\\Annotation\\' + data_dir + '\\' + annotation_name)
    xmin = int(mydoc.childNodes[0].getElementsByTagName("xmin")[0].childNodes[0].toxml())
    ymin = int(mydoc.childNodes[0].getElementsByTagName("ymin")[0].childNodes[0].toxml())
    xmax = int(mydoc.childNodes[0].getElementsByTagName("xmax")[0].childNodes[0].toxml())
    ymax = int(mydoc.childNodes[0].getElementsByTagName("ymax")[0].childNodes[0].toxml())
    cropped_image = img[ymin:ymax, xmin:xmax]
    return cropped_image

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
        for img_name in listdir(img_dir):
            img = cv2.imread(img_dir + "\\" + img_name)
            cropped_img = get_bbox(img, img_dir_name, img_name)
            if (count < train_count):
                cv2.imwrite(train_dir + "\\" + label + "\\" + img_name, cropped_img)
                count=count+1
            elif (count < (train_count + valid_count)):
                cv2.imwrite(valid_dir + "\\" + label + "\\" + img_name, cropped_img)
                count=count+1
            else:
                cv2.imwrite(test_dir + "\\" + label + "\\" + img_name, cropped_img)

if __name__ == '__main__':

    prepare_directories("bbox_dataset\\train","bbox_dataset\\valid","bbox_dataset\\test")
    split_data("dataset_raw\\Images","bbox_dataset\\train","bbox_dataset\\valid","bbox_dataset\\test")
