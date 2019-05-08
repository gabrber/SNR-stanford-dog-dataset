from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.preprocessing import image
from keras.applications import imagenet_utils, mobilenet

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# prepare an image to be mobilenet friendly
def prepare_image(img_path):
  img = image.load_img(img_path, target_size=(224, 224))
  img_array = image.img_to_array(img)
  img_array_expanded = np.expand_dims(img_array, axis=0)
  pImg = mobilenet.preprocess_input(img_array_expanded)
  return pImg

def predict(mobilenet, img):
  prediction = mobilenet.predict(img)
  # obtain the top-5 predictions
  results = imagenet_utils.decode_predictions(prediction)
  print(results)
  return results

if __name__ == '__main__':

  # path to test image
  test_img_path = "images\\test_image1.jpg"
  # process the test image
  pImg = prepare_image(test_img_path)
  # make predictions on test image using mobilenet
  # define the mobilenet model
  mNet = mobilenet.MobileNet()
  results = predict(mNet, pImg)
