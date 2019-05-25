import h5py
import matplotlib.pyplot as plt

import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers.core import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils, VGG16
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np
from IPython.display import Image
from keras.optimizers import Adam

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

def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(image_size, image_size))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

image_size = 112

#prepare model
base_model=VGG16(weights='imagenet',include_top=False,input_shape = (image_size, image_size, 3)) #imports the mobilenet model and discards the last 1000 neuron layer.
x=base_model.output

#Adding custom Layers
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(120, activation="softmax")(x)
model=Model(inputs=base_model.input,outputs=predictions)

for layer in model.layers[:-4]:
    layer.trainable=False

for layer in model.layers:
    print(layer, layer.trainable)

#training dataset
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies
train_generator=train_datagen.flow_from_directory('dataset\\train',
                                                 target_size=(image_size,image_size),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

# validation dataset
valid_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
valid_generator=valid_datagen.flow_from_directory('dataset\\valid',
                                                 target_size=(image_size,image_size),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train=train_generator.n//train_generator.batch_size
valid_steps=valid_generator.n//valid_generator.batch_size

#checkpoint = ModelCheckpoint("zad1_weights_vgg16.h5", monitor='acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')

model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=10,
                    validation_data = valid_generator,
                    validation_steps = valid_steps)
