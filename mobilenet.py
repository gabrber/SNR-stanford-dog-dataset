import h5py
import matplotlib.pyplot as plt

import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils, MobileNetV2
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np
from IPython.display import Image
from keras.optimizers import Adam
import json

def save_history(history, filename):
    with open(filename, 'w') as f:
        json.dump(history.history, f)

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
    plt.show()

def plot_history_read(history):
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']
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
    plt.show()

def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(image_size, image_size))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

image_size = 128

# Base model with top layer
# base_model_full=MobileNet(weights='imagenet',include_top=True)
# print(len(base_model_full.layers))
# for layer in base_model_full.layers:
#     print(layer, layer.trainable)

#prepare model
base_model=MobileNet(weights='imagenet',include_top=False,input_shape = (image_size, image_size, 3)) #imports the mobilenet model and discards the last 1000 neuron layer.
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
prediction=Dense(120,activation='softmax')(x) #final layer with softmax activation
model=Model(inputs=base_model.input,outputs=prediction)

for layer in model.layers[:-5]:
    layer.trainable=False

print(len(model.layers))
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

checkpoint = ModelCheckpoint("models\\zad1_mobilenet.h5", monitor='acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')

history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=10,
                    validation_data = valid_generator,
                    validation_steps = valid_steps,
                    callbacks=[checkpoint])

save_history(history, "history\\zad1_mobilenet")
plot_history(history)

# Load history from file
# history = json.load(open("history\\zad1_mobilenet", 'r'))
# plot_history_read(history)

# Check .h5 file
# with h5py.File('zad1_weights.h5', mode='r') as f:
#     for key in f:
#         print(key,f[key])

model_zad1 = load_model('models\\zad1_mobilenet.h5')
