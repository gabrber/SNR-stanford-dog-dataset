import h5py
import matplotlib.pyplot as plt
import functools
import keras
from keras import backend as K, Sequential
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
    train_top1_acc = history.history['acc']
    train_top5_acc = history.history['top5_acc']
    val_top1 = history.history['val_acc']
    val_top5 = history.history['val_top5_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(train_top1_acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, train_top1_acc, 'b', label='Training top1 acc')
    plt.plot(x, train_top5_acc, 'c', label='Training top5 acc')
    plt.plot(x, val_top1, 'r', label='Validation top1 acc')
    plt.plot(x, val_top5, 'g', label='Validation top5 acc')
    plt.title('Training and validation top1, top5 accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def plot_history_read(history):
    train_top1_acc = history['acc']
    train_top5_acc = history['top5_acc']
    val_top1 = history['val_acc']
    val_top5 = history['val_top5_acc']
    loss = history['loss']
    val_loss = history['val_loss']
    x = range(1, len(train_top1_acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, train_top1_acc, 'b', label='Training top1 acc')
    plt.plot(x, train_top5_acc, 'c', label='Training top5 acc')
    plt.plot(x, val_top1, 'r', label='Validation top1 acc')
    plt.plot(x, val_top5, 'g', label='Validation top5 acc')
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

top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)
top5_acc.__name__ = "top5_acc"

#prepare model
base_model = load_model('models\\zad3_mobilenet.h5', custom_objects={'top5_acc': top5_acc}) #imports the mobilenet model and discards the last 1000 neuron layer.

print(len(base_model.layers))
for layer in base_model.layers:
    print(layer, layer.trainable)
print(len(base_model.layers))


model = Sequential()
for layer in base_model.layers[0:44]:
    model.add(layer)

for layer in base_model.layers[73:]:
    #print(layer)
    model.add(layer)

print(len(model.layers))
for layer in model.layers:
    print(layer, layer.trainable)

#training dataset
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies
train_generator=train_datagen.flow_from_directory('bbox_dataset\\train',
                                                 target_size=(image_size,image_size),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

# validation dataset
valid_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
valid_generator=valid_datagen.flow_from_directory('bbox_dataset\\valid',
                                                 target_size=(image_size,image_size),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy',top5_acc])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train=train_generator.n//train_generator.batch_size
valid_steps=valid_generator.n//valid_generator.batch_size

checkpoint = ModelCheckpoint("models\\zad3_b_mobilenet.h5", monitor='acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')

# Train and save model
history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=10,
                    validation_data = valid_generator,
                    validation_steps = valid_steps,
                    callbacks=[checkpoint])

save_history(history, "history\\zad3_b_mobilenet")
plot_history(history)

# Load history from file
# history = json.load(open("history\\zad3_b_mobilenet", 'r'))
# plot_history_read(history)

# Check .h5 file
# with h5py.File('zad3_b_mobilenet.h5', mode='r') as f:
#     for key in f:
#         print(key,f[key])

#model = load_model('models\\zad3_b_mobilenet.h5', custom_objects={'top5_acc': top5_acc})

# test dataset
test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator=test_datagen.flow_from_directory('bbox_dataset\\test',
                                                 target_size=(image_size,image_size),
                                                 color_mode='rgb',
                                                 batch_size=1,
                                                 class_mode='categorical',
                                                 shuffle=True)
test_steps=test_generator.n//test_generator.batch_size
testing = model.evaluate_generator(test_generator,
                                        test_steps)

print(model.metrics_names)
print("Cropped images")
print(testing)

original_test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
original_test_generator=original_test_datagen.flow_from_directory('dataset\\test',
                                                 target_size=(image_size,image_size),
                                                 color_mode='rgb',
                                                 batch_size=1,
                                                 class_mode='categorical',
                                                 shuffle=True)
original_test_steps=original_test_generator.n//original_test_generator.batch_size


original_testing = model.evaluate_generator(original_test_generator,
                                        original_test_steps)

print("Original images")
print(original_testing)