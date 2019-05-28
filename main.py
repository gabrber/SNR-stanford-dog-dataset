import h5py
import matplotlib.pyplot as plt
import functools
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
    return keras.applications.mobilenet_v2.preprocess_input(img_array_expanded_dims)

def prepare_base_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(
    image_size, image_size, 3))  # imports the mobilenet model and discards the last 1000 neuron layer.
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    prediction = Dense(120, activation='softmax')(x)  # final layer with softmax activation
    model = Model(inputs=base_model.input, outputs=prediction)
    return model

def set_trainable_layers(model, count):
    for layer in model.layers[:count]:
        layer.trainable = False

    print(len(model.layers))
    for layer in model.layers:
        print(layer, layer.trainable)

    return model

def train_task(model, model_name):
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # included in our dependencies
    train_generator = train_datagen.flow_from_directory('bbox_dataset\\train',
                                                        target_size=(image_size, image_size),
                                                        color_mode='rgb',
                                                        batch_size=batch,
                                                        class_mode='categorical',
                                                        shuffle=True)

    valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    valid_generator = valid_datagen.flow_from_directory('bbox_dataset\\valid',
                                                        target_size=(image_size, image_size),
                                                        color_mode='rgb',
                                                        batch_size=batch,
                                                        class_mode='categorical',
                                                        shuffle=True)

    top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)
    top5_acc.__name__ = "top5_acc"

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy', top5_acc])
    # Adam optimizer
    # loss function will be categorical cross entropy
    # evaluation metric will be accuracy

    step_size_train = train_generator.n // train_generator.batch_size
    valid_steps = valid_generator.n // valid_generator.batch_size

    checkpoint = ModelCheckpoint("models\\" + model_name + ".h5", monitor='acc', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='max')

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=step_size_train,
                                  epochs=epochs,
                                  validation_data=valid_generator,
                                  validation_steps=valid_steps,
                                  callbacks=[checkpoint])

    save_history(history, "history\\" + model_name)
    plot_history(history)
    return model

def test_task(model):
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = test_datagen.flow_from_directory('bbox_dataset\\test',
                                                      target_size=(image_size, image_size),
                                                      color_mode='rgb',
                                                      batch_size=1,
                                                      class_mode='categorical',
                                                      shuffle=True)
    test_steps = test_generator.n // test_generator.batch_size
    testing = model.evaluate_generator(test_generator,
                                       test_steps)

    print(model.metrics_names)
    print("Cropped images")
    print(testing)

    original_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    original_test_generator = original_test_datagen.flow_from_directory('dataset\\test',
                                                                        target_size=(image_size, image_size),
                                                                        color_mode='rgb',
                                                                        batch_size=1,
                                                                        class_mode='categorical',
                                                                        shuffle=True)
    original_test_steps = original_test_generator.n // original_test_generator.batch_size

    original_testing = model.evaluate_generator(original_test_generator,
                                                original_test_steps)

    print("Original images")
    print(original_testing)

def load_model(model_name):
    top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)
    top5_acc.__name__ = "top5_acc"

    model = load_model("models\\" + model_name + ".h5", custom_objects={'top5_acc': top5_acc})
    return model

def load_hisotry(model_name):
    history = json.load(open("history\\" + model_name, 'r'))
    plot_history_read(history)


def zad1():
    base_model = prepare_base_model()
    trainable_layers_model = set_trainable_layers(base_model,-2)
    # load_hisotry("zad1")
    model = train_task(trainable_layers_model,"zad1")
    # model = load_model("zad1")
    test_task(model)

def zad2():
    base_model = prepare_base_model()
    trainable_layers_model = set_trainable_layers(base_model,-5)
    # load_hisotry("zad2")
    model = train_task(base_model,"zad2")
    # model = load_model("zad2")
    test_task(model)

def zad3a():
    base_model = prepare_base_model()
    # load_hisotry("zad3a")
    model = train_task(base_model,"zad3a")
    # model = load_model("zad3a")
    test_task(model)


if __name__ == "__main__":

    image_size = 96
    epochs = 10
    batch = 32

    zad1()
    zad2()
    zad3a()
