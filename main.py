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
from sklearn import preprocessing, neighbors, svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.svm import SVC
import pickle
from sklearn.metrics import roc_curve, auc

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

def prepare_base_model():
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(
    image_size, image_size, 3))  # imports the mobilenet model and discards the last 1000 neuron layer.
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    prediction = Dense(120, activation='softmax')(x)  # final layer with softmax activation
    model = Model(inputs=base_model.input, outputs=prediction)
    return model

def set_trainable_layers(model, count):
    for layer in model.layers[:count]:
        layer.trainable = False

    # print(len(model.layers))
    # for layer in model.layers:
    #     print(layer, layer.trainable)

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

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy', top5_acc, keras.metrics.precision, keras.metrics.recall])
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

def load_model_from_file(model_name):
    top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)
    top5_acc.__name__ = "top5_acc"
    model = load_model("models\\" + model_name + ".h5", custom_objects={'top5_acc': top5_acc})
    return model

def load_hisotry(model_name):
    history = json.load(open("history\\" + model_name, 'r'))
    plot_history_read(history)

def prepare_zad3b_model(model):
    new_model = Sequential()
    # Layers from block 0 - 12 (without last block - 13)
    for layer in model.layers[:-8]:
        new_model.add(layer)
    for layer in model.layers[-2:]:
        new_model.add(layer)
    return new_model

def zad4_train(kernel, basemodel_id):

    # model without last layer
    model = load_model_from_file(basemodel_id)
    new_model = Sequential()
    for layer in model.layers[:-1]:
        new_model.add(layer)
    new_model.summary()

    # training data
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # included in our dependencies
    train_generator = train_datagen.flow_from_directory('bbox_dataset\\train',
                                                        target_size=(image_size, image_size),
                                                        color_mode='rgb',
                                                        batch_size=batch,
                                                        class_mode='categorical',
                                                        shuffle=True)
    X_train = []
    y_train = []
    train_steps = train_generator.n // train_generator.batch_size

    # classifier
    clf = SVC(kernel=kernel, degree=2)
    clf.get_params(True)

    for j in range (10):
        for i in range(train_steps):
            X_train, y_train = train_generator.next()
            y = y_train.argmax(1)
            X2 = new_model.predict(X_train)
            print(y)
            clf.fit(X2, y)

    filename = 'models\\'+kernel+'_'+basemodel_id+'.sav'
    pickle.dump(clf, open(filename, 'wb'))

def zad4_test(image_dir, basemodel_id, kernel):
    base_model = load_model_from_file(basemodel_id)
    model = Sequential()
    for layer in base_model.layers[:-1]:
        model.add(layer)

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = test_datagen.flow_from_directory(image_dir,
                                                      target_size=(image_size, image_size),
                                                      color_mode='rgb',
                                                      batch_size=10000,
                                                      class_mode='categorical',
                                                      shuffle=True)
    X_test, y_test = test_generator.next()
    y_test = y_test.argmax(1)
    X2 = model.predict(X_test)

    # load SVC
    clf_filename='models\\' + kernel + '_' + basemodel_id + '.sav'
    clf = pickle.load(open(clf_filename, 'rb'))
    # get the accuracy
    print("accuracy")
    print(clf.score(X2, y_test))

def zad1():
    print("[INFO] Processing zad 1")
    base_model = prepare_base_model()
    trainable_layers_model = set_trainable_layers(base_model,-2)
    # load_hisotry("zad1")
    model = train_task(trainable_layers_model,"zad1")
    # model = load_model_from_file("zad1")
    test_task(model)

def zad2():
    print("[INFO] Processing zad 2")
    base_model = prepare_base_model()
    trainable_layers_model = set_trainable_layers(base_model,-5)
    # load_hisotry("zad2")
    model = train_task(trainable_layers_model,"zad2")
    # model = load_model_from_file("zad2")
    test_task(model)

def zad2_extended():
    print("[INFO] Processing zad 2_extended")
    base_model = prepare_base_model()
    trainable_layers_model = set_trainable_layers(base_model,-8)
    # load_hisotry("zad2_extended")
    model = train_task(trainable_layers_model,"zad2_extended")
    # model = load_model_from_file("zad2_extended")
    test_task(model)

def zad3a():
    print("[INFO] Processing zad 3a")
    base_model = prepare_base_model()
    # load_hisotry("zad3a")
    model = train_task(base_model,"zad3a")
    # model = load_model_from_file("zad3a")
    test_task(model)

def zad3b():
    print("[INFO] Processing zad 3b")
    base_model = load_model_from_file("zad3a")
    modified_model = prepare_zad3b_model(base_model)
    # load_hisotry("zad3b")
    model = train_task(modified_model,"zad3b")
    # model = load_model_from_file("zad3b")
    test_task(model)

def zad4():
    print("[INFO] Processing zad 4")
    # zad4_train('linear','zad3a')
    # zad4_train('poly','zad3a')
    # zad4_train('rbf','zad3a')
    # zad4_train('linear','zad3b')
    # zad4_train('poly', 'zad3b')
    # zad4_train('rbf','zad3b')
    # zad4_test('bbox_dataset\\test','zad3a','linear')
    # zad4_test('dataset\\test', 'zad3b', 'poly')


def get_roc(model):
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = test_datagen.flow_from_directory('bbox_dataset\\test',
                                                      target_size=(image_size, image_size),
                                                      color_mode='rgb',
                                                      batch_size=1,
                                                      class_mode='categorical',
                                                      shuffle=True)
    test_steps = test_generator.n // test_generator.batch_size

    Y = []
    print("[INFO]Get labels")
    for i in range(test_steps):
      X_train, y_train = test_generator.next()
      Y.append(y_train)

    Y = np.asarray(Y).reshape((4201,120))
    print(Y.shape)

    print("[INFO]Get predictions")
    predictions = model.predict_generator(test_generator, test_steps)
    predictions = np.array(predictions)
    print(predictions.shape)


    print("[INFO]Get precision")
    Y_label = []
    Y_pred = []

    for i in Y:
        Y_label.append(np.argmax(i))
    for i in predictions:
        Y_pred.append(np.argmax(i))

    Y_label = np.asarray(Y_label).flatten()
    Y_pred = np.asarray(Y_pred).flatten()
    print(Y_label)
    print(Y_pred)

    precisions, _, _, _ = precision_recall_fscore_support(Y_label, Y_pred)
    print(precisions)

    print("[INFO]Get 3 best precisions")
    best_values = precisions.argsort()[-3:][::-1]
    print(best_values)

    print("[INFO]Create ROC Curves")
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(120):
        fpr[i], tpr[i], _ = roc_curve(Y[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    for i in best_values:
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

def test():
    test_model = load_model_from_file("zad1")
    get_roc(test_model)

if __name__ == "__main__":

    image_size = 128
    epochs = 10
    batch = 32

    zad1()
    zad2()
    zad2_extended()
    zad3a()
    zad3b()
    zad4()
