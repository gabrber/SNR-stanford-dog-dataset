from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

img_width, img_height = 32, 32
train_data_dir = "dataset_test\\training"
validation_data_dir = "dataset_test\\validation"
nb_train_samples = 10264
nb_validation_samples = 4072
batch_size = 16
epochs = 50

valid_steps = nb_validation_samples / batch_size

model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

#Adding custom Layers
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
#x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(120, activation="softmax")(x)

# creating the final model
model_final = Model(input = model.input, output = predictions)

# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model_final.layers[:-4]:
    layer.trainable = False

for layer in model_final.layers:
    print(layer, layer.trainable)


# compile the model
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
rescale = 1./255)

test_datagen = ImageDataGenerator(
rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size,
class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size = (img_height, img_width),
class_mode = "categorical")

# Save the model according to the conditions
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


# Train the model
model_final.fit_generator(
train_generator,
samples_per_epoch = nb_train_samples,
nb_epoch = epochs,
validation_data = validation_generator,
validation_steps = valid_steps,
#nb_val_samples = nb_validation_samples)
callbacks = [checkpoint, early])