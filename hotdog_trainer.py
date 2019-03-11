import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import datetime as dt


#
# Credit to YouTube channel 'Sentdex' (https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ)
# For his video series on using Tensorflow for constructing convolutional neural networks.
#

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

IMAGES_DIR = "images"
LOGS_DIR = "logs"
MODELS_DIR = "models"

CATEGORIES = ["hotdog", "other"]
MAX_CATEGORY_TRAINING_IMAGES = 4485

NOMINAL_IMG_SIZE_X = 100
NOMINAL_IMG_SIZE_Y = 100

training_data = []

def create_training_data():
    print("Creating training data...")
    for category in CATEGORIES:
        path = os.path.join(IMAGES_DIR, category)  # path to hotdog or other dir
        class_num = CATEGORIES.index(category)  # index of the category for this image in CATEGORIES

        all_category_images = os.listdir(path)

        # If there are more training images in this category than we need, shuffle the images and select only
        # as many as we can process
        if len(all_category_images) > MAX_CATEGORY_TRAINING_IMAGES:
            random.shuffle(all_category_images)

        training_images = all_category_images[:MAX_CATEGORY_TRAINING_IMAGES]
        image_index = 0
        bad_images = 0
        bad_image_files = []
        print("Reading '{0}' images:".format(category))

        for image in training_images:
            try:
                print("\r\tProgress: {}/{} ({:.0f}%)".format(image_index + 1,
                                                             len(training_images),
                                                             image_index / len(training_images) * 100),
                      end=" ")

                img_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_COLOR)
                scaled_array = cv2.resize(img_array, (NOMINAL_IMG_SIZE_X, NOMINAL_IMG_SIZE_Y))
                training_data.append([scaled_array, class_num])
                image_index += 1

            except Exception:
                bad_images += 1
                bad_image_files.append(image)
                pass

        if bad_images != 0:
            print()
            print("Warning: Encountered & ignored {} bad images:".format(bad_images))
            print("\n".join(bad_image_files))
        print()

    # Shuffle the order of the training images to achieve better learning results
    random.shuffle(training_data)

    print("Done. Found {} training images.".format(len(training_data)))


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)



create_training_data()

# training_data[0]  scaled image
# training_data[1]  category
x = []
y = []

for features, label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, NOMINAL_IMG_SIZE_X, NOMINAL_IMG_SIZE_Y, 3)

# Dump training dataset to disk
# pickle_out = open("x.pickle", "wb")
# pickle.dump(x, pickle_out)
# pickle_out.close()
#
# pickle_out = open("y.pickle", "wb")
# pickle.dump(y, pickle_out)
# pickle_out.close()


# Normalise dataset
x = x / 255.0


# Parameterised model constraints
dense_layers = [1]
layer_sizes = [64]
conv_layers = [5]

epochs = 15

# Parameterised Tensorflow model builder
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:

            MODEL_NAME = "hotdogs-vs-other-cnn-{}-conv-{}-nodes-{}-dense-{}-epochs-{}".format(conv_layer,
                                                                                    layer_size,
                                                                                    dense_layer,
                                                                                    epochs,
                                                                                    dt.datetime.now().strftime("%Y-%m-%d_%I%M%Shrs"))
            tensorboard = TensorBoard(log_dir='{}/{}'.format(LOGS_DIR, MODEL_NAME))
            print("Building model '{}'...".format(MODEL_NAME))

            # Build CNN model
            model = Sequential()

            # L1
            model.add(Conv2D(layer_size, (3, 3), input_shape=x.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for i in range(conv_layer - 1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())  # convert 3D feature maps to 1D feature vectors
            for i in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))

            # Output layer
            model.add(Dense(1))
            model.add(Activation("sigmoid"))

            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
            model.fit(x, y, batch_size=32, epochs=epochs, validation_split=0.1, callbacks=[tensorboard])
            make_dir(MODELS_DIR)

            model_filename = "{}/{}{}".format(MODELS_DIR, MODEL_NAME, ".model")
            tflite_filename = "{}/{}{}".format(MODELS_DIR, MODEL_NAME, ".tflite")

            #model.save(model_filename)
            tf.keras.models.save_model(model, model_filename)

            # Convert to TensorFlow Lite model.
            converter = tf.lite.TFLiteConverter.from_keras_model_file(model_filename)
            tflite_model = converter.convert()
            open(tflite_filename, "wb").write(tflite_model)
