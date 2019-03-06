import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

IMAGES_DIR = "images"
LOGS_DIR = "logs"
MODELS_DIR = "models"

CATEGORIES = ["hotdog", "other"]
MAX_CATEGORY_TRAINING_IMAGES = 1000

NOMINAL_IMG_SIZE_X = 50
NOMINAL_IMG_SIZE_Y = 50

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
        print("Reading '{0}' images:".format(category))

        for image in training_images:
            try:
                print("\r\tProgress: {}/{} ({:.0f}%)".format(image_index + 1,
                                                             len(training_images),
                                                             image_index / len(training_images) * 100),
                      end=" ")

                img_array = cv2.imread(os.path.join(path, image))
                scaled_array = cv2.resize(img_array, (NOMINAL_IMG_SIZE_X, NOMINAL_IMG_SIZE_Y))
                training_data.append([scaled_array, class_num])
                image_index += 1

            except Exception:
                bad_images += 1
                pass

        if bad_images != 0:
            print("(Warning: Encountered {} bad images)".format(bad_images))

        print()

    # Shuffle the order of the training images to achieve better learning results
    random.shuffle(training_data)

    print("Done. Found {} training images.".format(len(training_data)))


def make_dir(file_path):
    directory = os.path.dirname(file_path)
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
dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

epochs = 10

# Parameterised Tensorflow model builder
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:

            MODEL_NAME = "hotdogs-vs-other-cnn-{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
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
            model.save("{}/{}{}".format(MODELS_DIR, MODEL_NAME, ".model"))

# Show raw image
# plt.imshow(img_array[:,:,[2,1,0]])
# plt.imshow(scaled_array[:, :, [2, 1, 0]])
# plt.title(image + " ({} x {})".format(img_array.shape[0], img_array.shape[1]))
# plt.show()


# Scale image
# IMG_SIZE = 50

# plt.imshow(new_array[:, :, [2, 1, 0]])
# plt.show()



