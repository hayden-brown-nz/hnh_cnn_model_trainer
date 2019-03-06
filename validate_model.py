import matplotlib.pyplot as plt
import cv2
import tensorflow as tf


CATEGORIES = ["hotdog", "other"]

NOMINAL_IMG_SIZE_X = 50
NOMINAL_IMG_SIZE_Y = 50

IMAGES_DIR = "images"
LOGS_DIR = "logs"
MODELS_DIR = "models"


# Convolutional Neural Network Model file (Tensorflow keras model)
CNN_MODEL = 'hotdogs-vs-other-cnn-2-conv-64-nodes-1-dense-1551868881.model'
TEST_IMAGE = 'test_hotdog.jpg'


def prepare(filepath):
    image_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    new_array = cv2.resize(image_array, NOMINAL_IMG_SIZE_X, NOMINAL_IMG_SIZE_Y)
    return new_array.reshape(-1, NOMINAL_IMG_SIZE_X, NOMINAL_IMG_SIZE_Y)


try:

    # Load model
    model = tf.keras.models.load_model("{}/{}".format(MODELS_DIR, CNN_MODEL, CNN_MODEL))

    # Predict
    prediction = model.predict([prepare(TEST_IMAGE)])
    prediction_category = CATEGORIES[int(prediction[0][0])]
    print("Prediction: {}".format(prediction_category)

    # Show image
    img_array = cv2.imread(TEST_IMAGE)
    plt.imshow(img_array[:,:,[2,1,0]])
    plt.title(TEST_IMAGE + " - Prediction: {}".format(prediction_category))
    plt.show()

except Exception:
    pass



# Scale image
# IMG_SIZE = 50

# plt.imshow(new_array[:, :, [2, 1, 0]])
# plt.show()


def prepare(filepath):
    NOMINAL_IMG_SIZE_X = 50
    NOMINAL_IMG_SIZE_Y = 50

    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    new_array = cv2.resize(img_array, NOMINAL_IMG_SIZE_X, NOMINAL_IMG_SIZE_Y)
    return new_array.reshape(-1, NOMINAL_IMG_SIZE_X, NOMINAL_IMG_SIZE_Y)



