from tensorflow.keras.models import load_model
import numpy as np
import cv2


def predict(img_array):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    resized_image = cv2.resize(img_array, (224, 224))
    # Load the model
    model = load_model("keras_model.h5", compile=False)
    # Load the labels
    class_names = open("labels.txt", "r").readlines()
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Normalize the image
    normalized_image_array = (resized_image.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    return class_name[2:]