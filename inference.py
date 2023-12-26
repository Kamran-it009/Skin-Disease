from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os

def predictor(img_array, model):
    # Extract the label from the image path
    # actual_label = os.path.basename(os.path.dirname(image_path))

    # Load the test image
    # img = image.load_img(image_path, target_size=(224, 224))
    # img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Load the model
    loaded_model = load_model(model)
    print(img_array.shape)

    # Predict the label for the image
    prediction = loaded_model.predict(img_array)
    predicted_id = np.argmax(prediction)

    # labels

    labels = ['Actinic keratosis', 'Atopic Dermatitis', 'Benign keratosis', 'Dermatofibroma',
              'Eczema', 'Melanocytic nevus', 'Melanoma', 'Squamous cell carcinoma', 'Tinea Ringworm Candidiasis',
              'Vascular lesion']
    # Decode labels
    class_labels = labels
    decoded_predicted_label = class_labels[predicted_id]
    return decoded_predicted_label

    # Display the test image along with true label and predicted label
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title(f"True Label: {actual_label}, Predicted Label: {decoded_predicted_label}")
    # plt.show()

# model_path = "vgg16_model.h5"
# test_image_path = "Skin_Dataset/val/Atopic Dermatitis/1_14.jpg"
# print(predictor(image_array, model_path))
