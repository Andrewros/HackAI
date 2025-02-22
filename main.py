import cv2
import numpy as np
from joblib import load
import sklearn
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import sklearn
import numpy as np
import cv2
import os
from joblib import dump, load  # Use joblib instead of pickle



model, label_encoder, trained_version = load("model.joblib")


# Function to preprocess a single test image
def preprocess_image(image_path, img_size=(24, 24)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Error loading image: {image_path}. Check the file path.")
    
    img_resized = cv2.resize(img, img_size)  # Resize to match training images
    img_flattened = img_resized.flatten()  # Flatten the image to a 1D array
    
    return img_flattened.reshape(1, -1)  # Reshape for model input

# Prompt user to enter an image file path
image_path = "/Users/andrewrosenblum/Desktop/IMG_3561.jpg"

try:
    # Preprocess the image
    test_image = preprocess_image(image_path)

    # Predict class label
    predicted_label = model.predict(test_image)
    predicted_class = label_encoder.inverse_transform(predicted_label)

    print(f"Predicted class: {predicted_class[0]}")

except Exception as e:
    print(f"Error: {e}")