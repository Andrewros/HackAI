from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
import cv2
import os
from flask import Flask
from flask_session import Session
from sklearn.metrics import accuracy_score
# Function to load images from folders and assign labels
def load_images_from_folder(folder_path, img_size=(24, 24)):
    images = []
    labels = []
    class_names = os.listdir(folder_path)  # Folder names as class labels
    
    for class_name in class_names:
        class_folder = os.path.join(folder_path, class_name)
        if os.path.isdir(class_folder):
            for filename in os.listdir(class_folder):
                img_path = os.path.join(class_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, img_size)
                    img_flattened = img_resized.flatten()
                    images.append(img_flattened)
                    labels.append(class_name)  # Use folder name as label
    return np.array(images), np.array(labels)
x,y=load_images_from_folder("./trainmult")
xtrain, xtest, ytrain, ytest=train_test_split(x,y, test_size=0.1, random_state=42)
KERNEL="rbf"
CVALUE=1.9
model=svm.SVC(kernel=KERNEL, C=CVALUE)
model.fit(xtrain, ytrain)