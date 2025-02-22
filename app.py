from flask import Flask, render_template, request, session, redirect, flash
from flask_session import Session
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import cv2
from joblib import dump, load
import sqlite3
import numpy as np
from PIL import Image
import sklearn
from werkzeug.security import check_password_hash, generate_password_hash
import os
from functools import wraps

# Flask App Setup
app = Flask(__name__)
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Constants
PIXELSIZE = 24
CLASSES = ['Andrew', 'Sai']
DATASET_PATH = "static/datasets"
N_ESTIMATORS = 100  # Number of trees in RandomForest
MAX_DEPTH = 10  # Max depth of trees

# Authentication Decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated_function

# Function to Load Images from Folders
def load_images_from_folder(folder_path, img_size=(PIXELSIZE, PIXELSIZE)):
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
                    labels.append(class_name)

    return np.array(images), np.array(labels)

# Function to Preprocess a Single Image
def preprocess_image(image_path, img_size=(24, 24)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Error loading image: {image_path}. Check the file path.")
    
    img_resized = cv2.resize(img, img_size)  # Resize to match training images
    img_flattened = img_resized.flatten()  # Flatten the image to a 1D array
    
    return img_flattened.reshape(1, -1)

# Login Route
@app.route("/login", methods=["GET", "POST"])
def login():
    session.clear()
    if request.method == "POST":
        if not request.form.get("username") or not request.form.get("password"):
            return "Must provide username and password"

        db = sqlite3.connect("static/users.db")
        cursor = db.cursor()
        try:
            rows = cursor.execute("SELECT * FROM users WHERE username = ?", (request.form.get("username"),)).fetchone()
            if not rows or not check_password_hash(rows[1], request.form.get("password")):
                return render_template("error.html", message="Invalid username and/or password")

            session["user_id"] = rows[2]
            return redirect("/")
        except:
            return render_template("error.html", message="Something you entered was invalid")
    else:
        return render_template("login.html")

# Registration Route
@app.route("/register", methods=["GET", "POST"])
def register():
    session.clear()
    if request.method == "POST":
        if not request.form.get("username") or not request.form.get("password"):
            return "Must provide username and password"

        hashedpassword = generate_password_hash(request.form.get("password"))
        db = sqlite3.connect("static/users.db")
        cursor = db.cursor()
        cursor.execute("INSERT INTO users(username, password) VALUES (?, ?)", (request.form.get("username"), hashedpassword))
        db.commit()
        rows = cursor.execute("SELECT * FROM users WHERE username = ?", (request.form.get("username"),)).fetchone()
        session["user_id"] = rows[2]
        return redirect("/")
    else:
        return render_template("register.html")

# Home Route (Prediction)
@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    if request.method == "GET":
        return render_template("index.html")
    else:
        try:
            file = request.files["image"].stream
            file = Image.open(file).convert("L")
            file = np.array(file)

            # Image preprocessing
            img_resized = cv2.resize(file, (PIXELSIZE, PIXELSIZE))
            img_flattened = img_resized.flatten().reshape(1, -1)

            # Load model
            model_path = f"{DATASET_PATH}/train{session['user_id']}/data.joblib"
            if os.path.exists(model_path):
                model, label_encoder, trained_version = load(model_path)
            else:
                model, label_encoder, trained_version = load("static/model.joblib")

            # Predict
            yhat = model.predict(img_flattened)
            predicted_class = label_encoder.inverse_transform(yhat)

            return render_template("prediction.html", name=predicted_class[0])
        except Exception as e:
            return render_template("error.html", message=str(e))

# Training Route
@app.route("/train", methods=["GET", "POST"])
@login_required
def train():
    if request.method == "GET":
        return render_template("train.html")
    else:
        classname = request.form.get("class").capitalize()
        user_train_path = f"{DATASET_PATH}/train{session['user_id']}"
        model_path = f"{user_train_path}/data.joblib"

        # Ensure user's dataset directory exists
        if not os.path.exists(user_train_path):
            os.makedirs(user_train_path)
            for clas in CLASSES:
                os.system(f"cp -R static/{clas} {user_train_path}")

        class_folder = f"{user_train_path}/{classname}"
        if os.path.exists(class_folder):
            return render_template("error.html", message="Class already exists.")

        # Create a directory for the new class
        os.makedirs(class_folder)

        # Save uploaded images
        for image in request.files.getlist("images"):
            filename = image.filename.replace("/", "")
            save_path = os.path.join(class_folder, filename)
            image.save(save_path)

        # Load the dataset (new + existing)
        x_new, y_new = load_images_from_folder(user_train_path)

        # Try loading an existing model
        if os.path.exists(model_path):
            model, label_encoder, trained_version = load(model_path)

            # Update label encoder with new classes
            all_labels = list(label_encoder.classes_) + [classname]
            label_encoder.fit(all_labels)  # Update the label encoder with all classes

            # Transform labels using the updated encoder
            y_new = label_encoder.transform(y_new)
        else:
            label_encoder = LabelEncoder()
            y_new = label_encoder.fit_transform(y_new)

        # Train or Retrain the Model
        model = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42)
        model.fit(x_new, y_new)

        # Save the updated model
        dump((model, label_encoder, sklearn.__version__), model_path)

        # Update the database
        db = sqlite3.connect("static/users.db")
        cursor = db.cursor()
        cursor.execute("INSERT INTO classes (id, class) VALUES (?, ?)", (session["user_id"], classname,))
        db.commit()

        flash("Training complete with updated data!")
        return redirect("/")

# def train():
#     if request.method == "GET":
#         return render_template("train.html")
#     else:
#         classname = request.form.get("class").capitalize()
#         user_train_path = f"{DATASET_PATH}/train{session['user_id']}"

#         # Create directory if not exists
#         if not os.path.exists(user_train_path):
#             os.makedirs(user_train_path)
#             for clas in CLASSES:
#                 os.system(f"cp -R static/{clas} {user_train_path}")

#         if os.path.exists(f"{user_train_path}/{classname}"):
#             return render_template("error.html", message="Class is already being used")

#         # Save uploaded images
#         os.makedirs(f"{user_train_path}/{classname}")
#         for image in request.files.getlist("images"):
#             filename = image.filename.replace("/", "")
#             save_path = f"{user_train_path}/{classname}/{filename}"
#             image.save(save_path)

#         # Load dataset
#         x, y = load_images_from_folder(user_train_path)
#         label_encoder = LabelEncoder()
#         y = label_encoder.fit_transform(y)

#         # Train Random Forest Model
#         model = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42)
#         model.fit(x, y)

#         dump((model, label_encoder, sklearn.__version__), f"{user_train_path}/data.joblib")

#         # Update database
#         db = sqlite3.connect("static/users.db")
#         cursor = db.cursor()
#         cursor.execute("INSERT INTO classes (id, class) VALUES (?, ?)", (session["user_id"], classname,))
#         db.commit()

#         flash("Training complete")
#         return redirect("/")
@app.route("/reset", methods=["GET", "POST"])
@login_required
def reset():
    if request.method=="GET":
        return render_template("reset.html")
    else:
        db=sqlite3.connect("static/users.db")
        cursor=db.cursor()
        cursor.execute("DELETE from classes WHERE id=?", (session['user_id'],))
        db.commit()
        os.system(f"rm -rf static/datasets/train{session['user_id']}")
        os.system(f"mkdir {DATASET_PATH}/train{session['user_id']}")
        for clas in CLASSES:
            os.system(f"cp -R static/{clas} {DATASET_PATH}/train{session['user_id']}")
        return redirect("/")
# Logout Route
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# Run Flask App
if __name__ == "__main__":
    app.run(port=3000, debug=True)
