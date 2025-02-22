from flask import Flask, render_template, request, session, redirect, flash
from flask_session import Session
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import cv2
import pickle
import sqlite3
import numpy as np
from PIL import Image
from werkzeug.security import check_password_hash, generate_password_hash
import os
from functools import wraps
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated_function
app=Flask(__name__)
PIXELSIZE=24
scaler=StandardScaler()
CLASSES = ['Andrew', 'Brad', 'Matthew']
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
USERNAME=0
PASSWORD=1
ID=2
DATASET_PATH="static/datasets"
KERNEL="rbf"
CVALUE=1.9
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
                    labels.append(class_name)  # Use folder name as label
    return np.array(images), np.array(labels)
@app.route("/login", methods=["GET", "POST"])
def login():
    # Forget any user_id
    session.clear() 
    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":    
      # Ensure username was submitted
        if not request.form.get("username"):
            return "must provide username"    
      # Ensure password was submitted
        elif not request.form.get("password"):
            return "must provide password"   
      # Query database for username
        db=sqlite3.connect("static/users.db")
        cursor=db.cursor()
        try:
            rows = cursor.execute("SELECT * FROM users WHERE username = ?",
                            request.form.get("username")).fetchone()
            # Ensure username exists and password is correct
            if not check_password_hash(rows[PASSWORD],
                                                        request.form.get("password")):
                return render_template("error.html", message="invalid username and/or password")

            # Remember which user has logged in
            session["user_id"] = rows[ID]

            # Redirect user to home page
            return redirect("/")
        except:
            return render_template("error.html", message="Something you entered was invalid")

  # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    session.clear()
    if request.method == "POST":
        # Ensure username was submitted
    
        if not request.form.get("username"):
          return "must provide username"
    
        # Ensure password was submitted
        elif not request.form.get("password") or not request.form.get(
            "password2") or request.form.get("password") != request.form.get(
              "password2"):
          return "must provide password"
        hashedpassword = generate_password_hash(request.form.get("password"))
        db=sqlite3.connect("static/users.db")
        cursor=db.cursor()
        cursor.execute("INSERT INTO users(username, password) VALUES (?, ?)",
                   (request.form.get("username"), hashedpassword))
        db.commit()
        rows = cursor.execute("SELECT * FROM users WHERE username = ?",
                          (request.form.get("username"), )).fetchone()
        session["user_id"] = rows[ID]
    # Remember which user has logged in

    # Redirect user to home page
        return redirect("/")

  # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("register.html")


@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    if request.method=="GET":
        return render_template("index.html")
    else:
        file=request.files["image"].stream
        file=Image.open(file).convert("L")
        file=np.array(file)
        #Rewrite this part
        img_resized = cv2.resize(file, (PIXELSIZE, PIXELSIZE))
        img_flattened = img_resized.flatten().reshape(1,-1)
        db=sqlite3.connect("static/users.db")
        cursor=db.cursor()
        data=cursor.execute("SELECT class FROM classes WHERE id=?", (session["user_id"],)).fetchall()
        classes=[datum[0].capitalize() for datum in data]
        classes.extend(CLASSES)
        classes.sort()
        if os.path.exists(f"{DATASET_PATH}/train{session['user_id']}/data.pt"):
            model=pickle.load(open(f"{DATASET_PATH}/train{session['user_id']}/data.pt", 'rb'))
        else:
            model=pickle.load(open(f"static/model.pt", 'rb'))
        yhat=model.predict(img_flattened)
        return render_template("prediction.html", name=yhat[0])
        # except:
        #     return redirect("/")
@app.route("/train", methods=["GET", "POST"])
@login_required
def train():
    if request.method=="GET":
        return render_template("train.html")
    else:
        classname=request.form.get("class").capitalize()
        if not os.path.exists(f"{DATASET_PATH}/train{session['user_id']}"):
            os.system(f"mkdir {DATASET_PATH}/train{session['user_id']}")
            for clas in CLASSES:
                os.system(f"cp -R static/{clas} {DATASET_PATH}/train{session['user_id']}")
        
        if os.path.exists(f"{DATASET_PATH}/train{session['user_id']}/{classname}"):
            return render_template("error.html", message="Class is already being used")
        
        images=request.files.getlist("images")
        #WRITE LOTS OF CODE BEFORE MAKING THE DIRECTORY
        os.system(f"mkdir {DATASET_PATH}/train{session['user_id']}/{classname}")
        for image in images:
            filename = image.filename
            filename=filename.replace("/", "")
            save_path = f"{DATASET_PATH}/train{session['user_id']}/{classname}/{filename}"
            image.save(save_path)
        db=sqlite3.connect("static/users.db")
        cursor=db.cursor()
        data=cursor.execute("SELECT class FROM classes WHERE id=?", (session["user_id"],)).fetchall()
        classes=[datum[0].capitalize() for datum in data]
        classes.extend(CLASSES)
        model=svm.SVC(kernel=KERNEL, C=CVALUE)
        model.fit()
        pickle.dump(model, open(f"{DATASET_PATH}/train{session['user_id']}/data.pt", 'wb'))
        db=sqlite3.connect("static/users.db")
        cursor=db.cursor()
        cursor.execute("INSERT INTO classes (id, class) VALUES (?,?)", (session["user_id"], classname,))
        db.commit()
        flash("Training complete")
        return redirect("/")
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


@app.route("/logout")
def logout():
  """Log user out"""

  # Forget any user_id
  session.clear()

  # Redirect user to login form
  return redirect("/")

if __name__=="__main__":
    app.run(port=3000, debug=True)
