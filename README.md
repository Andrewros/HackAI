Face Recognition Flask App
Description
This is a Flask-based web application that performs face recognition using RandomForestClassifier from scikit-learn. The app allows users to log in, register, train a model with images, and predict identities based on uploaded images.

Features
User Authentication: Register and login functionality with password hashing.

Image Preprocessing: Converts images to grayscale and resizes them for training.

RandomForest Classifier: Used to train a model for recognizing users.

Dataset Management: Users can add new classes and train the model dynamically.

Session Management: Uses Flask-Session to store user sessions.

SQLite Database: Stores user authentication data and class information.

Reset Functionality: Allows users to clear their training data.

Technologies Used
Python (Flask, OpenCV, NumPy, PIL)

Scikit-Learn (RandomForest, LabelEncoder)

SQLite Database

Flask-Session

Gunicorn (for Heroku deployment)

Installation
Prerequisites
Ensure you have Python 3.11+ installed.

Steps
Clone the Repository

git clone https://github.com/your-repo/face-recognition-flask.git
cd face-recognition-flask
Create and Activate Virtual Environment

python -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate    # For Windows
Install Dependencies

pip install -r requirements.txt
Set Up Database

sqlite3 static/users.db < schema.sql
Run the Application

python app.py
Usage
Register & Login
Visit http://127.0.0.1:5000/register to create an account.

Login at http://127.0.0.1:5000/login.

Upload & Predict
Upload an image on the homepage to predict the person’s identity.

Train Model
Go to http://127.0.0.1:5000/train to upload images and train the model.

Reset Training Data
Go to http://127.0.0.1:5000/reset to clear trained data.

Deployment on Heroku
Login to Heroku

heroku login
Create a New Heroku App

heroku create your-app-name
Deploy Code

git add .
git commit -m "Deploy to Heroku"
git push heroku main
Scale Up & Open App

heroku ps:scale web=1
heroku open
Troubleshooting
If the app crashes with H10 (App Crashed), run:

heroku logs --tail
Ensure gunicorn is installed and listed in requirements.txt:

pip install gunicorn
pip freeze > requirements.txt
git commit -am "Add gunicorn"
git push heroku main
License
This project is licensed under the MIT License. Feel free to modify and distribute it.

Author
Andrew Rosenblum
