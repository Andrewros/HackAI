ChatGPT 4o
You said:
When I run this code I get error 0
from joblib import load
import warnings
import sklearn

# Load the trained model
try:
    model, label_encoder = load("static/model.joblib")
    print(f"Model successfully loaded. Using scikit-learn version: {sklearn.__version__}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)
ChatGPT said:
Error code 0 typically indicates a silent failure where the program exits without raising an explicit error. However, since you have an exception handler (except Exception as e:), if something goes wrong while loading the model, it should print an error message.

Here are a few possible reasons why you might be encountering this issue:

1. Corrupt or Missing Model File
Ensure that "static/model.joblib" exists and is not corrupted.

Try manually loading the file using:

python
Copy
Edit
from joblib import load
model_data = load("static/model.joblib")
print(model_data)
If the file is corrupted, it will likely raise an error.

2. Mismatch in Scikit-Learn Versions
If the model was trained with a different version of scikit-learn, loading it might fail.

Print the current version using:

python
Copy
Edit
import sklearn
print(sklearn.__version__)
If your version differs significantly from the one used for training, try installing the exact version:

sh
Copy
Edit
pip install scikit-learn==X.Y.Z
3. Incompatibility Between Joblib Versions
If joblib versions differ between saving and loading, there may be incompatibilities.

Check your installed version:

python
Copy
Edit
import joblib
print(joblib.__version__)
If outdated, update it:

sh
Copy
Edit
pip install --upgrade joblib
4. Issue with Label Encoder Object
Ensure that model.joblib contains a tuple (model, label_encoder). If not, loading will fail.

Try loading without unpacking:

python
Copy
Edit
model_data = load("static/model.joblib")
print(type(model_data))  # Check if it's actually a tuple
5. Permission Issues
Make sure the script has permission to read the file:

sh
Copy
Edit
ls -l static/model.joblib
If needed, update permissions:

sh
Copy
Edit
chmod +r static/model.joblib
6. Try Ignoring Warnings
Add this before loading:

python
Copy
Edit
import warnings
warnings.simplefilter("ignore")
If none of these solutions work, let me know what error message (if any) is printed before exiting, and I can help debug further!






You said:
I got this error for part 1:
raceback (most recent call last):
  File "/Users/andrewrosenblum/VisualStudios Projects/HackAI Project/main.py", line 2, in <module>
    model_data = load("static/model.joblib")
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/andrewrosenblum/VisualStudios Projects/HackAI Project/venv/lib/python3.11/site-packages/joblib/numpy_pickle.py", line 587, in load
    obj = _unpickle(fobj, filename, mmap_mode)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/andrewrosenblum/VisualStudios Projects/HackAI Project/venv/lib/python3.11/site-packages/joblib/numpy_pickle.py", line 506, in _unpickle
    obj = unpickler.load()
          ^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/pickle.py", line 1213, in load
    dispatch[key[0]](self)
    ~~~~~~~~^^^^^^^^
KeyError: 0
ChatGPT said:
The KeyError: 0 when loading a .joblib file suggests that the model file is either:

Corrupted – The file may have been improperly saved.
Incompatible with your Python or scikit-learn version – If the file was created in a different environment, it might not deserialize correctly.
Not actually a joblib file – Something else may have been saved under that name.
🔍 How to Fix It
✅ 1. Check if the File is Corrupted
Try inspecting the file manually:

sh
Copy
Edit
file static/model.joblib
If it's not recognized as a valid binary file, it's likely corrupted.

✅ 2. Verify That It’s a Joblib File
Run this Python script:

python
Copy
Edit
with open("static/model.joblib", "rb") as f:
    print(f.read(10))  # Read first 10 bytes
If you get unreadable garbage or an empty response, the file might be broken.

✅ 3. Try Loading with Pickle Instead
Sometimes, .joblib files are actually just pickled objects. Try this:

python
Copy
Edit
import pickle

with open("static/model.joblib", "rb") as f:
    model_data = pickle.load(f)
    print(type(model_data))
If this works, the file was saved using pickle instead of joblib.

✅ 4. Ensure You're Using the Correct Joblib Version
Check your joblib version:

python
Copy
Edit
import joblib
print(joblib.__version__)
If outdated, update it:

sh
Copy
Edit
pip install --upgrade joblib
Then try reloading.

✅ 5. Make Sure You're Using the Same Scikit-Learn Version
If the model was saved with a different version of scikit-learn, it may not load properly.

Check your version:

python
Copy
Edit
import sklearn
print(sklearn.__version__)
If needed, try downgrading:

sh
Copy
Edit
pip install scikit-learn==X.Y.Z  # Replace with the version used to train the model
✅ 6. Re-Save the Model if You Have the Original Code
If possible, re-save the model in a fresh environment:

python
Copy
Edit
from joblib import dump

dump((model, label_encoder), "static/model.joblib")
Then try loading it again.

🚀 Final Steps
If none of these work, try re-downloading or re-generating the model file.
If you have access to the code that saved the model, confirm it was saved with joblib.dump().
If you need help, let me know how the model was saved! That will help narrow it down.





You said:
Does this qualify as unreadable garbage:
b'\x80\x04\x95z\x03\x00\x00\x00\x00\x00'
ChatGPT said:
No, that does not qualify as unreadable garbage. Instead, it looks like a valid pickled object.

The prefix b'\x80\x04' suggests that this file was saved using Python’s pickle module, rather than joblib. This means you should try loading it using pickle instead of joblib:

✅ Solution: Try Loading with Pickle
Modify your code to use pickle:

python
Copy
Edit
import pickle

with open("static/model.joblib", "rb") as f:
    model_data = pickle.load(f)

print(type(model_data))  # Check what it actually contains
🔍 Why Does This Happen?
Someone might have mistakenly saved the model using pickle.dump() instead of joblib.dump().
joblib internally uses pickle, but if the file was saved in an unexpected way, joblib.load() might fail.
If pickle.load() works and returns a tuple (model, label_encoder), then just replace your original loading code with:

python
Copy
Edit
import pickle

with open("static/model.joblib", "rb") as f:
    model, label_encoder = pickle.load(f)

print("Model successfully loaded!")
Let me know if this works! 🚀






You said:
For pickle I got this error:
  File "/Users/andrewrosenblum/VisualStudios Projects/HackAI Project/main.py", line 4, in <module>
    model_data = pickle.load(f)
                 ^^^^^^^^^^^^^^
_pickle.UnpicklingError: invalid load key, '\x0a'.
ChatGPT said:
The UnpicklingError: invalid load key, '\x0a' suggests that the file is not a valid pickle file or has been corrupted. The presence of '\x0a' (which is a newline character) at the start of the file suggests that:

The file was not saved correctly as a joblib or pickle object.
It might be a plain text file or something else entirely.
There was a file format mismatch during saving.
✅ How to Fix It
1️⃣ Check the File’s Contents
Try opening the file in a text editor (like VS Code or Sublime Text) and see if it contains readable text. Run:

sh
Copy
Edit
head -n 10 static/model.joblib
If you see human-readable text, the file was likely saved incorrectly.

2️⃣ Ensure It’s Actually a Joblib File
Instead of joblib.load(), try using joblib.load() with mmap_mode='r':

python
Copy
Edit
from joblib import load

try:
    model_data = load("static/model.joblib", mmap_mode='r')
    print(type(model_data))
except Exception as e:
    print(f"Error loading file: {e}")
If it still fails, the file is either corrupted or was not saved properly.

3️⃣ Check the File Type
Run this in your terminal:

sh
Copy
Edit
file static/model.joblib
If it says something like "ASCII text", the file was not saved as a proper model.

4️⃣ Re-Save the Model (if Possible)
If you have access to the original code that created model.joblib, re-save it correctly using:

python
Copy
Edit
from joblib import dump

dump((model, label_encoder), "static/model.joblib")
Then, reload it normally:

python
Copy
Edit
model, label_encoder = load("static/model.joblib")
🔥 Final Conclusion
The most likely cause is that static/model.joblib is either:

Corrupted
Not actually a joblib or pickle file
Saved incorrectly (e.g., as plain text or JSON)
Let me know what you find when checking the file! 🚀






You said:
Is this saving it correctly:
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import sklearn
import numpy as np
import cv2
import os
from joblib import dump, load  # Use joblib instead of pickle

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

# Load images and labels
x, y = load_images_from_folder("./compscitrainML")

# Encode class labels (convert text labels to numbers)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset (Stratified to keep class balance)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Define and train Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)  # Quick & effective
model.fit(xtrain, ytrain)

# Evaluate the model
train_accuracy = accuracy_score(ytrain, model.predict(xtrain))
test_accuracy = accuracy_score(ytest, model.predict(xtest))

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the trained model using joblib
dump((model, label_encoder, sklearn.__version__), "model.joblib")
print("Model saved as 'model.joblib'.")
ChatGPT said:
Yes, your code is saving the model correctly using joblib. However, there are a couple of improvements to ensure compatibility when loading it in the future.

✅ What’s Done Right
✔ Saving with joblib:
You're correctly using dump((model, label_encoder, sklearn.__version__), "model.joblib"), which stores:

The trained RandomForestClassifier model.
The LabelEncoder (important for decoding class labels).
The scikit-learn version (useful for avoiding compatibility issues).
✔ Flattening images properly:
This ensures that each image becomes a feature vector.

✔ Using train_test_split(stratify=y):
Good practice to keep class balance during training.

🛠 Possible Issue: File Path
Your loading code earlier tried static/model.joblib, but you're saving to "model.joblib". Ensure that you use the same file path in both saving and loading.

If you want to save in static/, change this:

python
Copy
Edit
dump((model, label_encoder, sklearn.__version__), "static/model.joblib")
🔄 Loading the Model Correctly
Since you've stored the scikit-learn version, make sure to load it like this:

python
Copy
Edit
from joblib import load
import sklearn

# Load the model
try:
    model, label_encoder, saved_sklearn_version = load("static/model.joblib")
    print(f"Model successfully loaded. Saved with scikit-learn {saved_sklearn_version}, using {sklearn.__version__}.")
    
    # Warn if the scikit-learn version is different
    if saved_sklearn_version != sklearn.__version__:
        print("⚠ Warning: The model was saved with a different scikit-learn version. Consider using:")
        print(f"pip install scikit-learn=={saved_sklearn_version}")
except Exception as e:
    print(f"Error loading model: {e}")
🚀 Final Fixes
Ensure Paths Match: Save and load from the same path ("static/model.joblib" if you intended to store it in static/).
Include scikit-learn Version: Already done, but check it when loading to avoid compatibility issues.
Handle File Errors Gracefully: Already handled with try-except.
With these fixes, your saving and loading process should work flawlessly! 🎯






You said:
Just checking it is fine that I move the model.joblib file right
ChatGPT said:
Yes, it's completely fine to move the model.joblib file to a different directory, as long as you update the file path accordingly when loading it.

✅ Things to Keep in Mind When Moving model.joblib
Update the File Path in Your Code

If you move the file to static/, update your loading code:
python
Copy
Edit
model, label_encoder, saved_sklearn_version = load("static/model.joblib")
If you move it elsewhere, just adjust the path accordingly.
Check File Permissions (if necessary)

If you move it to a protected directory, make sure your script has read access:
sh
Copy
Edit
chmod +r static/model.joblib
Avoid File Corruption

Moving the file via command line or manually (drag-and-drop) is fine.
If using a script, ensure it properly moves the file:
python
Copy
Edit
import shutil
shutil.move("model.joblib", "static/model.joblib")
Ensure the Model Loads from the Correct Directory

If your script is running from a different directory, use:
python
Copy
Edit
import os
model_path = os.path.join(os.getcwd(), "static/model.joblib")
model, label_encoder, saved_sklearn_version = load(model_path)
🚀 Final Answer: Yes, moving model.joblib is perfectly fine as long as you update the path when loading it! 🎯





You said:
When I did this I got error loading model: 0
from joblib import load
import sklearn

# Load the model
try:
    model, label_encoder, saved_sklearn_version = load("static/model.joblib")
    print(f"Model successfully loaded. Saved with scikit-learn {saved_sklearn_version}, using {sklearn.__version__}.")
    
    # Warn if the scikit-learn version is different
    if saved_sklearn_version != sklearn.__version__:
        print("⚠ Warning: The model was saved with a different scikit-learn version. Consider using:")
        print(f"pip install scikit-learn=={saved_sklearn_version}")
except Exception as e:
    print(f"Error loading model: {e}")
ChatGPT said:
Since you're still getting "Error loading model: 0", let's systematically debug the issue.

✅ Step 1: Confirm That the File Exists
Run this in your terminal or Python script:

sh
Copy
Edit
ls -l static/model.joblib
Or check in Python:

python
Copy
Edit
import os
print(os.path.exists("static/model.joblib"))  # Should print True if the file exists
If the file does not exist, you likely moved it incorrectly. In that case:

Ensure it's inside the static/ directory.
Use the correct relative or absolute path when loading it.
✅ Step 2: Check If the File is Readable
If the file exists but still won’t load, check its permissions:

sh
Copy
Edit
ls -l static/model.joblib
If you see something like -rw-r--r--, then the file is readable. Otherwise, fix permissions:

sh
Copy
Edit
chmod +r static/model.joblib
✅ Step 3: Verify the File’s Integrity
Check the file’s contents by loading it in a safe mode:

python
Copy
Edit
try:
    with open("static/model.joblib", "rb") as f:
        print(f.read(10))  # Read the first few bytes
except Exception as e:
    print(f"Error reading file: {e}")
If this fails or prints unreadable garbage, the file might be corrupted.

✅ Step 4: Try Loading Without Unpacking
Instead of:

python
Copy
Edit
model, label_encoder, saved_sklearn_version = load("static/model.joblib")
Try:

python
Copy
Edit
data = load("static/model.joblib")
print(type(data))
If this works, print data and check if it's actually a tuple. If not, maybe the file was saved incorrectly.

✅ Step 5: Check joblib and scikit-learn Versions
Mismatch in versions can cause deserialization issues. Run:

python
Copy
Edit
import joblib
import sklearn

print("joblib version:", joblib.__version__)
print("scikit-learn version:", sklearn.__version__)
If your scikit-learn version is different from when the model was saved, install the correct version:

sh
Copy
Edit
pip install scikit-learn==<saved_version>
🚀 Final Steps
Check that the file exists (os.path.exists()).
Verify it’s readable (ls -l or chmod +r).
Try loading without unpacking (data = load(...)).
Check joblib and scikit-learn versions.
Try these and let me know what you find! 🚀






You said:
When I did step 4 and did this:
from joblib import load
data = load("static/model.joblib")
print(type(data))
I got the following error:
Traceback (most recent call last):
  File "/Users/andrewrosenblum/VisualStudios Projects/HackAI Project/main.py", line 2, in <module>
    data = load("static/model.joblib")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/andrewrosenblum/VisualStudios Projects/HackAI Project/venv/lib/python3.11/site-packages/joblib/numpy_pickle.py", line 587, in load
    obj = _unpickle(fobj, filename, mmap_mode)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/andrewrosenblum/VisualStudios Projects/HackAI Project/venv/lib/python3.11/site-packages/joblib/numpy_pickle.py", line 506, in _unpickle
    obj = unpickler.load()
          ^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/pickle.py", line 1213, in load
    dispatch[key[0]](self)
    ~~~~~~~~^^^^^^^^
KeyError: 0
ChatGPT said:
The KeyError: 0 when loading the file suggests that static/model.joblib is corrupted or was not saved correctly.

✅ Step-by-Step Fix
1️⃣ Verify That the File is Actually a Joblib File
Run this command in your terminal to check the file type:

sh
Copy
Edit
file static/model.joblib
If it says something like "ASCII text" instead of "data" or "binary", then it's likely not a valid joblib file.
Alternatively, in Python:

python
Copy
Edit
import os

if os.path.exists("static/model.joblib"):
    print("File exists.")
else:
    print("File does not exist.")
If the file does not exist, it may not have been moved properly.

2️⃣ Try Opening the File in a Text Editor
If you see readable text, the file is not a valid joblib model.
If you see binary characters, it may still be valid but corrupted.
3️⃣ Re-Save the Model and Ensure It’s Saved Correctly
If you have access to the code that created the model, re-save it properly:

python
Copy
Edit
from joblib import dump

dump((model, label_encoder, sklearn.__version__), "static/model.joblib")
print("Model saved successfully.")
Then, try loading it again.

4️⃣ Make Sure the Model Was Saved With Joblib, Not Pickle
If someone used pickle.dump() instead of joblib.dump(), joblib.load() won’t work.

Try loading it with pickle instead:

python
Copy
Edit
import pickle

with open("static/model.joblib", "rb") as f:
    data = pickle.load(f)

print(type(data))  # Check what is inside the file
If this works, then the file was mistakenly saved with pickle.

5️⃣ Check joblib and scikit-learn Versions
If the model was trained with an older version of scikit-learn, loading it with a newer version may fail. Run:

python
Copy
Edit
import joblib
import sklearn

print("joblib version:", joblib.__version__)
print("scikit-learn version:", sklearn.__version__)
If the versions are different, try installing the correct version:

sh
Copy
Edit
pip install --upgrade joblib
pip install scikit-learn==<saved_version>
🔥 Conclusion
Since you're getting KeyError: 0, the most likely cause is:

The file is corrupted.
It was not saved correctly.
It was saved with pickle instead of joblib.
Try these fixes and let me know what happens! 🚀






You said:
The importing works in this model:
import cv2
import numpy as np
from joblib import load
import sklearn
import warnings

# Load the trained model
model, label_encoder, trained_version = load("model.joblib")

# Check current scikit-learn version
current_version = sklearn.__version__
print(f"Trained with scikit-learn version: {trained_version}, Current version: {current_version}")

# If versions mismatch, warn the user
if trained_version != current_version:
    warnings.warn(f"Model was trained with scikit-learn {trained_version}, but you are using {current_version}. "
                  "This may cause compatibility issues.")

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
But not in this code:
Traceback (most recent call last):
  File "/Users/andrewrosenblum/VisualStudios Projects/HackAI Project/main.py", line 3, in <module>
    model, label_encoder, trained_version = load("static/model.joblib")
                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/andrewrosenblum/VisualStudios Projects/HackAI Project/venv/lib/python3.11/site-packages/joblib/numpy_pickle.py", line 579, in load
    with open(filename, 'rb') as f:
         ^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'static/model.joblib'
Even though they us e the same sklearn module
ChatGPT said:
The error message "FileNotFoundError: [Errno 2] No such file or directory: 'static/model.joblib'" indicates that the file does not exist in the expected directory. The issue is not related to scikit-learn but rather to file path differences.

✅ How to Fix the Issue
1️⃣ Confirm the File Exists in the Expected Location
Run this in the terminal:

sh
Copy
Edit
ls -l static/model.joblib
Or check in Python:

python
Copy
Edit
import os
print(os.path.exists("static/model.joblib"))  # Should print True
If the file does not exist, it may not have been moved properly.
If the file exists elsewhere, adjust the file path.
2️⃣ Use an Absolute File Path Instead of a Relative Path
Try this:

python
Copy
Edit
import os
from joblib import load

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the model file
model_path = os.path.join(script_dir, "static", "model.joblib")

# Load the model
model, label_encoder, trained_version = load(model_path)
This ensures that the file is found regardless of the working directory.

3️⃣ Check the Current Working Directory
If you're running the script from a different directory, static/model.joblib may not be found. Run:

python
Copy
Edit
import os
print(os.getcwd())  # Print current working directory
If it's different from the expected directory, either change the working directory or use an absolute path.
To change the working directory in the terminal:

sh
Copy
Edit
cd /Users/andrewrosenblum/VisualStudios\ Projects/HackAI\ Project/
python main.py
4️⃣ Move the Model to the Expected Directory
If model.joblib is located in your main project folder but not inside static/, move it:

sh
Copy
Edit
mv model.joblib static/
Or rename the loading path in your script to just "model.joblib" if that's where the file is.

🚀 Final Fix
Try this absolute path approach to avoid issues:

python
Copy
Edit
import os
from joblib import load

# Get absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "static", "model.joblib")

# Load model
model, label_encoder, trained_version = load(model_path)
This ensures your script finds the model no matter where you run it from.

Let me know if this works! 🚀






You said:
Sorry this is the error I get:
  model, label_encoder, trained_version = load("model.joblib")
                                            ^^^^^^^^^^^^^^^^^^^^
  File "/Users/andrewrosenblum/VisualStudios Projects/HackAI Project/venv/lib/python3.11/site-packages/joblib/numpy_pickle.py", line 587, in load
    obj = _unpickle(fobj, filename, mmap_mode)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/andrewrosenblum/VisualStudios Projects/HackAI Project/venv/lib/python3.11/site-packages/joblib/numpy_pickle.py", line 506, in _unpickle
    obj = unpickler.load()
          ^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/pickle.py", line 1213, in load
    dispatch[key[0]](self)
    ~~~~~~~~^^^^^^^^
KeyError: 0
ChatGPT said:
The KeyError: 0 when loading model.joblib suggests that the file is corrupted or was not saved correctly.

✅ Step 1: Confirm That the File Exists and Is Valid
Run this in the terminal:

sh
Copy
Edit
ls -l model.joblib
If the file exists, check if it's a valid joblib file:

sh
Copy
Edit
file model.joblib
If it says something like "ASCII text", it's not a valid joblib file.
If it says "data" or "binary", it's likely valid.
✅ Step 2: Try Loading the File Without Unpacking
Instead of:

python
Copy
Edit
model, label_encoder, trained_version = load("model.joblib")
Try:

python
Copy
Edit
from joblib import load

data = load("model.joblib")
print(type(data))  # Check what it contains
If this works, then the problem is that the file doesn't contain a tuple.

✅ Step 3: Verify How the Model Was Saved
If the file was saved incorrectly, it may not be a (model, label_encoder, version) tuple.

Check how the model was originally saved. It should have been:

python
Copy
Edit
from joblib import dump

dump((model, label_encoder, sklearn.__version__), "model.joblib")
If it wasn’t saved this way, then load() won’t return a tuple.

✅ Step 4: Try Using Pickle
If joblib.load() fails, try pickle:

python
Copy
Edit
import pickle

with open("model.joblib", "rb") as f:
    data = pickle.load(f)

print(type(data))  # Check if it's a tuple
If this works, then the file was mistakenly saved with pickle instead of joblib.

✅ Step 5: Re-Save the Model (If Possible)
If you still have access to the original training script, re-save the model correctly:

python
Copy
Edit
from joblib import dump

dump((model, label_encoder, sklearn.__version__), "model.joblib")
print("Model saved successfully.")
Then try loading again:

python
Copy
Edit
model, label_encoder, trained_version = load("model.joblib")
🚀 Final Conclusion
The KeyError: 0 usually means:

The file is corrupted (fix: re-save it).
It was saved incorrectly (fix: check how it was saved).
It was saved with pickle instead of joblib (fix: use pickle.load()).
Try these steps and let me know what happens! 🚀






You said:
I think I need to update sklearn to 1.6.1
ChatGPT said:
Yes! If your model was trained with scikit-learn 1.6.1, but you're using a different version, that could be causing compatibility issues when loading the model.

✅ How to Upgrade scikit-learn to 1.6.1
Run this command in your terminal:

sh
Copy
Edit
pip install --upgrade scikit-learn==1.6.1
If you're using a virtual environment, make sure you're inside it before running the command:

sh
Copy
Edit
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate  # For Windows
pip install --upgrade scikit-learn==1.6.1
✅ Verify Installation
After updating, confirm the version:

python
Copy
Edit
import sklearn
print(sklearn.__version__)  # Should print 1.6.1
Then, try loading the model again:

python
Copy
Edit
from joblib import load

model, label_encoder, trained_version = load("model.joblib")
print(f"Model successfully loaded. Trained with scikit-learn {trained_version}, using {sklearn.__version__}.")
If this works, then the issue was indeed a version mismatch! 🚀 Let me know if you need further help!






You said:
In this code:
from flask import Flask, render_template, request, session, redirect, flash
from flask_session import Session
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import cv2
from joblib import dump, load
import sqlite3
import numpy as np
from PIL import Image
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
                model, label_encoder = load(model_path)
            else:
                model, label_encoder = load("static/model.joblib")

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

        # Create directory if not exists
        if not os.path.exists(user_train_path):
            os.makedirs(user_train_path)
            for clas in CLASSES:
                os.system(f"cp -R static/{clas} {user_train_path}")

        if os.path.exists(f"{user_train_path}/{classname}"):
            return render_template("error.html", message="Class is already being used")

        # Save uploaded images
        os.makedirs(f"{user_train_path}/{classname}")
        for image in request.files.getlist("images"):
            filename = image.filename.replace("/", "")
            save_path = f"{user_train_path}/{classname}/{filename}"
            image.save(save_path)

        # Load dataset
        x, y = load_images_from_folder(user_train_path)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        # Train Random Forest Model
        model = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42)
        model.fit(x, y)

        # Save model using joblib
        dump((model, label_encoder), f"{user_train_path}/data.joblib")

        # Update database
        db = sqlite3.connect("static/users.db")
        cursor = db.cursor()
        cursor.execute("INSERT INTO classes (id, class) VALUES (?, ?)", (session["user_id"], classname,))
        db.commit()

        flash("Training complete")
        return redirect("/")

# Logout Route
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# Run Flask App
if __name__ == "__main__":
    app.run(port=3000, debug=True)
I got an error when using predict:
Too many values to unpack expected 2
ChatGPT said:
The error "Too many values to unpack (expected 2)" occurs when loading the model using:

python
Copy
Edit
model, label_encoder = load(model_path)
🔍 Why is This Happening?
Your model was originally saved using dump((model, label_encoder, sklearn.__version__), "model.joblib").
This means the saved file contains three items:
(model, label_encoder, sklearn_version).
But when you load it, you're only expecting two values (model, label_encoder), which causes the unpacking error.
✅ Fix: Unpack the Correct Number of Values
Modify this line:

python
Copy
Edit
model, label_encoder = load(model_path)
To correctly unpack all three values:

python
Copy
Edit
model, label_encoder, trained_version = load(model_path)
Now, your script should correctly extract the model, label encoder, and scikit-learn version.

🚀 Final Fix in Your Code
Modify this part inside index():

python
Copy
Edit
# Load model
model_path = f"{DATASET_PATH}/train{session['user_id']}/data.joblib"
if os.path.exists(model_path):
    model, label_encoder, trained_version = load(model_path)  # ✅ Fix unpacking
else:
    model, label_encoder, trained_version = load("static/model.joblib")  # ✅ Fix unpacking

# Predict
yhat = model.predict(img_flattened)
predicted_class = label_encoder.inverse_transform(yhat)
🎯 Why This Fix Works
Ensures all three saved values (model, label_encoder, sklearn version) are properly unpacked.
Prevents errors due to mismatch in saved data format.
Now, try running the prediction again, and it should work! 🚀 Let me know if you still have issues.






You said:
After retraining this model with new data I got the following error:
from flask import Flask, render_template, request, session, redirect, flash
from flask_session import Session
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import cv2
from joblib import dump, load
import sqlite3
import numpy as np
from PIL import Image
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

        # Create directory if not exists
        if not os.path.exists(user_train_path):
            os.makedirs(user_train_path)
            for clas in CLASSES:
                os.system(f"cp -R static/{clas} {user_train_path}")

        if os.path.exists(f"{user_train_path}/{classname}"):
            return render_template("error.html", message="Class is already being used")

        # Save uploaded images
        os.makedirs(f"{user_train_path}/{classname}")
        for image in request.files.getlist("images"):
            filename = image.filename.replace("/", "")
            save_path = f"{user_train_path}/{classname}/{filename}"
            image.save(save_path)

        # Load dataset
        x, y = load_images_from_folder(user_train_path)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        # Train Random Forest Model
        model = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42)
        model.fit(x, y)

        # Save model using joblib
        dump((model, label_encoder), f"{user_train_path}/data.joblib")

        # Update database
        db = sqlite3.connect("static/users.db")
        cursor = db.cursor()
        cursor.execute("INSERT INTO classes (id, class) VALUES (?, ?)", (session["user_id"], classname,))
        db.commit()

        flash("Training complete")
        return redirect("/")

# Logout Route
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# Run Flask App
if __name__ == "__main__":
    app.run(port=3000, debug=True)
not enough values to unpack (expected 3, got 2)
ChatGPT said:
🔍 Why Are You Getting This Error?
The error "not enough values to unpack (expected 3, got 2)" means that when loading the model with:

python
Copy
Edit
model, label_encoder, trained_version = load(model_path)
the file only contains 2 items, but your code expects 3 (model, label_encoder, and trained_version).

✅ Fix: Check How the Model Was Saved
In your training route, you are saving the model like this:

python
Copy
Edit
dump((model, label_encoder), f"{user_train_path}/data.joblib")
This means that the file only contains 2 items (not 3). But your prediction code expects 3 items:

python
Copy
Edit
model, label_encoder, trained_version = load(model_path)  # ❌ This expects 3 items, but only 2 exist
🔧 Solution: Load Only 2 Items
Since you only saved 2 items, modify your loading code to only expect 2:

python
Copy
Edit
model, label_encoder = load(model_path)  # ✅ Load only 2 items
And update the else block:

python
Copy
Edit
model, label_encoder = load("static/model.joblib")  # ✅ Load only 2 items
🛠 Alternative Fix: Save 3 Items
If you want to keep the trained version and avoid changing the loading code, update your saving method in the training route to include sklearn.__version__:

python
Copy
Edit
import sklearn
dump((model, label_encoder, sklearn.__version__), f"{user_train_path}/data.joblib")  # ✅ Save 3 items
This way, the file will contain (model, label_encoder, trained_version), and your existing code will work.

🎯 Final Fix Summary
If you want to load only 2 items, update the loading code:
python
Copy
Edit
model, label_encoder = load(model_path)  # ✅ Expect only 2 values
If you want to keep 3 items, update the saving code:
python
Copy
Edit
dump((model, label_encoder, sklearn.__version__), f"{user_train_path}/data.joblib")  # ✅ Save 3 items
Let me know if you need more help! 🚀






You said:
Can u fix the training because currently it is not working and is only training on the most recent thing:
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

        # Create directory if not exists
        if not os.path.exists(user_train_path):
            os.makedirs(user_train_path)
            for clas in CLASSES:
                os.system(f"cp -R static/{clas} {user_train_path}")

        if os.path.exists(f"{user_train_path}/{classname}"):
            return render_template("error.html", message="Class is already being used")

        # Save uploaded images
        os.makedirs(f"{user_train_path}/{classname}")
        for image in request.files.getlist("images"):
            filename = image.filename.replace("/", "")
            save_path = f"{user_train_path}/{classname}/{filename}"
            image.save(save_path)

        # Load dataset
        x, y = load_images_from_folder(user_train_path)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        # Train Random Forest Model
        model = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42)
        model.fit(x, y)

        dump((model, label_encoder, sklearn.__version__), f"{user_train_path}/data.joblib")

        # Update database
        db = sqlite3.connect("static/users.db")
        cursor = db.cursor()
        cursor.execute("INSERT INTO classes (id, class) VALUES (?, ?)", (session["user_id"], classname,))
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
# Logout Route
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# Run Flask App
if __name__ == "__main__":
    app.run(port=3000, debug=True)
ChatGPT said:
🔍 Issue: Training Overwrites the Model Instead of Updating It
Right now, when you train a new class, the model only trains on the most recently added images, not all previously saved data. This is happening because the dataset is reloaded from scratch each time:

python
Copy
Edit
x, y = load_images_from_folder(user_train_path)
This means only the newly added images are used for training, while previous training data is lost.

✅ Solution: Load the Existing Model If It Exists
To preserve previously trained data, modify the training logic to:

Check if a trained model exists.
Load and extend the existing dataset.
Retrain using both old and new data.
🔧 Fixed Training Code
Replace your train() function with this updated version:

python
Copy
Edit
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
🚀 How This Fix Works
Checks if an existing model is available

If yes, it loads the existing model and label encoder.
If no, it creates a new model and encoder.
Extends the dataset

It keeps previously trained data and adds new images.
Retrains with all available data

Instead of training on just the new class, it trains with all previous and new images.
Saves the updated model

Ensures the model retains past training knowledge.
🔥 Final Result
Your model remembers previously trained data instead of overwriting it.
New classes get added dynamically.
The label encoder updates automatically to handle new classes.
The model will grow as new data is added.
Now, every time a user adds a new class, all old data is still included, and the model gets smarter instead of resetting! 🚀 Let me know if you need any modifications! 😊






You said:
I have the following code how do I easily publish it to a website:
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
ChatGPT said:
🚀 How to Easily Deploy Your Flask App to a Website
There are multiple ways to publish your Flask app online. I'll outline a simple and effective approach based on your needs:

✅ Option 1: Deploy on Render (Easiest Free Hosting)
Render is one of the simplest ways to host your Flask app.

Steps to Deploy on Render
Push Your Code to GitHub

If you haven’t already, initialize a Git repository:
sh
Copy
Edit
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
Create a New Web Service on Render

Go to Render
Click New Web Service
Connect your GitHub repository
Select Flask as the framework
Set Up Environment Variables

In Render’s dashboard, add any required environment variables (like SECRET_KEY if needed).
Specify Start Command

Render needs to know how to start your Flask app.
Add this command in Render's settings:
sh
Copy
Edit
gunicorn -w 4 -b 0.0.0.0:10000 app:app
(Replace app.py with your main script name if needed.)
Deploy & Test Your Website

Click Deploy, and within a few minutes, your Flask app will be live.
✅ Option 2: Deploy on PythonAnywhere (Good for Small Apps)
If you don’t need real-time training and just want the app online, PythonAnywhere is a solid free choice.

Steps for PythonAnywhere
Create a PythonAnywhere Account (Signup Here)
Upload Your Flask Project
Use PythonAnywhere’s file manager or push your project using Git.
Set Up a Web App
Go to the Web tab.
Click Add a new web app → Choose Flask.
Point to Your Flask App
Set the correct working directory and wsgi.py entry point.
Restart the Server
Click Reload Web App.
Your Flask app will be live at yourusername.pythonanywhere.com 🎉

✅ Option 3: Deploy on Heroku (Good for Database-Heavy Apps)
If you need a database (like PostgreSQL) and scalability, Heroku is great.

Steps for Heroku
Install the Heroku CLI

sh
Copy
Edit
brew tap heroku/brew && brew install heroku  # Mac
sudo snap install --classic heroku           # Linux
Login to Heroku

sh
Copy
Edit
heroku login
Create a New Heroku App

sh
Copy
Edit
heroku create your-app-name
Deploy Your Code

sh
Copy
Edit
git add .
git commit -m "Deploying Flask app"
git push heroku main
Scale the App

sh
Copy
Edit
heroku ps:scale web=1
View Your Website

sh
Copy
Edit
heroku open
✅ Which One Should You Choose?
Hosting Service	Best For	Ease of Setup	Free Plan?
Render	Best for all-around free hosting	⭐⭐⭐⭐⭐ (Easiest)	✅ Yes
PythonAnywhere	Simple, good for small apps	⭐⭐⭐	✅ Yes (limited)
Heroku	More control, scalable	⭐⭐⭐⭐	✅ Yes (with limits)
Final Steps to Make Deployment Easier
Ensure You Have a requirements.txt File Run this command:

sh
Copy
Edit
pip freeze > requirements.txt
This tells the server which packages to install.

Ensure You Have a Procfile (For Render/Heroku) Create a Procfile and add:

makefile
Copy
Edit
web: gunicorn -w 4 -b 0.0.0.0:10000 app:app
(Replace app.py with your script name.)

Check Static File Paths If using Render or Heroku, make sure:

python
Copy
Edit
app.config['STATIC_FOLDER'] = 'static'
is correctly set.

🎯 Conclusion: Best for You
If you're new to deployment, Render is the easiest option.

If you just need a simple free app, PythonAnywhere works fine.

If you want full control, Heroku is a good option.

Let me know which method you choose, and I can guide you through it! 🚀






You said:
Show me how to do it with heroku
ChatGPT said:
🚀 Step-by-Step Guide to Deploy Your Flask App on Heroku
Heroku is a great way to deploy Flask apps with free hosting. Follow these simple steps to get your app online.

✅ Step 1: Install Heroku CLI
You need the Heroku CLI to deploy your app. Install it based on your OS:

MacOS (Homebrew)
sh
Copy
Edit
brew tap heroku/brew && brew install heroku
Windows (via Installer) Download the Heroku CLI
Linux (Snap)
sh
Copy
Edit
sudo snap install --classic heroku
Verify Installation:

sh
Copy
Edit
heroku --version
You should see output like:

bash
Copy
Edit
heroku/x.y.z (darwin-arm64) go1.17.1
✅ Step 2: Log In to Heroku
Open your terminal and log in:

sh
Copy
Edit
heroku login
This will open a browser where you authorize access.

✅ Step 3: Set Up Your Flask Project for Heroku
3.1 Create a Virtual Environment (Optional but Recommended)
Run:

sh
Copy
Edit
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate      # Windows
3.2 Install Required Packages
Ensure your Flask app has all dependencies installed:

sh
Copy
Edit
pip install flask gunicorn joblib numpy scikit-learn pillow opencv-python sqlite3
3.3 Create a requirements.txt File
Heroku needs this to install dependencies:

sh
Copy
Edit
pip freeze > requirements.txt
3.4 Create a Procfile (Tells Heroku How to Run the App)
In your project root, create a file called Procfile (no file extension) and add:

makefile
Copy
Edit
web: gunicorn -w 4 -b 0.0.0.0:10000 app:app
Replace app.py with your Flask script name (e.g., main.py → app:main).
Gunicorn is used to serve your Flask app in production.
3.5 Create a .gitignore File
Add:

bash
Copy
Edit
venv/
__pycache__/
*.pyc
instance/
.env
This prevents unnecessary files from being pushed to Heroku.

✅ Step 4: Initialize Git and Push to Heroku
Make sure your project is in a Git repository:

sh
Copy
Edit
git init
git add .
git commit -m "Initial commit for Heroku deployment"
Create a new Heroku app:

sh
Copy
Edit
heroku create your-app-name
(Replace your-app-name with a unique name, or let Heroku generate one)

Push the project to Heroku:

sh
Copy
Edit
git push heroku main
✅ Step 5: Scale & Run the App
Scale the app to one dyno (Heroku's unit of computation):

sh
Copy
Edit
heroku ps:scale web=1
Open the app in a browser:

sh
Copy
Edit
heroku open
✅ Step 6: (Optional) Set Up a Database on Heroku
If your app uses SQLite, Heroku does not support SQLite directly. You need a PostgreSQL database.

Add PostgreSQL to Your App:

sh
Copy
Edit
heroku addons:create heroku-postgresql:hobby-dev
Get Database URL:

sh
Copy
Edit
heroku config
You’ll see an entry like:

bash
Copy
Edit
DATABASE_URL=postgres://username:password@host:port/dbname
Modify Your Flask App to Use PostgreSQL

Install PostgreSQL support:
sh
Copy
Edit
pip install psycopg2-binary
Change SQLite connection to:
python
Copy
Edit
import os
import psycopg2
DATABASE_URL = os.environ['DATABASE_URL']
conn = psycopg2.connect(DATABASE_URL, sslmode="require")
✅ Step 7: View Logs & Debug (If Needed)
If something goes wrong, check the logs:

sh
Copy
Edit
heroku logs --tail
🎯 Final Summary
Step	Command
Login to Heroku	heroku login
Initialize Git	git init && git add . && git commit -m "Deploy"
Create Heroku App	heroku create your-app-name
Add Requirements File	pip freeze > requirements.txt
Create Procfile	echo "web: gunicorn -w 4 -b 0.0.0.0:10000 app:app" > Procfile
Push to Heroku	git push heroku main
Scale App	heroku ps:scale web=1
Open App in Browser	heroku open
Check Logs for Errors	heroku logs --tail
Now your Flask app is live on Heroku! 🎉 Let me know if you run into any issues. 🚀






You said:
I got this error when doing git push heroku main:
11.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/_internal.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/common/__init__.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/common/__pycache__/__init__.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/common/__pycache__/_aliases.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/common/__pycache__/_fft.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/common/__pycache__/_helpers.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/common/__pycache__/_linalg.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/common/__pycache__/_typing.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/common/_aliases.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/common/_fft.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/common/_helpers.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/common/_linalg.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/common/_typing.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/cupy/__init__.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/cupy/__pycache__/__init__.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/cupy/__pycache__/_aliases.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/cupy/__pycache__/_info.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/cupy/__pycache__/_typing.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/cupy/__pycache__/fft.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/cupy/__pycache__/linalg.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/cupy/_aliases.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/cupy/_info.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/cupy/_typing.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/cupy/fft.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/cupy/linalg.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/dask/__pycache__/__init__.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/dask/array/__init__.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/dask/array/__pycache__/__init__.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/dask/array/__pycache__/_aliases.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/dask/array/__pycache__/_info.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/dask/array/__pycache__/fft.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/dask/array/__pycache__/linalg.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/dask/array/_aliases.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/dask/array/_info.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/dask/array/fft.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/dask/array/linalg.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/numpy/__init__.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/numpy/__pycache__/__init__.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/numpy/__pycache__/_aliases.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/numpy/__pycache__/_info.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/numpy/__pycache__/_typing.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/numpy/__pycache__/fft.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/numpy/__pycache__/linalg.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/numpy/_aliases.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/numpy/_info.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/numpy/_typing.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/numpy/fft.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/numpy/linalg.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/torch/__init__.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/torch/__pycache__/__init__.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/torch/__pycache__/_aliases.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/torch/__pycache__/_info.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/torch/__pycache__/fft.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/torch/__pycache__/linalg.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/torch/_aliases.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/torch/_info.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/torch/fft.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_compat/torch/linalg.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_extra/__init__.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_extra/__pycache__/__init__.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_extra/__pycache__/_funcs.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_extra/__pycache__/_typing.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_extra/_funcs.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/array_api_extra/_typing.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/__init__.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/__pycache__/__init__.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/__pycache__/framework.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/__pycache__/main.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/__pycache__/models.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/__pycache__/problem.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/__pycache__/settings.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/framework.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/main.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/models.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/problem.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/settings.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/subsolvers/__init__.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/subsolvers/__pycache__/__init__.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/subsolvers/__pycache__/geometry.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/subsolvers/__pycache__/optim.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/subsolvers/geometry.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/subsolvers/optim.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/utils/__init__.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/utils/__pycache__/__init__.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/utils/__pycache__/exceptions.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/utils/__pycache__/math.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/utils/__pycache__/versions.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/utils/exceptions.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/utils/math.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/cobyqa/utils/versions.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/tests/__pycache__/test_array_api.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/tests/__pycache__/test_config.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/_lib/tests/test_array_api.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/differentiate/__init__.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/differentiate/__pycache__/__init__.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/differentiate/__pycache__/_differentiate.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/differentiate/_differentiate.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/differentiate/tests/__pycache__/__init__.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/differentiate/tests/__pycache__/test_differentiate.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/differentiate/tests/test_differentiate.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/fft/__pycache__/_basic_backend.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/fft/__pycache__/_fftlog_backend.cpython-311.pyc
 create mode 100644 venv/lib/python3.11/site-packages/scipy/fft/__pycache__/_fftlog_multimethods.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/fft/__pycache__/_realtransforms_backend.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/fft/_basic_backend.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/fft/_fftlog_backend.py
 create mode 100644 venv/lib/python3.11/site-packages/scipy/fft/_fftlog_multimethods.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/fft/_realtransforms_backend.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/fft/tests/__pycache__/test_basic.cpython-311.pyc
 create mode 100644 venv/lib/python3.11/site-packages/scipy/fft/tests/__pycache__/test_fft_function.cpython-311.pyc
 create mode 100644 venv/lib/python3.11/site-packages/scipy/fft/tests/__pycache__/test_numpy.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/fft/tests/test_basic.py
 create mode 100644 venv/lib/python3.11/site-packages/scipy/fft/tests/test_fft_function.py
 create mode 100644 venv/lib/python3.11/site-packages/scipy/fft/tests/test_numpy.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/integrate/__pycache__/_cubature.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/integrate/__pycache__/_lebedev.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/integrate/__pycache__/_tanhsinh.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/integrate/_cubature.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/integrate/_lebedev.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/integrate/_rules/__init__.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/integrate/_rules/__pycache__/__init__.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/integrate/_rules/__pycache__/_base.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/integrate/_rules/__pycache__/_gauss_kronrod.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/integrate/_rules/__pycache__/_gauss_legendre.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/integrate/_rules/__pycache__/_genz_malik.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/integrate/_rules/_base.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/integrate/_rules/_gauss_kronrod.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/integrate/_rules/_gauss_legendre.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/integrate/_rules/_genz_malik.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/integrate/_tanhsinh.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/integrate/tests/__pycache__/test_cubature.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/integrate/tests/__pycache__/test_tanhsinh.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/integrate/tests/test_cubature.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/integrate/tests/test_tanhsinh.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/interpolate/__pycache__/_bary_rational.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/interpolate/__pycache__/_fitpack_repro.cpython-311.pyc
 create mode 100644 venv/lib/python3.11/site-packages/scipy/interpolate/__pycache__/_interpnd_info.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/interpolate/__pycache__/_ndbspline.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/interpolate/__pycache__/dfitpack.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/interpolate/__pycache__/interpnd.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/interpolate/_bary_rational.py
 delete mode 100755 venv/lib/python3.11/site-packages/scipy/interpolate/_dfitpack.cpython-311-darwin.so
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/interpolate/_fitpack_repro.py
 delete mode 100755 venv/lib/python3.11/site-packages/scipy/interpolate/_interpnd.cpython-311-darwin.so
 create mode 100644 venv/lib/python3.11/site-packages/scipy/interpolate/_interpnd_info.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/interpolate/_ndbspline.py
 create mode 100755 venv/lib/python3.11/site-packages/scipy/interpolate/dfitpack.cpython-311-darwin.so
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/interpolate/dfitpack.py
 create mode 100755 venv/lib/python3.11/site-packages/scipy/interpolate/interpnd.cpython-311-darwin.so
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/interpolate/interpnd.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/interpolate/tests/__pycache__/test_bary_rational.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/interpolate/tests/test_bary_rational.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/io/_fast_matrix_market/__init__.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/io/_fast_matrix_market/__pycache__/__init__.cpython-311.pyc
 delete mode 100755 venv/lib/python3.11/site-packages/scipy/io/_fast_matrix_market/_fmm_core.cpython-311-darwin.so
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/io/matlab/tests/data/debigged_m4.mat
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/io/tests/data/test-44100Hz-le-1ch-4bytes-rf64.wav
 rename venv/lib/python3.11/site-packages/scipy/io/tests/data/{test-1234Hz-le-1ch-10S-20bit-extra.wav => test-8000Hz-le-1ch-10S-20bit-extra.wav} (100%)
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/io/tests/data/test-8000Hz-le-3ch-5S-24bit-rf64.wav
 create mode 100644 venv/lib/python3.11/site-packages/scipy/linalg.pxd
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/linalg/__init__.pxd
 create mode 100644 venv/lib/python3.11/site-packages/scipy/linalg/__pycache__/_flinalg_py.cpython-311.pyc
 create mode 100644 venv/lib/python3.11/site-packages/scipy/linalg/__pycache__/_interpolative_backend.cpython-311.pyc
 create mode 100644 venv/lib/python3.11/site-packages/scipy/linalg/__pycache__/flinalg.cpython-311.pyc
 create mode 100644 venv/lib/python3.11/site-packages/scipy/linalg/_blas_subroutine_wrappers.f
 delete mode 100755 venv/lib/python3.11/site-packages/scipy/linalg/_decomp_interpolative.cpython-311-darwin.so
 delete mode 100755 venv/lib/python3.11/site-packages/scipy/linalg/_decomp_lu_cython.cpython-311-darwin.so
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/linalg/_decomp_lu_cython.pyi
 create mode 100755 venv/lib/python3.11/site-packages/scipy/linalg/_flinalg.cpython-311-darwin.so
 create mode 100644 venv/lib/python3.11/site-packages/scipy/linalg/_flinalg_py.py
 create mode 100755 venv/lib/python3.11/site-packages/scipy/linalg/_interpolative.cpython-311-darwin.so
 create mode 100644 venv/lib/python3.11/site-packages/scipy/linalg/_interpolative_backend.py
 create mode 100644 venv/lib/python3.11/site-packages/scipy/linalg/_lapack_subroutine_wrappers.f
 delete mode 100755 venv/lib/python3.11/site-packages/scipy/linalg/_linalg_pythran.cpython-311-darwin.so
 create mode 100644 venv/lib/python3.11/site-packages/scipy/linalg/flinalg.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/linalg/tests/__pycache__/test_decomp_lu.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/linalg/tests/__pycache__/test_extending.cpython-311.pyc
 create mode 100644 venv/lib/python3.11/site-packages/scipy/linalg/tests/__pycache__/test_misc.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/linalg/tests/_cython_examples/extending.pyx
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/linalg/tests/_cython_examples/meson.build
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/linalg/tests/test_decomp_lu.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/linalg/tests/test_extending.py
 create mode 100644 venv/lib/python3.11/site-packages/scipy/linalg/tests/test_misc.py
 create mode 100644 venv/lib/python3.11/site-packages/scipy/misc/__pycache__/_common.cpython-311.pyc
 create mode 100644 venv/lib/python3.11/site-packages/scipy/misc/_common.py
 create mode 100644 venv/lib/python3.11/site-packages/scipy/misc/ascent.dat
 create mode 100644 venv/lib/python3.11/site-packages/scipy/misc/ecg.dat
 create mode 100644 venv/lib/python3.11/site-packages/scipy/misc/face.dat
 rename venv/lib/python3.11/site-packages/scipy/{optimize/_highspy => misc/tests}/__init__.py (100%)
 create mode 100644 venv/lib/python3.11/site-packages/scipy/misc/tests/__pycache__/__init__.cpython-311.pyc
 create mode 100644 venv/lib/python3.11/site-packages/scipy/misc/tests/__pycache__/test_common.cpython-311.pyc
 create mode 100644 venv/lib/python3.11/site-packages/scipy/misc/tests/__pycache__/test_config.cpython-311.pyc
 rename venv/lib/python3.11/site-packages/scipy/{_lib => misc}/tests/__pycache__/test_doccer.cpython-311.pyc (55%)
 create mode 100644 venv/lib/python3.11/site-packages/scipy/misc/tests/test_common.py
 rename venv/lib/python3.11/site-packages/scipy/{_lib => misc}/tests/test_config.py (97%)
 rename venv/lib/python3.11/site-packages/scipy/{_lib => misc}/tests/test_doccer.py (87%)
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/ndimage/__pycache__/_delegators.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/ndimage/__pycache__/_ndimage_api.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/ndimage/__pycache__/_support_alternative_backends.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/ndimage/_delegators.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/ndimage/_ndimage_api.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/ndimage/_support_alternative_backends.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/ndimage/tests/__pycache__/test_ni_support.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/ndimage/tests/test_ni_support.py
 rename venv/lib/python3.11/site-packages/scipy/{optimize/__init__.pxd => optimize.pxd} (100%)
 create mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/README
 rename venv/lib/python3.11/site-packages/scipy/{interpolate/_dierckx.cpython-311-darwin.so => optimize/__nnls.cpython-311-darwin.so} (55%)
 create mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/__nnls.pyi
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/__pycache__/_bracket.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/__pycache__/_chandrupatla.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/__pycache__/_cobyqa_py.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/__pycache__/_dcsrch.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/__pycache__/_elementwise.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/__pycache__/_isotonic.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/__pycache__/elementwise.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_bracket.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_chandrupatla.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_cobyqa_py.py
 delete mode 100755 venv/lib/python3.11/site-packages/scipy/optimize/_cython_nnls.cpython-311-darwin.so
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_dcsrch.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_elementwise.py
 rename venv/lib/python3.11/site-packages/scipy/{special/tests/data => optimize/_highs}/__init__.py (100%)
 create mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_highs/__pycache__/__init__.cpython-311.pyc
 rename venv/lib/python3.11/site-packages/scipy/{ndimage/_rank_filter_1d.cpython-311-darwin.so => optimize/_highs/_highs_constants.cpython-311-darwin.so} (59%)
 create mode 100755 venv/lib/python3.11/site-packages/scipy/optimize/_highs/_highs_wrapper.cpython-311-darwin.so
 create mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_highs/src/cython/HConst.pxd
 create mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_highs/src/cython/Highs.pxd
 create mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_highs/src/cython/HighsIO.pxd
 create mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_highs/src/cython/HighsInfo.pxd
 create mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_highs/src/cython/HighsLp.pxd
 create mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_highs/src/cython/HighsLpUtils.pxd
 create mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_highs/src/cython/HighsModelUtils.pxd
 create mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_highs/src/cython/HighsOptions.pxd
 create mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_highs/src/cython/HighsRuntimeOptions.pxd
 create mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_highs/src/cython/HighsStatus.pxd
 create mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_highs/src/cython/SimplexConst.pxd
 create mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_highs/src/cython/highs_c_api.pxd
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_highspy/__pycache__/__init__.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_highspy/__pycache__/_highs_wrapper.cpython-311.pyc
 delete mode 100755 venv/lib/python3.11/site-packages/scipy/optimize/_highspy/_core.cpython-311-darwin.so
 delete mode 100755 venv/lib/python3.11/site-packages/scipy/optimize/_highspy/_highs_options.cpython-311-darwin.so
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_highspy/_highs_wrapper.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_isotonic.py
 rename venv/lib/python3.11/site-packages/{numpy/_core/_rational_tests.cpython-311-darwin.so => scipy/optimize/_minpack2.cpython-311-darwin.so} (50%)
 delete mode 100755 venv/lib/python3.11/site-packages/scipy/optimize/_pava_pybind.cpython-311-darwin.so
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_shgo_lib/__pycache__/_complex.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_shgo_lib/__pycache__/_vertex.cpython-311.pyc
 create mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_shgo_lib/__pycache__/triangulation.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_shgo_lib/_complex.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_shgo_lib/_vertex.py
 create mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_shgo_lib/triangulation.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_trustregion_constr/tests/__pycache__/test_nested_minimize.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/_trustregion_constr/tests/test_nested_minimize.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/elementwise.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/tests/__pycache__/test_bracket.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/tests/__pycache__/test_chandrupatla.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/tests/__pycache__/test_cobyqa.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/tests/__pycache__/test_extending.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/tests/__pycache__/test_isotonic_regression.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/tests/_cython_examples/extending.pyx
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/tests/_cython_examples/meson.build
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/tests/test_bracket.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/tests/test_chandrupatla.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/tests/test_cobyqa.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/tests/test_extending.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/optimize/tests/test_isotonic_regression.py
 create mode 100644 venv/lib/python3.11/site-packages/scipy/signal/__pycache__/_bsplines.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/signal/__pycache__/_short_time_fft.cpython-311.pyc
 create mode 100644 venv/lib/python3.11/site-packages/scipy/signal/__pycache__/_spectral.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/signal/__pycache__/_spline_filters.cpython-311.pyc
 create mode 100644 venv/lib/python3.11/site-packages/scipy/signal/_bsplines.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/signal/_short_time_fft.py
 rename venv/lib/python3.11/site-packages/{numpy/_core/_umath_tests.cpython-311-darwin.so => scipy/signal/_spectral.cpython-311-darwin.so} (58%)
 create mode 100644 venv/lib/python3.11/site-packages/scipy/signal/_spectral.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/signal/_spline.pyi
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/signal/_spline_filters.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/signal/tests/__pycache__/_scipy_spectral_test_shim.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/signal/tests/__pycache__/test_short_time_fft.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/signal/tests/__pycache__/test_splines.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/signal/tests/_scipy_spectral_test_shim.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/signal/tests/test_short_time_fft.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/signal/tests/test_splines.py
 create mode 100644 venv/lib/python3.11/site-packages/scipy/sparse/__pycache__/_arrays.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/sparse/__pycache__/_matrix.cpython-311.pyc
 create mode 100644 venv/lib/python3.11/site-packages/scipy/sparse/_arrays.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/sparse/_matrix.py
 create mode 100644 venv/lib/python3.11/site-packages/scipy/sparse/csgraph/__pycache__/setup.cpython-311.pyc
 create mode 100644 venv/lib/python3.11/site-packages/scipy/sparse/csgraph/setup.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/sparse/csgraph/tests/__pycache__/test_pydata_sparse.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/sparse/csgraph/tests/test_pydata_sparse.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/sparse/linalg/__pycache__/_special_sparse_arrays.cpython-311.pyc
 create mode 100755 venv/lib/python3.11/site-packages/scipy/sparse/linalg/_isolve/_iterative.cpython-311-darwin.so
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/sparse/linalg/_special_sparse_arrays.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/sparse/linalg/tests/__pycache__/test_special_sparse_arrays.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/sparse/linalg/tests/test_special_sparse_arrays.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/sparse/tests/__pycache__/test_arithmetic1d.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/sparse/tests/__pycache__/test_common1d.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/sparse/tests/__pycache__/test_coo.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/sparse/tests/__pycache__/test_dok.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/sparse/tests/__pycache__/test_indexing1d.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/sparse/tests/__pycache__/test_minmax1d.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/sparse/tests/test_arithmetic1d.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/sparse/tests/test_common1d.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/sparse/tests/test_coo.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/sparse/tests/test_dok.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/sparse/tests/test_indexing1d.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/sparse/tests/test_minmax1d.py
 create mode 100644 venv/lib/python3.11/site-packages/scipy/spatial/_ckdtree.pyi
 create mode 100644 venv/lib/python3.11/site-packages/scipy/spatial/transform/_rotation.pyi
 create mode 100644 venv/lib/python3.11/site-packages/scipy/special.pxd
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/__init__.pxd
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/__pycache__/_input_validation.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/__pycache__/_multiufuncs.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/__pycache__/_support_alternative_backends.cpython-311.pyc
 delete mode 100755 venv/lib/python3.11/site-packages/scipy/special/_gufuncs.cpython-311-darwin.so
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/_input_validation.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/_multiufuncs.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/_precompute/__pycache__/hyp2f1_data.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/_precompute/hyp2f1_data.py
 delete mode 100755 venv/lib/python3.11/site-packages/scipy/special/_special_ufuncs.cpython-311-darwin.so
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/_support_alternative_backends.py
 create mode 100644 venv/lib/python3.11/site-packages/scipy/special/cython_special.pyx
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/tests/__pycache__/test_boost_ufuncs.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/tests/__pycache__/test_cephes_intp_cast.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/tests/__pycache__/test_extending.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/tests/__pycache__/test_iv_ratio.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/tests/__pycache__/test_legendre.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/tests/__pycache__/test_specfun.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/tests/__pycache__/test_support_alternative_backends.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/tests/__pycache__/test_ufunc_signatures.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/tests/__pycache__/test_xsf_cuda.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/tests/_cython_examples/extending.pyx
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/tests/_cython_examples/meson.build
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/tests/data/__pycache__/__init__.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/tests/test_boost_ufuncs.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/tests/test_cephes_intp_cast.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/tests/test_extending.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/tests/test_iv_ratio.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/tests/test_legendre.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/tests/test_specfun.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/tests/test_support_alternative_backends.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/tests/test_ufunc_signatures.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/tests/test_xsf_cuda.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/binom.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cdflib.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/airy.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/besselpoly.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/beta.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/cbrt.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/chbevl.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/chdtr.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/const.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/ellie.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/ellik.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/ellpe.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/ellpk.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/expn.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/gamma.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/hyp2f1.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/hyperg.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/i0.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/i1.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/igam.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/igam_asymp_coeff.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/igami.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/j0.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/j1.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/jv.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/k0.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/k1.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/kn.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/lanczos.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/ndtr.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/poch.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/polevl.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/psi.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/rgamma.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/scipy_iv.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/shichi.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/sici.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/sindg.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/tandg.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/trig.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/unity.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/cephes/zeta.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/config.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/digamma.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/error.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/evalpoly.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/expint.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/hyp2f1.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/iv_ratio.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/lambertw.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/loggamma.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/sici.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/tools.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/trig.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/wright_bessel.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/special/xsf/zlog1.h
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_bws_test.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_censored_data.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_correlation.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_distribution_infrastructure.cpython-311.pyc
 create mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_generate_pyx.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_mgc.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_multicomp.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_new_distributions.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_probability_distribution.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_qmvnt.cpython-311.pyc
 create mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_rvs_sampling.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_sampling.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_sensitivity_analysis.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_survival.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_wilcoxon.cpython-311.pyc
 create mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/statlib.cpython-311.pyc
 delete mode 100755 venv/lib/python3.11/site-packages/scipy/stats/_ansari_swilk_statistics.cpython-311-darwin.so
 create mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_boost/__init__.py
 create mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_boost/__pycache__/__init__.cpython-311.pyc
 create mode 100755 venv/lib/python3.11/site-packages/scipy/stats/_boost/beta_ufunc.cpython-311-darwin.so
 create mode 100755 venv/lib/python3.11/site-packages/scipy/stats/_boost/binom_ufunc.cpython-311-darwin.so
 create mode 100755 venv/lib/python3.11/site-packages/scipy/stats/_boost/hypergeom_ufunc.cpython-311-darwin.so
 create mode 100755 venv/lib/python3.11/site-packages/scipy/stats/_boost/invgauss_ufunc.cpython-311-darwin.so
 create mode 100755 venv/lib/python3.11/site-packages/scipy/stats/_boost/nbinom_ufunc.cpython-311-darwin.so
 create mode 100755 venv/lib/python3.11/site-packages/scipy/stats/_boost/ncf_ufunc.cpython-311-darwin.so
 create mode 100755 venv/lib/python3.11/site-packages/scipy/stats/_boost/nct_ufunc.cpython-311-darwin.so
 create mode 100755 venv/lib/python3.11/site-packages/scipy/stats/_boost/ncx2_ufunc.cpython-311-darwin.so
 create mode 100755 venv/lib/python3.11/site-packages/scipy/stats/_boost/skewnorm_ufunc.cpython-311-darwin.so
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_bws_test.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_censored_data.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_correlation.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_distribution_infrastructure.py
 create mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_generate_pyx.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_mgc.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_multicomp.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_new_distributions.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_probability_distribution.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_qmvnt.py
 create mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_rvs_sampling.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_sampling.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_sensitivity_analysis.py
 create mode 100755 venv/lib/python3.11/site-packages/scipy/stats/_statlib.cpython-311-darwin.so
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_survival.py
 create mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_unuran/unuran.pxd
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_wilcoxon.py
 create mode 100644 venv/lib/python3.11/site-packages/scipy/stats/statlib.py
 create mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/__pycache__/test_boost_ufuncs.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/__pycache__/test_censored_data.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/__pycache__/test_continuous.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/__pycache__/test_continuous_fit_censored.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/__pycache__/test_correlation.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/__pycache__/test_fast_gen_inversion.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/__pycache__/test_mgc.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/__pycache__/test_multicomp.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/__pycache__/test_sensitivity_analysis.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/__pycache__/test_survival.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/data/__pycache__/_mvt.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/data/_mvt.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/data/jf_skew_t_gamlss_pdf_data.npy
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/data/rel_breitwigner_pdf_sample_data_ROOT.npy
 create mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/test_boost_ufuncs.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/test_censored_data.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/test_continuous.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/test_continuous_fit_censored.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/test_correlation.py
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % python -u "/Users/andrewrosenblum/VisualStudios Pro
jects/HackAI Project/app.py"
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:3000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 117-048-725
127.0.0.1 - - [22/Feb/2025 17:10:20] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:10:20] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:10:25] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:10:25] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:10:26] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:10:26] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:11:47] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:11:47] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:11:56] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:11:56] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:12:02] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:12:02] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:12:03] "GET /train HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:12:03] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:12:03] "GET /static/loading.gif HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:12:32] "POST /train HTTP/1.1" 302 -
127.0.0.1 - - [22/Feb/2025 17:12:32] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:12:32] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:12:42] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:12:42] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:12:50] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:12:50] "GET /static/styles.css HTTP/1.1" 304 -
 * Detected change in '/Users/andrewrosenblum/VisualStudios Projects/HackAI Project/app.py', reloading
 * Restarting with stat
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % python -u "/Users/andrewrosenblum/VisualStudios Pro
jects/HackAI Project/app.py"
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:3000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 117-048-725
127.0.0.1 - - [22/Feb/2025 17:15:28] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:15:28] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:15:33] "GET /reset HTTP/1.1" 404 -
127.0.0.1 - - [22/Feb/2025 17:15:35] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:15:36] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:15:36] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:15:36] "GET /train HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:15:36] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:15:36] "GET /static/loading.gif HTTP/1.1" 200 -
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % python -u "/Users/andrewrosenblum/VisualStudios Pro
jects/HackAI Project/app.py"
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:3000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 117-048-725
127.0.0.1 - - [22/Feb/2025 17:18:40] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:18:40] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:18:44] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:18:44] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:18:45] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:18:45] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:18:46] "GET /train HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:18:46] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:18:46] "GET /static/loading.gif HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:18:57] "POST /train HTTP/1.1" 302 -
127.0.0.1 - - [22/Feb/2025 17:18:57] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:18:57] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:19:02] "POST / HTTP/1.1" 200 -
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % python -u "/Users/andrewrosenblum/VisualStudios Pro
jects/HackAI Project/app.py"
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:3000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 117-048-725
127.0.0.1 - - [22/Feb/2025 17:21:03] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:21:03] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:21:07] "GET /reset HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:21:07] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:21:09] "POST /reset HTTP/1.1" 302 -
127.0.0.1 - - [22/Feb/2025 17:21:09] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:21:09] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:21:13] "GET /train HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:21:13] "GET /static/styles.css HTTP/1.1" 304 -
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % python -u "/Users/andrewrosenblum/VisualStudios Pro
jects/HackAI Project/app.py"
Traceback (most recent call last):
  File "/Users/andrewrosenblum/VisualStudios Projects/HackAI Project/app.py", line 140, in <module>
    @app.route("/train", methods=["GET", "POST"])
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/andrewrosenblum/VisualStudios Projects/HackAI Project/venv/lib/python3.11/site-packages/flask/sansio/scaffold.py", line 362, in decorator
    self.add_url_rule(rule, endpoint, f, **options)
  File "/Users/andrewrosenblum/VisualStudios Projects/HackAI Project/venv/lib/python3.11/site-packages/flask/sansio/scaffold.py", line 47, in wrapper_func
    return f(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/andrewrosenblum/VisualStudios Projects/HackAI Project/venv/lib/python3.11/site-packages/flask/sansio/app.py", line 657, in add_url_rule
    raise AssertionError(
AssertionError: View function mapping is overwriting an existing endpoint function: train
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % python -u "/Users/andrewrosenblum/VisualStudios Pro
jects/HackAI Project/app.py"
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:3000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 117-048-725
127.0.0.1 - - [22/Feb/2025 17:23:33] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:23:33] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:23:38] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:23:38] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:23:39] "GET /reset HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:23:39] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:23:40] "POST /reset HTTP/1.1" 302 -
127.0.0.1 - - [22/Feb/2025 17:23:40] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:23:40] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:23:47] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:23:47] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:23:48] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:23:48] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:23:52] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:23:52] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:23:54] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:23:54] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:23:55] "GET /train HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:23:55] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:23:55] "GET /static/loading.gif HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:24:15] "POST /train HTTP/1.1" 302 -
127.0.0.1 - - [22/Feb/2025 17:24:15] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:24:15] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:24:21] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:24:21] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:24:22] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:24:22] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:24:33] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:24:33] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:24:34] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:24:34] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:26:53] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:26:53] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:26:55] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:26:55] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:27:06] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:27:06] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:27:08] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:27:08] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:27:19] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:27:19] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:27:20] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:27:20] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:27:31] "GET /reset HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:27:31] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:27:32] "POST /reset HTTP/1.1" 302 -
127.0.0.1 - - [22/Feb/2025 17:27:32] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:27:32] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:28:14] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:28:14] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:28:20] "GET /favicon.ico HTTP/1.1" 404 -
127.0.0.1 - - [22/Feb/2025 17:28:26] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:28:26] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:28:28] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:28:28] "GET /static/styles.css HTTP/1.1" 304 -
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % git add .
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % git commit -m "working project"
[main f1c48d9] working project
 38 files changed, 89 insertions(+), 15 deletions(-)
 create mode 100644 static/datasets/train13/Akita/akitaakita_0.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_1.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_10.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_11.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_12.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_13.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_14.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_15.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_16.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_17.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_18.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_19.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_2.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_3.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_4.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_5.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_6.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_7.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_8.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_9.jpg
 create mode 100644 static/datasets/train13/Sai/.DS_Store
 create mode 100644 static/datasets/train13/Sai/Sai_10.jpg
 create mode 100644 static/datasets/train13/Sai/Sai_11.jpg
 create mode 100644 static/datasets/train13/Sai/Sai_12.jpg
 create mode 100644 static/datasets/train13/Sai/Sai_2.jpg
 create mode 100644 static/datasets/train13/Sai/Sai_3.jpg
 create mode 100644 static/datasets/train13/Sai/Sai_4.jpg
 create mode 100644 static/datasets/train13/Sai/Sai_5.jpg
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % heroku login
heroku: Press any key to open up the browser to login or q to exit: 
Opening browser to https://cli-auth.heroku.com/auth/cli/browser/92ad3dcc-ace5-4f86-955e-94039b3454cd?requestor=SFMyNTY.g2gDbQAAAA4xNDAuMjU0Ljc3LjIyNW4GAHj9GDCVAWIAAVGA.Tm1AfIVexEATzy9AU-aCSVpgvutqQykfs3jbCxkmM2Q
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % heroku create website-classification
Creating ⬢ website-classification... done
https://website-classification-d6f350ecbf10.herokuapp.com/ | https://git.heroku.com/website-classification.git
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % git push heroku main
Enumerating objects: 12395, done.
Counting objects: 100% (12395/12395), done.
Delta compression using up to 8 threads
Compressing objects: 100% (9021/9021), done.
Writing objects: 100% (12395/12395), 222.35 MiB | 16.85 MiB/s, done.
Total 12395 (delta 3263), reused 12362 (delta 3254), pack-reused 0
remote: Resolving deltas: 100% (3263/3263), done.
remote: Updated 7440 paths from 16190dd
remote: Compressing source files... done.
remote: Building source:
remote: 
remote: -----> Building on the Heroku-24 stack
remote: -----> Determining which buildpack to use for this app
remote: -----> Python app detected
remote: -----> No Python version was specified. Using the buildpack default: Python 3.13
remote:        To use a different version, see: https://devcenter.heroku.com/articles/python-runtimes
remote: -----> Installing Python 3.13.2
remote: -----> Installing pip 24.3.1
remote: -----> Installing dependencies using 'pip install -r requirements.txt'
remote:        Collecting blinker==1.9.0 (from -r requirements.txt (line 1))
remote:          Downloading blinker-1.9.0-py3-none-any.whl.metadata (1.6 kB)
remote:        Collecting cachelib==0.13.0 (from -r requirements.txt (line 2))
remote:          Downloading cachelib-0.13.0-py3-none-any.whl.metadata (2.0 kB)
remote:        Collecting click==8.1.8 (from -r requirements.txt (line 3))
remote:          Downloading click-8.1.8-py3-none-any.whl.metadata (2.3 kB)
remote:        Collecting Flask==3.1.0 (from -r requirements.txt (line 4))
remote:          Downloading flask-3.1.0-py3-none-any.whl.metadata (2.7 kB)
remote:        Collecting Flask-Session==0.8.0 (from -r requirements.txt (line 5))
remote:          Downloading flask_session-0.8.0-py3-none-any.whl.metadata (5.2 kB)
remote:        Collecting itsdangerous==2.2.0 (from -r requirements.txt (line 6))
remote:          Downloading itsdangerous-2.2.0-py3-none-any.whl.metadata (1.9 kB)
remote:        Collecting Jinja2==3.1.5 (from -r requirements.txt (line 7))
remote:          Downloading jinja2-3.1.5-py3-none-any.whl.metadata (2.6 kB)
remote:        Collecting joblib==1.4.2 (from -r requirements.txt (line 8))
remote:          Downloading joblib-1.4.2-py3-none-any.whl.metadata (5.4 kB)
remote:        Collecting MarkupSafe==3.0.2 (from -r requirements.txt (line 9))
remote:          Downloading MarkupSafe-3.0.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.0 kB)
remote:        Collecting msgspec==0.19.0 (from -r requirements.txt (line 10))
remote:          Downloading msgspec-0.19.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.9 kB)
remote:        Collecting numpy==1.23.5 (from -r requirements.txt (line 11))
remote:          Downloading numpy-1.23.5.tar.gz (10.7 MB)
remote:          Installing build dependencies: started
remote:          Installing build dependencies: finished with status 'done'
remote:          Getting requirements to build wheel: started
remote:          Getting requirements to build wheel: finished with status 'done'
remote:        ERROR: Exception:
remote:        Traceback (most recent call last):
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/cli/base_command.py", line 105, in _run_wrapper
remote:            status = _inner_run()
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/cli/base_command.py", line 96, in _inner_run
remote:            return self.run(options, args)
remote:                   ~~~~~~~~^^^^^^^^^^^^^^^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/cli/req_command.py", line 67, in wrapper
remote:            return func(self, options, args)
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/commands/install.py", line 379, in run
remote:            requirement_set = resolver.resolve(
remote:                reqs, check_supported_wheels=not options.target_dir
remote:            )
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/resolver.py", line 95, in resolve
remote:            result = self._result = resolver.resolve(
remote:                                    ~~~~~~~~~~~~~~~~^
remote:                collected.requirements, max_rounds=limit_how_complex_resolution_can_be
remote:                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
remote:            )
remote:            ^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_vendor/resolvelib/resolvers.py", line 546, in resolve
remote:            state = resolution.resolve(requirements, max_rounds=max_rounds)
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_vendor/resolvelib/resolvers.py", line 397, in resolve
remote:            self._add_to_criteria(self.state.criteria, r, parent=None)
remote:            ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_vendor/resolvelib/resolvers.py", line 173, in _add_to_criteria
remote:            if not criterion.candidates:
remote:                   ^^^^^^^^^^^^^^^^^^^^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_vendor/resolvelib/structs.py", line 156, in __bool__
remote:            return bool(self._sequence)
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/found_candidates.py", line 174, in __bool__
remote:            return any(self)
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/found_candidates.py", line 162, in <genexpr>
remote:            return (c for c in iterator if id(c) not in self._incompatible_ids)
remote:                               ^^^^^^^^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/found_candidates.py", line 53, in _iter_built
remote:            candidate = func()
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/factory.py", line 187, in _make_candidate_from_link
remote:            base: Optional[BaseCandidate] = self._make_base_candidate_from_link(
remote:                                            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
remote:                link, template, name, version
remote:                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
remote:            )
remote:            ^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/factory.py", line 233, in _make_base_candidate_from_link
remote:            self._link_candidate_cache[link] = LinkCandidate(
remote:                                               ~~~~~~~~~~~~~^
remote:                link,
remote:                ^^^^^
remote:            ...<3 lines>...
remote:                version=version,
remote:                ^^^^^^^^^^^^^^^^
remote:            )
remote:            ^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/candidates.py", line 304, in __init__
remote:            super().__init__(
remote:            ~~~~~~~~~~~~~~~~^
remote:                link=link,
remote:                ^^^^^^^^^^
remote:            ...<4 lines>...
remote:                version=version,
remote:                ^^^^^^^^^^^^^^^^
remote:            )
remote:            ^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/candidates.py", line 159, in __init__
remote:            self.dist = self._prepare()
remote:                        ~~~~~~~~~~~~~^^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/candidates.py", line 236, in _prepare
remote:            dist = self._prepare_distribution()
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/candidates.py", line 315, in _prepare_distribution
remote:            return preparer.prepare_linked_requirement(self._ireq, parallel_builds=True)
remote:                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/operations/prepare.py", line 527, in prepare_linked_requirement
remote:            return self._prepare_linked_requirement(req, parallel_builds)
remote:                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/operations/prepare.py", line 642, in _prepare_linked_requirement
remote:            dist = _get_prepared_distribution(
remote:                req,
remote:            ...<3 lines>...
remote:                self.check_build_deps,
remote:            )
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/operations/prepare.py", line 72, in _get_prepared_distribution
remote:            abstract_dist.prepare_distribution_metadata(
remote:            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
remote:                finder, build_isolation, check_build_deps
remote:                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
remote:            )
remote:            ^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/distributions/sdist.py", line 56, in prepare_distribution_metadata
remote:            self._install_build_reqs(finder)
remote:            ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/distributions/sdist.py", line 126, in _install_build_reqs
remote:            build_reqs = self._get_build_requires_wheel()
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/distributions/sdist.py", line 103, in _get_build_requires_wheel
remote:            return backend.get_requires_for_build_wheel()
remote:                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/utils/misc.py", line 701, in get_requires_for_build_wheel
remote:            return super().get_requires_for_build_wheel(config_settings=cs)
remote:                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_vendor/pyproject_hooks/_impl.py", line 166, in get_requires_for_build_wheel
remote:            return self._call_hook('get_requires_for_build_wheel', {
remote:                   ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
remote:                'config_settings': config_settings
remote:                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
remote:            })
remote:            ^^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_vendor/pyproject_hooks/_impl.py", line 321, in _call_hook
remote:            raise BackendUnavailable(data.get('traceback', ''))
remote:        pip._vendor.pyproject_hooks._impl.BackendUnavailable: Traceback (most recent call last):
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 77, in _build_backend
remote:            obj = import_module(mod_path)
remote:          File "/app/.heroku/python/lib/python3.13/importlib/__init__.py", line 88, in import_module
remote:            return _bootstrap._gcd_import(name[level:], package, level)
remote:                   ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
remote:          File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
remote:          File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
remote:          File "<frozen importlib._bootstrap>", line 1310, in _find_and_load_unlocked
remote:          File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
remote:          File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
remote:          File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
remote:          File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
remote:          File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
remote:          File "<frozen importlib._bootstrap_external>", line 1026, in exec_module
remote:          File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
remote:          File "/tmp/pip-build-env-o9dh0g3n/overlay/lib/python3.13/site-packages/setuptools/__init__.py", line 10, in <module>
remote:            import distutils.core
remote:        ModuleNotFoundError: No module named 'distutils'
remote:        
remote: 
remote:  !     Error: Unable to install dependencies using pip.
remote:  !     
remote:  !     See the log output above for more information.
remote: 
remote:  !     Push rejected, failed to compile Python app.
remote: 
remote:  !     Push failed
remote: Verifying deploy...
remote: 
remote: !       Push rejected to website-classification.
remote: 
To https://git.heroku.com/website-classification.git
 ! [remote rejected] main -> main (pre-receive hook declined)
error: failed to push some refs to 'https://git.heroku.com/website-classification.git'
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % 
ChatGPT said:
Fixing the Heroku Deployment Issues
You're running into a dependency installation issue on Heroku, specifically with NumPy and distutils. Let's go step by step to fix it.

✅ Step 1: Specify a Compatible Python Version
Heroku is defaulting to Python 3.13, which is causing compatibility issues. Let's explicitly set Python 3.10 or 3.11 (which work well with Flask and NumPy).

Modify or Create runtime.txt in your project root and add:

Copy
Edit
python-3.10.12
or

Copy
Edit
python-3.11.6
Then commit the change:

sh
Copy
Edit
git add runtime.txt
git commit -m "Specify Python version"
✅ Step 2: Install distutils
Heroku's Python 3.13 lacks distutils, which is needed for NumPy. Add it explicitly:

Edit requirements.txt and add:

nginx
Copy
Edit
setuptools
Then commit:

sh
Copy
Edit
git add requirements.txt
git commit -m "Added setuptools to fix distutils issue"
✅ Step 3: Upgrade pip, setuptools, and wheel
Locally, before re-deploying, run:

sh
Copy
Edit
pip install --upgrade pip setuptools wheel
pip freeze > requirements.txt
Then commit the updated requirements.txt:

sh
Copy
Edit
git add requirements.txt
git commit -m "Upgrade pip, setuptools, and wheel"
✅ Step 4: Ensure NumPy Compatibility
Your logs show that NumPy 1.23.5 failed. Change it to a stable version:

Modify requirements.txt:

shell
Copy
Edit
numpy>=1.24
Then commit:

sh
Copy
Edit
git add requirements.txt
git commit -m "Updated NumPy version"
✅ Step 5: Push Again to Heroku
Now retry the deployment:

sh
Copy
Edit
git push heroku main
✅ Bonus: Set the Correct Start Command in Procfile
Ensure your Procfile contains:

makefile
Copy
Edit
web: gunicorn -w 4 -b 0.0.0.0:10000 app:app
Then commit:

sh
Copy
Edit
git add Procfile
git commit -m "Updated Procfile"
Final Summary
Specify Python version (runtime.txt)
Add setuptools to requirements.txt
Upgrade pip, setuptools, and wheel
Ensure NumPy is compatible (numpy>=1.24)
Retry pushing to Heroku
Now, run:

sh
Copy
Edit
git push heroku main
This should work! 🚀 Let me know if you hit any other issues.






You said:
It heroku commiting works but when I open the webpage it immediately crashes here are the logs:
2025-02-23T00:08:14.504088+00:00 app[web.1]: Python buildpack: Detected 512 MB available memory and 8 CPU cores.
2025-02-23T00:08:14.504161+00:00 app[web.1]: Python buildpack: Defaulting WEB_CONCURRENCY to 2 based on the available memory.
2025-02-23T00:08:14.527320+00:00 app[web.1]: /bin/bash: line 1: gunicorn: command not found
2025-02-23T00:08:14.573353+00:00 heroku[web.1]: Process exited with status 127
2025-02-23T00:08:14.595281+00:00 heroku[web.1]: State changed from starting to crashed
2025-02-23T00:08:15.000000+00:00 app[api]: Build succeeded
2025-02-23T00:08:16.163684+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/" host=website-classification-d6f350ecbf10.herokuapp.com request_id=774e759a-5218-4334-8860-12027c43234c fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:08:16.520666+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/favicon.ico" host=website-classification-d6f350ecbf10.herokuapp.com request_id=66b99f2f-4f4c-4975-af1f-dec4832676ef fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:08:19.755229+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/" host=website-classification-d6f350ecbf10.herokuapp.com request_id=687b871b-3d4c-405e-8a07-7460c9868a6c fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:08:19.899551+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/favicon.ico" host=website-classification-d6f350ecbf10.herokuapp.com request_id=e4507cb5-3835-4e88-a060-8c8be2a18502 fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:08:21.069831+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/" host=website-classification-d6f350ecbf10.herokuapp.com request_id=5957c772-4b06-495e-b078-8d43917bec78 fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:08:21.217781+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/favicon.ico" host=website-classification-d6f350ecbf10.herokuapp.com request_id=bc96b55f-15fe-4622-8367-3106d073aa1e fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:08:21.883574+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/" host=website-classification-d6f350ecbf10.herokuapp.com request_id=6cdee750-ad45-4d14-ad6d-4131650abac6 fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:08:22.086609+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/favicon.ico" host=website-classification-d6f350ecbf10.herokuapp.com request_id=430f9b8d-f259-4382-9329-f3f6e17e6f65 fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:08:23.367218+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/" host=website-classification-d6f350ecbf10.herokuapp.com request_id=24be13a0-b9ce-457a-87df-47211be89f48 fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:08:23.590221+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/favicon.ico" host=website-classification-d6f350ecbf10.herokuapp.com request_id=6ec7418a-1d69-49d7-9027-f73507142678 fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:08:24.296210+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/" host=website-classification-d6f350ecbf10.herokuapp.com request_id=b3ca441e-ad9e-4de2-8d8c-403744de69ad fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:08:24.496179+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/favicon.ico" host=website-classification-d6f350ecbf10.herokuapp.com request_id=d2d9c1c9-f52e-409e-be36-9b95ec939cf7 fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:08:32.607968+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/" host=website-classification-d6f350ecbf10.herokuapp.com request_id=f8495a14-1dc6-4cdc-9515-77a57a42b981 fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:08:32.747277+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/favicon.ico" host=website-classification-d6f350ecbf10.herokuapp.com request_id=17589894-cc6c-46ee-8463-5e77b831d503 fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
ChatGPT said:
Fixing Heroku "App Crashed" Issue (Gunicorn Missing)
Your error log shows that Gunicorn is not found (gunicorn: command not found). This means your app is crashing because Gunicorn is missing or not installed.

✅ Step 1: Install Gunicorn
Ensure Gunicorn is in your requirements.txt:

Run:

sh
Copy
Edit
pip install gunicorn
pip freeze > requirements.txt
Then commit and push:

sh
Copy
Edit
git add requirements.txt
git commit -m "Added Gunicorn to requirements"
git push heroku main
✅ Step 2: Check Your Procfile
Ensure you have a valid Procfile in the root of your project.

Run:

sh
Copy
Edit
echo "web: gunicorn app:app" > Procfile
git add Procfile
git commit -m "Fixed Procfile for Heroku"
git push heroku main
✅ Step 3: Restart the App
sh
Copy
Edit
heroku restart
✅ Step 4: Check Logs for Errors
sh
Copy
Edit
heroku logs --tail
If Gunicorn is installed and your Procfile is correct, your Flask app should work 🚀!

Try opening the app:

sh
Copy
Edit
heroku open
Let me know if you need more help!






You said:
I now get this error:
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_censored_data.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_correlation.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_distribution_infrastructure.cpython-311.pyc
 create mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_generate_pyx.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_mgc.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_multicomp.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_new_distributions.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_probability_distribution.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_qmvnt.cpython-311.pyc
 create mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_rvs_sampling.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_sampling.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_sensitivity_analysis.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_survival.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/_wilcoxon.cpython-311.pyc
 create mode 100644 venv/lib/python3.11/site-packages/scipy/stats/__pycache__/statlib.cpython-311.pyc
 delete mode 100755 venv/lib/python3.11/site-packages/scipy/stats/_ansari_swilk_statistics.cpython-311-darwin.so
 create mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_boost/__init__.py
 create mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_boost/__pycache__/__init__.cpython-311.pyc
 create mode 100755 venv/lib/python3.11/site-packages/scipy/stats/_boost/beta_ufunc.cpython-311-darwin.so
 create mode 100755 venv/lib/python3.11/site-packages/scipy/stats/_boost/binom_ufunc.cpython-311-darwin.so
 create mode 100755 venv/lib/python3.11/site-packages/scipy/stats/_boost/hypergeom_ufunc.cpython-311-darwin.so
 create mode 100755 venv/lib/python3.11/site-packages/scipy/stats/_boost/invgauss_ufunc.cpython-311-darwin.so
 create mode 100755 venv/lib/python3.11/site-packages/scipy/stats/_boost/nbinom_ufunc.cpython-311-darwin.so
 create mode 100755 venv/lib/python3.11/site-packages/scipy/stats/_boost/ncf_ufunc.cpython-311-darwin.so
 create mode 100755 venv/lib/python3.11/site-packages/scipy/stats/_boost/nct_ufunc.cpython-311-darwin.so
 create mode 100755 venv/lib/python3.11/site-packages/scipy/stats/_boost/ncx2_ufunc.cpython-311-darwin.so
 create mode 100755 venv/lib/python3.11/site-packages/scipy/stats/_boost/skewnorm_ufunc.cpython-311-darwin.so
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_bws_test.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_censored_data.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_correlation.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_distribution_infrastructure.py
 create mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_generate_pyx.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_mgc.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_multicomp.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_new_distributions.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_probability_distribution.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_qmvnt.py
 create mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_rvs_sampling.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_sampling.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_sensitivity_analysis.py
 create mode 100755 venv/lib/python3.11/site-packages/scipy/stats/_statlib.cpython-311-darwin.so
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_survival.py
 create mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_unuran/unuran.pxd
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/_wilcoxon.py
 create mode 100644 venv/lib/python3.11/site-packages/scipy/stats/statlib.py
 create mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/__pycache__/test_boost_ufuncs.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/__pycache__/test_censored_data.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/__pycache__/test_continuous.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/__pycache__/test_continuous_fit_censored.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/__pycache__/test_correlation.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/__pycache__/test_fast_gen_inversion.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/__pycache__/test_mgc.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/__pycache__/test_multicomp.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/__pycache__/test_sensitivity_analysis.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/__pycache__/test_survival.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/data/__pycache__/_mvt.cpython-311.pyc
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/data/_mvt.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/data/jf_skew_t_gamlss_pdf_data.npy
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/data/rel_breitwigner_pdf_sample_data_ROOT.npy
 create mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/test_boost_ufuncs.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/test_censored_data.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/test_continuous.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/test_continuous_fit_censored.py
 delete mode 100644 venv/lib/python3.11/site-packages/scipy/stats/tests/test_correlation.py
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % python -u "/Users/andrewrosenblum/VisualStudios Pro
jects/HackAI Project/app.py"
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:3000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 117-048-725
127.0.0.1 - - [22/Feb/2025 17:10:20] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:10:20] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:10:25] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:10:25] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:10:26] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:10:26] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:11:47] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:11:47] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:11:56] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:11:56] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:12:02] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:12:02] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:12:03] "GET /train HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:12:03] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:12:03] "GET /static/loading.gif HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:12:32] "POST /train HTTP/1.1" 302 -
127.0.0.1 - - [22/Feb/2025 17:12:32] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:12:32] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:12:42] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:12:42] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:12:50] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:12:50] "GET /static/styles.css HTTP/1.1" 304 -
 * Detected change in '/Users/andrewrosenblum/VisualStudios Projects/HackAI Project/app.py', reloading
 * Restarting with stat
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % python -u "/Users/andrewrosenblum/VisualStudios Pro
jects/HackAI Project/app.py"
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:3000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 117-048-725
127.0.0.1 - - [22/Feb/2025 17:15:28] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:15:28] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:15:33] "GET /reset HTTP/1.1" 404 -
127.0.0.1 - - [22/Feb/2025 17:15:35] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:15:36] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:15:36] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:15:36] "GET /train HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:15:36] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:15:36] "GET /static/loading.gif HTTP/1.1" 200 -
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % python -u "/Users/andrewrosenblum/VisualStudios Pro
jects/HackAI Project/app.py"
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:3000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 117-048-725
127.0.0.1 - - [22/Feb/2025 17:18:40] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:18:40] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:18:44] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:18:44] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:18:45] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:18:45] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:18:46] "GET /train HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:18:46] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:18:46] "GET /static/loading.gif HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:18:57] "POST /train HTTP/1.1" 302 -
127.0.0.1 - - [22/Feb/2025 17:18:57] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:18:57] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:19:02] "POST / HTTP/1.1" 200 -
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % python -u "/Users/andrewrosenblum/VisualStudios Pro
jects/HackAI Project/app.py"
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:3000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 117-048-725
127.0.0.1 - - [22/Feb/2025 17:21:03] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:21:03] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:21:07] "GET /reset HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:21:07] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:21:09] "POST /reset HTTP/1.1" 302 -
127.0.0.1 - - [22/Feb/2025 17:21:09] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:21:09] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:21:13] "GET /train HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:21:13] "GET /static/styles.css HTTP/1.1" 304 -
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % python -u "/Users/andrewrosenblum/VisualStudios Pro
jects/HackAI Project/app.py"
Traceback (most recent call last):
  File "/Users/andrewrosenblum/VisualStudios Projects/HackAI Project/app.py", line 140, in <module>
    @app.route("/train", methods=["GET", "POST"])
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/andrewrosenblum/VisualStudios Projects/HackAI Project/venv/lib/python3.11/site-packages/flask/sansio/scaffold.py", line 362, in decorator
    self.add_url_rule(rule, endpoint, f, **options)
  File "/Users/andrewrosenblum/VisualStudios Projects/HackAI Project/venv/lib/python3.11/site-packages/flask/sansio/scaffold.py", line 47, in wrapper_func
    return f(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/andrewrosenblum/VisualStudios Projects/HackAI Project/venv/lib/python3.11/site-packages/flask/sansio/app.py", line 657, in add_url_rule
    raise AssertionError(
AssertionError: View function mapping is overwriting an existing endpoint function: train
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % python -u "/Users/andrewrosenblum/VisualStudios Pro
jects/HackAI Project/app.py"
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:3000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 117-048-725
127.0.0.1 - - [22/Feb/2025 17:23:33] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:23:33] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:23:38] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:23:38] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:23:39] "GET /reset HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:23:39] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:23:40] "POST /reset HTTP/1.1" 302 -
127.0.0.1 - - [22/Feb/2025 17:23:40] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:23:40] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:23:47] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:23:47] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:23:48] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:23:48] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:23:52] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:23:52] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:23:54] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:23:54] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:23:55] "GET /train HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:23:55] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:23:55] "GET /static/loading.gif HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:24:15] "POST /train HTTP/1.1" 302 -
127.0.0.1 - - [22/Feb/2025 17:24:15] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:24:15] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:24:21] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:24:21] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:24:22] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:24:22] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:24:33] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:24:33] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:24:34] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:24:34] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:26:53] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:26:53] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:26:55] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:26:55] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:27:06] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:27:06] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:27:08] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:27:08] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:27:19] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:27:19] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:27:20] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:27:20] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:27:31] "GET /reset HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:27:31] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:27:32] "POST /reset HTTP/1.1" 302 -
127.0.0.1 - - [22/Feb/2025 17:27:32] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:27:32] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:28:14] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:28:14] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:28:20] "GET /favicon.ico HTTP/1.1" 404 -
127.0.0.1 - - [22/Feb/2025 17:28:26] "POST / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:28:26] "GET /static/styles.css HTTP/1.1" 304 -
127.0.0.1 - - [22/Feb/2025 17:28:28] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [22/Feb/2025 17:28:28] "GET /static/styles.css HTTP/1.1" 304 -
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % git add .
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % git commit -m "working project"
[main f1c48d9] working project
 38 files changed, 89 insertions(+), 15 deletions(-)
 create mode 100644 static/datasets/train13/Akita/akitaakita_0.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_1.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_10.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_11.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_12.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_13.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_14.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_15.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_16.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_17.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_18.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_19.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_2.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_3.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_4.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_5.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_6.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_7.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_8.jpg
 create mode 100644 static/datasets/train13/Akita/akitaakita_9.jpg
 create mode 100644 static/datasets/train13/Sai/.DS_Store
 create mode 100644 static/datasets/train13/Sai/Sai_10.jpg
 create mode 100644 static/datasets/train13/Sai/Sai_11.jpg
 create mode 100644 static/datasets/train13/Sai/Sai_12.jpg
 create mode 100644 static/datasets/train13/Sai/Sai_2.jpg
 create mode 100644 static/datasets/train13/Sai/Sai_3.jpg
 create mode 100644 static/datasets/train13/Sai/Sai_4.jpg
 create mode 100644 static/datasets/train13/Sai/Sai_5.jpg
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % heroku login
heroku: Press any key to open up the browser to login or q to exit: 
Opening browser to https://cli-auth.heroku.com/auth/cli/browser/92ad3dcc-ace5-4f86-955e-94039b3454cd?requestor=SFMyNTY.g2gDbQAAAA4xNDAuMjU0Ljc3LjIyNW4GAHj9GDCVAWIAAVGA.Tm1AfIVexEATzy9AU-aCSVpgvutqQykfs3jbCxkmM2Q
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % heroku create website-classification
Creating ⬢ website-classification... done
https://website-classification-d6f350ecbf10.herokuapp.com/ | https://git.heroku.com/website-classification.git
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % git push heroku main
Enumerating objects: 12395, done.
Counting objects: 100% (12395/12395), done.
Delta compression using up to 8 threads
Compressing objects: 100% (9021/9021), done.
Writing objects: 100% (12395/12395), 222.35 MiB | 16.85 MiB/s, done.
Total 12395 (delta 3263), reused 12362 (delta 3254), pack-reused 0
remote: Resolving deltas: 100% (3263/3263), done.
remote: Updated 7440 paths from 16190dd
remote: Compressing source files... done.
remote: Building source:
remote: 
remote: -----> Building on the Heroku-24 stack
remote: -----> Determining which buildpack to use for this app
remote: -----> Python app detected
remote: -----> No Python version was specified. Using the buildpack default: Python 3.13
remote:        To use a different version, see: https://devcenter.heroku.com/articles/python-runtimes
remote: -----> Installing Python 3.13.2
remote: -----> Installing pip 24.3.1
remote: -----> Installing dependencies using 'pip install -r requirements.txt'
remote:        Collecting blinker==1.9.0 (from -r requirements.txt (line 1))
remote:          Downloading blinker-1.9.0-py3-none-any.whl.metadata (1.6 kB)
remote:        Collecting cachelib==0.13.0 (from -r requirements.txt (line 2))
remote:          Downloading cachelib-0.13.0-py3-none-any.whl.metadata (2.0 kB)
remote:        Collecting click==8.1.8 (from -r requirements.txt (line 3))
remote:          Downloading click-8.1.8-py3-none-any.whl.metadata (2.3 kB)
remote:        Collecting Flask==3.1.0 (from -r requirements.txt (line 4))
remote:          Downloading flask-3.1.0-py3-none-any.whl.metadata (2.7 kB)
remote:        Collecting Flask-Session==0.8.0 (from -r requirements.txt (line 5))
remote:          Downloading flask_session-0.8.0-py3-none-any.whl.metadata (5.2 kB)
remote:        Collecting itsdangerous==2.2.0 (from -r requirements.txt (line 6))
remote:          Downloading itsdangerous-2.2.0-py3-none-any.whl.metadata (1.9 kB)
remote:        Collecting Jinja2==3.1.5 (from -r requirements.txt (line 7))
remote:          Downloading jinja2-3.1.5-py3-none-any.whl.metadata (2.6 kB)
remote:        Collecting joblib==1.4.2 (from -r requirements.txt (line 8))
remote:          Downloading joblib-1.4.2-py3-none-any.whl.metadata (5.4 kB)
remote:        Collecting MarkupSafe==3.0.2 (from -r requirements.txt (line 9))
remote:          Downloading MarkupSafe-3.0.2-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.0 kB)
remote:        Collecting msgspec==0.19.0 (from -r requirements.txt (line 10))
remote:          Downloading msgspec-0.19.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.9 kB)
remote:        Collecting numpy==1.23.5 (from -r requirements.txt (line 11))
remote:          Downloading numpy-1.23.5.tar.gz (10.7 MB)
remote:          Installing build dependencies: started
remote:          Installing build dependencies: finished with status 'done'
remote:          Getting requirements to build wheel: started
remote:          Getting requirements to build wheel: finished with status 'done'
remote:        ERROR: Exception:
remote:        Traceback (most recent call last):
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/cli/base_command.py", line 105, in _run_wrapper
remote:            status = _inner_run()
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/cli/base_command.py", line 96, in _inner_run
remote:            return self.run(options, args)
remote:                   ~~~~~~~~^^^^^^^^^^^^^^^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/cli/req_command.py", line 67, in wrapper
remote:            return func(self, options, args)
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/commands/install.py", line 379, in run
remote:            requirement_set = resolver.resolve(
remote:                reqs, check_supported_wheels=not options.target_dir
remote:            )
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/resolver.py", line 95, in resolve
remote:            result = self._result = resolver.resolve(
remote:                                    ~~~~~~~~~~~~~~~~^
remote:                collected.requirements, max_rounds=limit_how_complex_resolution_can_be
remote:                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
remote:            )
remote:            ^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_vendor/resolvelib/resolvers.py", line 546, in resolve
remote:            state = resolution.resolve(requirements, max_rounds=max_rounds)
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_vendor/resolvelib/resolvers.py", line 397, in resolve
remote:            self._add_to_criteria(self.state.criteria, r, parent=None)
remote:            ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_vendor/resolvelib/resolvers.py", line 173, in _add_to_criteria
remote:            if not criterion.candidates:
remote:                   ^^^^^^^^^^^^^^^^^^^^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_vendor/resolvelib/structs.py", line 156, in __bool__
remote:            return bool(self._sequence)
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/found_candidates.py", line 174, in __bool__
remote:            return any(self)
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/found_candidates.py", line 162, in <genexpr>
remote:            return (c for c in iterator if id(c) not in self._incompatible_ids)
remote:                               ^^^^^^^^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/found_candidates.py", line 53, in _iter_built
remote:            candidate = func()
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/factory.py", line 187, in _make_candidate_from_link
remote:            base: Optional[BaseCandidate] = self._make_base_candidate_from_link(
remote:                                            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
remote:                link, template, name, version
remote:                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
remote:            )
remote:            ^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/factory.py", line 233, in _make_base_candidate_from_link
remote:            self._link_candidate_cache[link] = LinkCandidate(
remote:                                               ~~~~~~~~~~~~~^
remote:                link,
remote:                ^^^^^
remote:            ...<3 lines>...
remote:                version=version,
remote:                ^^^^^^^^^^^^^^^^
remote:            )
remote:            ^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/candidates.py", line 304, in __init__
remote:            super().__init__(
remote:            ~~~~~~~~~~~~~~~~^
remote:                link=link,
remote:                ^^^^^^^^^^
remote:            ...<4 lines>...
remote:                version=version,
remote:                ^^^^^^^^^^^^^^^^
remote:            )
remote:            ^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/candidates.py", line 159, in __init__
remote:            self.dist = self._prepare()
remote:                        ~~~~~~~~~~~~~^^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/candidates.py", line 236, in _prepare
remote:            dist = self._prepare_distribution()
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/resolution/resolvelib/candidates.py", line 315, in _prepare_distribution
remote:            return preparer.prepare_linked_requirement(self._ireq, parallel_builds=True)
remote:                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/operations/prepare.py", line 527, in prepare_linked_requirement
remote:            return self._prepare_linked_requirement(req, parallel_builds)
remote:                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/operations/prepare.py", line 642, in _prepare_linked_requirement
remote:            dist = _get_prepared_distribution(
remote:                req,
remote:            ...<3 lines>...
remote:                self.check_build_deps,
remote:            )
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/operations/prepare.py", line 72, in _get_prepared_distribution
remote:            abstract_dist.prepare_distribution_metadata(
remote:            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
remote:                finder, build_isolation, check_build_deps
remote:                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
remote:            )
remote:            ^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/distributions/sdist.py", line 56, in prepare_distribution_metadata
remote:            self._install_build_reqs(finder)
remote:            ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/distributions/sdist.py", line 126, in _install_build_reqs
remote:            build_reqs = self._get_build_requires_wheel()
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/distributions/sdist.py", line 103, in _get_build_requires_wheel
remote:            return backend.get_requires_for_build_wheel()
remote:                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_internal/utils/misc.py", line 701, in get_requires_for_build_wheel
remote:            return super().get_requires_for_build_wheel(config_settings=cs)
remote:                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_vendor/pyproject_hooks/_impl.py", line 166, in get_requires_for_build_wheel
remote:            return self._call_hook('get_requires_for_build_wheel', {
remote:                   ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
remote:                'config_settings': config_settings
remote:                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
remote:            })
remote:            ^^
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_vendor/pyproject_hooks/_impl.py", line 321, in _call_hook
remote:            raise BackendUnavailable(data.get('traceback', ''))
remote:        pip._vendor.pyproject_hooks._impl.BackendUnavailable: Traceback (most recent call last):
remote:          File "/app/.heroku/python/lib/python3.13/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 77, in _build_backend
remote:            obj = import_module(mod_path)
remote:          File "/app/.heroku/python/lib/python3.13/importlib/__init__.py", line 88, in import_module
remote:            return _bootstrap._gcd_import(name[level:], package, level)
remote:                   ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
remote:          File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
remote:          File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
remote:          File "<frozen importlib._bootstrap>", line 1310, in _find_and_load_unlocked
remote:          File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
remote:          File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
remote:          File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
remote:          File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
remote:          File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
remote:          File "<frozen importlib._bootstrap_external>", line 1026, in exec_module
remote:          File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
remote:          File "/tmp/pip-build-env-o9dh0g3n/overlay/lib/python3.13/site-packages/setuptools/__init__.py", line 10, in <module>
remote:            import distutils.core
remote:        ModuleNotFoundError: No module named 'distutils'
remote:        
remote: 
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % git add .
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % git commit -m "add runtime.txt"
[main e2411e6] add runtime.txt
 4 files changed, 4 insertions(+), 2 deletions(-)
 create mode 100644 Procfile
 create mode 100644 runtime.txt
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % git push heroku main
Enumerating objects: 12401, done.
Counting objects: 100% (12401/12401), done.
Delta compression using up to 8 threads
Compressing objects: 100% (9025/9025), done.
Writing objects: 100% (12401/12401), 222.35 MiB | 15.13 MiB/s, done.
Total 12401 (delta 3265), reused 12362 (delta 3254), pack-reused 0
remote: Resolving deltas: 100% (3265/3265), done.
remote: Updated 7442 paths from b470332
remote: Compressing source files... done.
remote: Building source:
remote: 
remote: -----> Building on the Heroku-24 stack
remote: -----> Determining which buildpack to use for this app
remote: -----> Python app detected
remote: -----> Using Python 3.11.0 specified in runtime.txt
remote: 
remote:  !     Warning: The runtime.txt file is deprecated.
remote:  !     
remote:  !     The runtime.txt file is deprecated since it has been replaced
remote:  !     by the more widely supported .python-version file:
remote:  !     https://devcenter.heroku.com/changelog-items/3141
remote:  !     
remote:  !     Please delete your runtime.txt file and create a new file named:
remote:  !     .python-version
remote:  !     
remote:  !     Make sure to include the '.' at the start of the filename.
remote:  !     
remote:  !     In the new file, specify your app's Python version without
remote:  !     quotes or a 'python-' prefix. For example:
remote:  !     3.11
remote:  !     
remote:  !     We strongly recommend that you use the major version form
remote:  !     instead of pinning to an exact version, since it will allow
remote:  !     your app to receive Python security updates.
remote:  !     
remote:  !     In the future support for runtime.txt will be removed and
remote:  !     this warning will be made an error.
remote: 
remote: 
remote:  !     Warning: A Python patch update is available!
remote:  !     
remote:  !     Your app is using Python 3.11.0, however, there is a newer
remote:  !     patch release of Python 3.11 available: 3.11.11
remote:  !     
remote:  !     It is important to always use the latest patch version of
remote:  !     Python to keep your app secure.
remote:  !     
remote:  !     Update your runtime.txt file to use the new version.
remote:  !     
remote:  !     We strongly recommend that you do not pin your app to an
remote:  !     exact Python version such as 3.11.0, and instead only specify
remote:  !     the major Python version of 3.11 in your runtime.txt file.
remote:  !     This will allow your app to receive the latest available Python
remote:  !     patch version automatically and prevent this warning.
remote: 
remote: -----> Installing Python 3.11.0
remote: -----> Installing pip 24.3.1, setuptools 70.3.0 and wheel 0.45.1
remote: -----> Installing SQLite3
remote: -----> Installing dependencies using 'pip install -r requirements.txt'
remote:        Collecting blinker==1.9.0 (from -r requirements.txt (line 1))
remote:          Downloading blinker-1.9.0-py3-none-any.whl.metadata (1.6 kB)
remote:        Collecting cachelib==0.13.0 (from -r requirements.txt (line 2))
remote:          Downloading cachelib-0.13.0-py3-none-any.whl.metadata (2.0 kB)
remote:        Collecting click==8.1.8 (from -r requirements.txt (line 3))
remote:          Downloading click-8.1.8-py3-none-any.whl.metadata (2.3 kB)
remote:        Collecting Flask==3.1.0 (from -r requirements.txt (line 4))
remote:          Downloading flask-3.1.0-py3-none-any.whl.metadata (2.7 kB)
remote:        Collecting Flask-Session==0.8.0 (from -r requirements.txt (line 5))
remote:          Downloading flask_session-0.8.0-py3-none-any.whl.metadata (5.2 kB)
remote:        Collecting itsdangerous==2.2.0 (from -r requirements.txt (line 6))
remote:          Downloading itsdangerous-2.2.0-py3-none-any.whl.metadata (1.9 kB)
remote:        Collecting Jinja2==3.1.5 (from -r requirements.txt (line 7))
remote:          Downloading jinja2-3.1.5-py3-none-any.whl.metadata (2.6 kB)
remote:        Collecting joblib==1.4.2 (from -r requirements.txt (line 8))
remote:          Downloading joblib-1.4.2-py3-none-any.whl.metadata (5.4 kB)
remote:        Collecting MarkupSafe==3.0.2 (from -r requirements.txt (line 9))
remote:          Downloading MarkupSafe-3.0.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.0 kB)
remote:        Collecting msgspec==0.19.0 (from -r requirements.txt (line 10))
remote:          Downloading msgspec-0.19.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.9 kB)
remote:        Collecting numpy>=1.24 (from -r requirements.txt (line 11))
remote:          Downloading numpy-2.2.3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (62 kB)
remote:        Collecting opencv-python==4.11.0.86 (from -r requirements.txt (line 12))
remote:          Downloading opencv_python-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
remote:        Collecting pillow==11.1.0 (from -r requirements.txt (line 13))
remote:          Downloading pillow-11.1.0-cp311-cp311-manylinux_2_28_x86_64.whl.metadata (9.1 kB)
remote:        Collecting scikit-learn==1.6.1 (from -r requirements.txt (line 14))
remote:          Downloading scikit_learn-1.6.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (18 kB)
remote:        Collecting scipy==1.10.1 (from -r requirements.txt (line 15))
remote:          Downloading scipy-1.10.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (58 kB)
remote:        Collecting threadpoolctl==3.5.0 (from -r requirements.txt (line 16))
remote:          Downloading threadpoolctl-3.5.0-py3-none-any.whl.metadata (13 kB)
remote:        Collecting Werkzeug==3.1.3 (from -r requirements.txt (line 17))
remote:          Downloading werkzeug-3.1.3-py3-none-any.whl.metadata (3.7 kB)
remote:        Collecting numpy>=1.24 (from -r requirements.txt (line 11))
remote:          Downloading numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
remote:        Downloading blinker-1.9.0-py3-none-any.whl (8.5 kB)
remote:        Downloading cachelib-0.13.0-py3-none-any.whl (20 kB)
remote:        Downloading click-8.1.8-py3-none-any.whl (98 kB)
remote:        Downloading flask-3.1.0-py3-none-any.whl (102 kB)
remote:        Downloading flask_session-0.8.0-py3-none-any.whl (24 kB)
remote:        Downloading itsdangerous-2.2.0-py3-none-any.whl (16 kB)
remote:        Downloading jinja2-3.1.5-py3-none-any.whl (134 kB)
remote:        Downloading joblib-1.4.2-py3-none-any.whl (301 kB)
remote:        Downloading MarkupSafe-3.0.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (23 kB)
remote:        Downloading msgspec-0.19.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (210 kB)
remote:        Downloading opencv_python-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (63.0 MB)
remote:        Downloading pillow-11.1.0-cp311-cp311-manylinux_2_28_x86_64.whl (4.5 MB)
remote:        Downloading scikit_learn-1.6.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (13.5 MB)
remote:        Downloading scipy-1.10.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (34.1 MB)
remote:        Downloading threadpoolctl-3.5.0-py3-none-any.whl (18 kB)
remote:        Downloading werkzeug-3.1.3-py3-none-any.whl (224 kB)
remote:        Downloading numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.3 MB)
remote:        Installing collected packages: threadpoolctl, pillow, numpy, msgspec, MarkupSafe, joblib, itsdangerous, click, cachelib, blinker, Werkzeug, scipy, opencv-python, Jinja2, scikit-learn, Flask, Flask-Session
remote:        Successfully installed Flask-3.1.0 Flask-Session-0.8.0 Jinja2-3.1.5 MarkupSafe-3.0.2 Werkzeug-3.1.3 blinker-1.9.0 cachelib-0.13.0 click-8.1.8 itsdangerous-2.2.0 joblib-1.4.2 msgspec-0.19.0 numpy-1.26.4 opencv-python-4.11.0.86 pillow-11.1.0 scikit-learn-1.6.1 scipy-1.10.1 threadpoolctl-3.5.0
remote: -----> Discovering process types
remote:        Procfile declares types -> web
remote: 
remote: -----> Compressing...
remote:        Done: 290.4M
remote: -----> Launching...
remote:        Released v3
remote:        https://website-classification-d6f350ecbf10.herokuapp.com/ deployed to Heroku
remote: 
remote: Verifying deploy... done.
To https://git.heroku.com/website-classification.git
 * [new branch]      main -> main
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % heroku log
 ›   Warning: log is not a heroku command.
Did you mean logs? [y/n]: y
2025-02-23T00:02:36.929605+00:00 app[api]: Release v1 created by user ar20060306@gmail.com
2025-02-23T00:02:36.929605+00:00 app[api]: Initial release by user ar20060306@gmail.com
2025-02-23T00:02:37.106078+00:00 app[api]: Enable Logplex by user ar20060306@gmail.com
2025-02-23T00:02:37.106078+00:00 app[api]: Release v2 created by user ar20060306@gmail.com
2025-02-23T00:03:24.000000+00:00 app[api]: Build started by user ar20060306@gmail.com
2025-02-23T00:03:39.000000+00:00 app[api]: Build failed -- check your build output: https://dashboard.heroku.com/apps/c9ea1cdc-4c42-485c-bf74-776b6e64bfeb/activity/builds/ef520034-aae0-4ae8-9d8f-8d68e050f63a
2025-02-23T00:04:14.993361+00:00 heroku[router]: at=info code=H81 desc="Blank app" method=GET path="/" host=website-classification-d6f350ecbf10.herokuapp.com request_id=3c8cf421-6613-4ee4-a797-946472c88ef0 fwd="140.254.77.225" dyno= connect= service= status=502 bytes= protocol=https
2025-02-23T00:04:15.329868+00:00 heroku[router]: at=info code=H81 desc="Blank app" method=GET path="/favicon.ico" host=website-classification-d6f350ecbf10.herokuapp.com request_id=050270ce-4ae2-41d3-af36-9ef32d4ef9b1 fwd="140.254.77.225" dyno= connect= service= status=502 bytes= protocol=https
2025-02-23T00:04:18.916332+00:00 heroku[router]: at=info code=H81 desc="Blank app" method=GET path="/" host=website-classification-d6f350ecbf10.herokuapp.com request_id=58f4c398-dc09-4f41-adb1-66ad0d95566d fwd="140.254.77.225" dyno= connect= service= status=502 bytes= protocol=https
2025-02-23T00:04:19.063726+00:00 heroku[router]: at=info code=H81 desc="Blank app" method=GET path="/favicon.ico" host=website-classification-d6f350ecbf10.herokuapp.com request_id=a1400de5-8638-4e38-bcf7-62e7f4d51669 fwd="140.254.77.225" dyno= connect= service= status=502 bytes= protocol=https
2025-02-23T00:04:19.622382+00:00 heroku[router]: at=info code=H81 desc="Blank app" method=GET path="/" host=website-classification-d6f350ecbf10.herokuapp.com request_id=072725e4-098a-4327-bc1c-f49b94e93d9b fwd="140.254.77.225" dyno= connect= service= status=502 bytes= protocol=https
2025-02-23T00:04:19.817853+00:00 heroku[router]: at=info code=H81 desc="Blank app" method=GET path="/favicon.ico" host=website-classification-d6f350ecbf10.herokuapp.com request_id=05730d71-f02d-48a5-8bc3-86c58d35bddb fwd="140.254.77.225" dyno= connect= service= status=502 bytes= protocol=https
2025-02-23T00:07:01.000000+00:00 app[api]: Build started by user ar20060306@gmail.com
2025-02-23T00:07:50.363120+00:00 app[api]: Release v3 created by user ar20060306@gmail.com
2025-02-23T00:07:50.363120+00:00 app[api]: Deploy e2411e6d by user ar20060306@gmail.com
2025-02-23T00:07:50.376508+00:00 app[api]: Scaled to web@1:Basic by user ar20060306@gmail.com
2025-02-23T00:08:02.220467+00:00 heroku[web.1]: Starting process with command gunicorn -w 4 -b 0.0.0.0:10000 app:app
2025-02-23T00:08:02.885155+00:00 app[web.1]: Python buildpack: Detected 512 MB available memory and 8 CPU cores.
2025-02-23T00:08:02.885168+00:00 app[web.1]: Python buildpack: Defaulting WEB_CONCURRENCY to 2 based on the available memory.
2025-02-23T00:08:02.910754+00:00 app[web.1]: /bin/bash: line 1: gunicorn: command not found
2025-02-23T00:08:02.951971+00:00 heroku[web.1]: Process exited with status 127
2025-02-23T00:08:02.975287+00:00 heroku[web.1]: State changed from starting to crashed
2025-02-23T00:08:02.977965+00:00 heroku[web.1]: State changed from crashed to starting
2025-02-23T00:08:14.020925+00:00 heroku[web.1]: Starting process with command gunicorn -w 4 -b 0.0.0.0:10000 app:app
2025-02-23T00:08:14.504088+00:00 app[web.1]: Python buildpack: Detected 512 MB available memory and 8 CPU cores.
2025-02-23T00:08:14.504161+00:00 app[web.1]: Python buildpack: Defaulting WEB_CONCURRENCY to 2 based on the available memory.
2025-02-23T00:08:14.527320+00:00 app[web.1]: /bin/bash: line 1: gunicorn: command not found
2025-02-23T00:08:14.573353+00:00 heroku[web.1]: Process exited with status 127
2025-02-23T00:08:14.595281+00:00 heroku[web.1]: State changed from starting to crashed
2025-02-23T00:08:15.000000+00:00 app[api]: Build succeeded
2025-02-23T00:08:16.163684+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/" host=website-classification-d6f350ecbf10.herokuapp.com request_id=774e759a-5218-4334-8860-12027c43234c fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:08:16.520666+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/favicon.ico" host=website-classification-d6f350ecbf10.herokuapp.com request_id=66b99f2f-4f4c-4975-af1f-dec4832676ef fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:08:19.755229+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/" host=website-classification-d6f350ecbf10.herokuapp.com request_id=687b871b-3d4c-405e-8a07-7460c9868a6c fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:08:19.899551+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/favicon.ico" host=website-classification-d6f350ecbf10.herokuapp.com request_id=e4507cb5-3835-4e88-a060-8c8be2a18502 fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:08:21.069831+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/" host=website-classification-d6f350ecbf10.herokuapp.com request_id=5957c772-4b06-495e-b078-8d43917bec78 fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:08:21.217781+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/favicon.ico" host=website-classification-d6f350ecbf10.herokuapp.com request_id=bc96b55f-15fe-4622-8367-3106d073aa1e fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:08:21.883574+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/" host=website-classification-d6f350ecbf10.herokuapp.com request_id=6cdee750-ad45-4d14-ad6d-4131650abac6 fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:08:22.086609+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/favicon.ico" host=website-classification-d6f350ecbf10.herokuapp.com request_id=430f9b8d-f259-4382-9329-f3f6e17e6f65 fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:08:23.367218+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/" host=website-classification-d6f350ecbf10.herokuapp.com request_id=24be13a0-b9ce-457a-87df-47211be89f48 fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:08:23.590221+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/favicon.ico" host=website-classification-d6f350ecbf10.herokuapp.com request_id=6ec7418a-1d69-49d7-9027-f73507142678 fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:08:24.296210+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/" host=website-classification-d6f350ecbf10.herokuapp.com request_id=b3ca441e-ad9e-4de2-8d8c-403744de69ad fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:08:24.496179+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/favicon.ico" host=website-classification-d6f350ecbf10.herokuapp.com request_id=d2d9c1c9-f52e-409e-be36-9b95ec939cf7 fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % pip freeze > requirements.txt
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % git add requirements.txt
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % git commit -m "added gunicorn"
[main 51bd168] added gunicorn
 1 file changed, 3 insertions(+), 1 deletion(-)
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % git push heroku main
Enumerating objects: 5, done.
Counting objects: 100% (5/5), done.
Delta compression using up to 8 threads
Compressing objects: 100% (3/3), done.
Writing objects: 100% (3/3), 344 bytes | 344.00 KiB/s, done.
Total 3 (delta 2), reused 0 (delta 0), pack-reused 0
remote: Updated 7442 paths from db3693f
remote: Compressing source files... done.
remote: Building source:
remote: 
remote: -----> Building on the Heroku-24 stack
remote: -----> Using buildpack: heroku/python
remote: -----> Python app detected
remote: -----> Using Python 3.11.0 specified in runtime.txt
remote: 
remote:  !     Warning: The runtime.txt file is deprecated.
remote:  !     
remote:  !     The runtime.txt file is deprecated since it has been replaced
remote:  !     by the more widely supported .python-version file:
remote:  !     https://devcenter.heroku.com/changelog-items/3141
remote:  !     
remote:  !     Please delete your runtime.txt file and create a new file named:
remote:  !     .python-version
remote:  !     
remote:  !     Make sure to include the '.' at the start of the filename.
remote:  !     
remote:  !     In the new file, specify your app's Python version without
remote:  !     quotes or a 'python-' prefix. For example:
remote:  !     3.11
remote:  !     
remote:  !     We strongly recommend that you use the major version form
remote:  !     instead of pinning to an exact version, since it will allow
remote:  !     your app to receive Python security updates.
remote:  !     
remote:  !     In the future support for runtime.txt will be removed and
remote:  !     this warning will be made an error.
remote: 
remote: 
remote:  !     Warning: A Python patch update is available!
remote:  !     
remote:  !     Your app is using Python 3.11.0, however, there is a newer
remote:  !     patch release of Python 3.11 available: 3.11.11
remote:  !     
remote:  !     It is important to always use the latest patch version of
remote:  !     Python to keep your app secure.
remote:  !     
remote:  !     Update your runtime.txt file to use the new version.
remote:  !     
remote:  !     We strongly recommend that you do not pin your app to an
remote:  !     exact Python version such as 3.11.0, and instead only specify
remote:  !     the major Python version of 3.11 in your runtime.txt file.
remote:  !     This will allow your app to receive the latest available Python
remote:  !     patch version automatically and prevent this warning.
remote: 
remote: -----> Discarding cache since:
remote:        - The contents of requirements.txt changed
remote: -----> Installing Python 3.11.0
remote: -----> Installing pip 24.3.1, setuptools 70.3.0 and wheel 0.45.1
remote: -----> Installing SQLite3
remote: -----> Installing dependencies using 'pip install -r requirements.txt'
remote:        Collecting blinker==1.9.0 (from -r requirements.txt (line 1))
remote:          Downloading blinker-1.9.0-py3-none-any.whl.metadata (1.6 kB)
remote:        Collecting cachelib==0.13.0 (from -r requirements.txt (line 2))
remote:          Downloading cachelib-0.13.0-py3-none-any.whl.metadata (2.0 kB)
remote:        Collecting click==8.1.8 (from -r requirements.txt (line 3))
remote:          Downloading click-8.1.8-py3-none-any.whl.metadata (2.3 kB)
remote:        Collecting Flask==3.1.0 (from -r requirements.txt (line 4))
remote:          Downloading flask-3.1.0-py3-none-any.whl.metadata (2.7 kB)
remote:        Collecting Flask-Session==0.8.0 (from -r requirements.txt (line 5))
remote:          Downloading flask_session-0.8.0-py3-none-any.whl.metadata (5.2 kB)
remote:        Collecting gunicorn==23.0.0 (from -r requirements.txt (line 6))
remote:          Downloading gunicorn-23.0.0-py3-none-any.whl.metadata (4.4 kB)
remote:        Collecting itsdangerous==2.2.0 (from -r requirements.txt (line 7))
remote:          Downloading itsdangerous-2.2.0-py3-none-any.whl.metadata (1.9 kB)
remote:        Collecting Jinja2==3.1.5 (from -r requirements.txt (line 8))
remote:          Downloading jinja2-3.1.5-py3-none-any.whl.metadata (2.6 kB)
remote:        Collecting joblib==1.4.2 (from -r requirements.txt (line 9))
remote:          Downloading joblib-1.4.2-py3-none-any.whl.metadata (5.4 kB)
remote:        Collecting MarkupSafe==3.0.2 (from -r requirements.txt (line 10))
remote:          Downloading MarkupSafe-3.0.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.0 kB)
remote:        Collecting msgspec==0.19.0 (from -r requirements.txt (line 11))
remote:          Downloading msgspec-0.19.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.9 kB)
remote:        Collecting numpy==1.23.5 (from -r requirements.txt (line 12))
remote:          Downloading numpy-1.23.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.3 kB)
remote:        Collecting opencv-python==4.11.0.86 (from -r requirements.txt (line 13))
remote:          Downloading opencv_python-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
remote:        Collecting packaging==24.2 (from -r requirements.txt (line 14))
remote:          Downloading packaging-24.2-py3-none-any.whl.metadata (3.2 kB)
remote:        Collecting pillow==11.1.0 (from -r requirements.txt (line 15))
remote:          Downloading pillow-11.1.0-cp311-cp311-manylinux_2_28_x86_64.whl.metadata (9.1 kB)
remote:        Collecting scikit-learn==1.6.1 (from -r requirements.txt (line 16))
remote:          Downloading scikit_learn-1.6.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (18 kB)
remote:        Collecting scipy==1.10.1 (from -r requirements.txt (line 17))
remote:          Downloading scipy-1.10.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (58 kB)
remote:        Collecting threadpoolctl==3.5.0 (from -r requirements.txt (line 18))
remote:          Downloading threadpoolctl-3.5.0-py3-none-any.whl.metadata (13 kB)
remote:        Collecting Werkzeug==3.1.3 (from -r requirements.txt (line 19))
remote:          Downloading werkzeug-3.1.3-py3-none-any.whl.metadata (3.7 kB)
remote:        Downloading blinker-1.9.0-py3-none-any.whl (8.5 kB)
remote:        Downloading cachelib-0.13.0-py3-none-any.whl (20 kB)
remote:        Downloading click-8.1.8-py3-none-any.whl (98 kB)
remote:        Downloading flask-3.1.0-py3-none-any.whl (102 kB)
remote:        Downloading flask_session-0.8.0-py3-none-any.whl (24 kB)
remote:        Downloading gunicorn-23.0.0-py3-none-any.whl (85 kB)
remote:        Downloading itsdangerous-2.2.0-py3-none-any.whl (16 kB)
remote:        Downloading jinja2-3.1.5-py3-none-any.whl (134 kB)
remote:        Downloading joblib-1.4.2-py3-none-any.whl (301 kB)
remote:        Downloading MarkupSafe-3.0.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (23 kB)
remote:        Downloading msgspec-0.19.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (210 kB)
remote:        Downloading numpy-1.23.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (17.1 MB)
remote:        Downloading opencv_python-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (63.0 MB)
remote:        Downloading packaging-24.2-py3-none-any.whl (65 kB)
remote:        Downloading pillow-11.1.0-cp311-cp311-manylinux_2_28_x86_64.whl (4.5 MB)
remote:        Downloading scikit_learn-1.6.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (13.5 MB)
remote:        Downloading scipy-1.10.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (34.1 MB)
remote:        Downloading threadpoolctl-3.5.0-py3-none-any.whl (18 kB)
remote:        Downloading werkzeug-3.1.3-py3-none-any.whl (224 kB)
remote:        Installing collected packages: threadpoolctl, pillow, packaging, numpy, msgspec, MarkupSafe, jobli
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % git log
commit 51bd168a074d6478b18ab002ea67cb4db2307260 (HEAD -> main, heroku/main)
Author: Andrew <97545903+Andrewros@users.noreply.github.com>
Date:   Sat Feb 22 19:11:19 2025 -0500

    added gunicorn

commit e2411e6dd9e2117b96ea9ab20c423c39a8749c13
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % heroku logs --tail
2025-02-23T00:12:56.834815+00:00 app[web.1]: File "/app/app.py", line 5, in <module>
2025-02-23T00:12:56.834816+00:00 app[web.1]: import cv2
2025-02-23T00:12:56.834818+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/site-packages/cv2/__init__.py", line 181, in <module>
2025-02-23T00:12:56.834818+00:00 app[web.1]: bootstrap()
2025-02-23T00:12:56.834818+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/site-packages/cv2/__init__.py", line 153, in bootstrap
2025-02-23T00:12:56.834819+00:00 app[web.1]: native_module = importlib.import_module("cv2")
2025-02-23T00:12:56.834819+00:00 app[web.1]: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-02-23T00:12:56.834819+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/importlib/__init__.py", line 126, in import_module
2025-02-23T00:12:56.834820+00:00 app[web.1]: return _bootstrap._gcd_import(name[level:], package, level)
2025-02-23T00:12:56.834820+00:00 app[web.1]: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-02-23T00:12:56.834820+00:00 app[web.1]: ImportError: libGL.so.1: cannot open shared object file: No such file or directory
2025-02-23T00:12:56.834899+00:00 app[web.1]: [2025-02-23 00:12:56 +0000] [26] [INFO] Worker exiting (pid: 26)
2025-02-23T00:12:56.858013+00:00 app[web.1]: [2025-02-23 00:12:56 +0000] [11] [ERROR] Exception in worker process
2025-02-23T00:12:56.858015+00:00 app[web.1]: Traceback (most recent call last):
2025-02-23T00:12:56.858016+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/site-packages/gunicorn/arbiter.py", line 608, in spawn_worker
2025-02-23T00:12:56.858016+00:00 app[web.1]: worker.init_process()
2025-02-23T00:12:56.858016+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/site-packages/gunicorn/workers/base.py", line 135, in init_process
2025-02-23T00:12:56.858017+00:00 app[web.1]: self.load_wsgi()
2025-02-23T00:12:56.858017+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/site-packages/gunicorn/workers/base.py", line 147, in load_wsgi
2025-02-23T00:12:56.858017+00:00 app[web.1]: self.wsgi = self.app.wsgi()
2025-02-23T00:12:56.858018+00:00 app[web.1]: ^^^^^^^^^^^^^^^
2025-02-23T00:12:56.858018+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/site-packages/gunicorn/app/base.py", line 66, in wsgi
2025-02-23T00:12:56.858019+00:00 app[web.1]: self.callable = self.load()
2025-02-23T00:12:56.858019+00:00 app[web.1]: ^^^^^^^^^^^
2025-02-23T00:12:56.858019+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/site-packages/gunicorn/app/wsgiapp.py", line 57, in load
2025-02-23T00:12:56.858019+00:00 app[web.1]: return self.load_wsgiapp()
2025-02-23T00:12:56.858019+00:00 app[web.1]: ^^^^^^^^^^^^^^^^^^^
2025-02-23T00:12:56.858020+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/site-packages/gunicorn/app/wsgiapp.py", line 47, in load_wsgiapp
2025-02-23T00:12:56.858020+00:00 app[web.1]: return util.import_app(self.app_uri)
2025-02-23T00:12:56.858020+00:00 app[web.1]: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-02-23T00:12:56.858020+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/site-packages/gunicorn/util.py", line 370, in import_app
2025-02-23T00:12:56.858021+00:00 app[web.1]: mod = importlib.import_module(module)
2025-02-23T00:12:56.858021+00:00 app[web.1]: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-02-23T00:12:56.858022+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/importlib/__init__.py", line 126, in import_module
2025-02-23T00:12:56.858022+00:00 app[web.1]: return _bootstrap._gcd_import(name[level:], package, level)
2025-02-23T00:12:56.858033+00:00 app[web.1]: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-02-23T00:12:56.858033+00:00 app[web.1]: File "<frozen importlib._bootstrap>", line 1206, in _gcd_import
2025-02-23T00:12:56.858033+00:00 app[web.1]: File "<frozen importlib._bootstrap>", line 1178, in _find_and_load
2025-02-23T00:12:56.858034+00:00 app[web.1]: File "<frozen importlib._bootstrap>", line 1149, in _find_and_load_unlocked
2025-02-23T00:12:56.858034+00:00 app[web.1]: File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
2025-02-23T00:12:56.858034+00:00 app[web.1]: File "<frozen importlib._bootstrap_external>", line 940, in exec_module
2025-02-23T00:12:56.858034+00:00 app[web.1]: File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
2025-02-23T00:12:56.858035+00:00 app[web.1]: File "/app/app.py", line 5, in <module>
2025-02-23T00:12:56.858035+00:00 app[web.1]: import cv2
2025-02-23T00:12:56.858035+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/site-packages/cv2/__init__.py", line 181, in <module>
2025-02-23T00:12:56.858036+00:00 app[web.1]: bootstrap()
2025-02-23T00:12:56.858036+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/site-packages/cv2/__init__.py", line 153, in bootstrap
2025-02-23T00:12:56.858036+00:00 app[web.1]: native_module = importlib.import_module("cv2")
2025-02-23T00:12:56.858036+00:00 app[web.1]: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-02-23T00:12:56.858037+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/importlib/__init__.py", line 126, in import_module
2025-02-23T00:12:56.858037+00:00 app[web.1]: return _bootstrap._gcd_import(name[level:], package, level)
2025-02-23T00:12:56.858037+00:00 app[web.1]: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-02-23T00:12:56.858051+00:00 app[web.1]: ImportError: libGL.so.1: cannot open shared object file: No such file or directory
2025-02-23T00:12:56.858118+00:00 app[web.1]: [2025-02-23 00:12:56 +0000] [11] [INFO] Worker exiting (pid: 11)
2025-02-23T00:12:56.916376+00:00 app[web.1]: [2025-02-23 00:12:56 +0000] [2] [ERROR] Worker (pid:9) exited with code 3
2025-02-23T00:12:56.926295+00:00 app[web.1]: [2025-02-23 00:12:56 +0000] [2] [ERROR] Worker (pid:26) was sent SIGTERM!
2025-02-23T00:12:56.926379+00:00 app[web.1]: [2025-02-23 00:12:56 +0000] [2] [ERROR] Worker (pid:10) exited with code 3
2025-02-23T00:12:56.926448+00:00 app[web.1]: Traceback (most recent call last):
2025-02-23T00:12:56.926459+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/site-packages/gunicorn/arbiter.py", line 208, in run
2025-02-23T00:12:56.926632+00:00 app[web.1]: [2025-02-23 00:12:56 +0000] [2] [ERROR] Worker (pid:11) was sent SIGTERM!
2025-02-23T00:12:56.926734+00:00 app[web.1]: self.sleep()
2025-02-23T00:12:56.926767+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/site-packages/gunicorn/arbiter.py", line 359, in sleep
2025-02-23T00:12:56.926849+00:00 app[web.1]: ready = select.select([self.PIPE[0]], [], [], 1.0)
2025-02-23T00:12:56.926948+00:00 app[web.1]: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-02-23T00:12:56.926948+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/site-packages/gunicorn/arbiter.py", line 241, in handle_chld
2025-02-23T00:12:56.926999+00:00 app[web.1]: self.reap_workers()
2025-02-23T00:12:56.927013+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/site-packages/gunicorn/arbiter.py", line 529, in reap_workers
2025-02-23T00:12:56.927089+00:00 app[web.1]: raise HaltServer(reason, self.WORKER_BOOT_ERROR)
2025-02-23T00:12:56.927121+00:00 app[web.1]: gunicorn.errors.HaltServer: <HaltServer 'Worker failed to boot.' 3>
2025-02-23T00:12:56.927125+00:00 app[web.1]: 
2025-02-23T00:12:56.927126+00:00 app[web.1]: During handling of the above exception, another exception occurred:
2025-02-23T00:12:56.927129+00:00 app[web.1]: 
2025-02-23T00:12:56.927136+00:00 app[web.1]: Traceback (most recent call last):
2025-02-23T00:12:56.927137+00:00 app[web.1]: File "/app/.heroku/python/bin/gunicorn", line 8, in <module>
2025-02-23T00:12:56.927170+00:00 app[web.1]: sys.exit(run())
2025-02-23T00:12:56.927201+00:00 app[web.1]: ^^^^^
2025-02-23T00:12:56.927210+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/site-packages/gunicorn/app/wsgiapp.py", line 66, in run
2025-02-23T00:12:56.927244+00:00 app[web.1]: WSGIApplication("%(prog)s [OPTIONS] [APP_MODULE]", prog=prog).run()
2025-02-23T00:12:56.927267+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/site-packages/gunicorn/app/base.py", line 235, in run
2025-02-23T00:12:56.927354+00:00 app[web.1]: super().run()
2025-02-23T00:12:56.927374+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/site-packages/gunicorn/app/base.py", line 71, in run
2025-02-23T00:12:56.927409+00:00 app[web.1]: Arbiter(self).run()
2025-02-23T00:12:56.927424+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/site-packages/gunicorn/arbiter.py", line 228, in run
2025-02-23T00:12:56.927472+00:00 app[web.1]: self.halt(reason=inst.reason, exit_status=inst.exit_status)
2025-02-23T00:12:56.927494+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/site-packages/gunicorn/arbiter.py", line 341, in halt
2025-02-23T00:12:56.927555+00:00 app[web.1]: self.stop()
2025-02-23T00:12:56.927567+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/site-packages/gunicorn/arbiter.py", line 395, in stop
2025-02-23T00:12:56.927628+00:00 app[web.1]: time.sleep(0.1)
2025-02-23T00:12:56.927643+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/site-packages/gunicorn/arbiter.py", line 241, in handle_chld
2025-02-23T00:12:56.927687+00:00 app[web.1]: self.reap_workers()
2025-02-23T00:12:56.927699+00:00 app[web.1]: File "/app/.heroku/python/lib/python3.11/site-packages/gunicorn/arbiter.py", line 529, in reap_workers
2025-02-23T00:12:56.927770+00:00 app[web.1]: raise HaltServer(reason, self.WORKER_BOOT_ERROR)
2025-02-23T00:12:56.927796+00:00 app[web.1]: gunicorn.errors.HaltServer: <HaltServer 'Worker failed to boot.' 3>
2025-02-23T00:12:56.998247+00:00 heroku[web.1]: Process exited with status 1
2025-02-23T00:12:57.023575+00:00 heroku[web.1]: State changed from starting to crashed
2025-02-23T00:13:09.000000+00:00 app[api]: Build succeeded
2025-02-23T00:13:20.697363+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/" host=website-classification-d6f350ecbf10.herokuapp.com request_id=4c0c89ac-4844-4de5-94ba-5ebebb9622db fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:13:20.849548+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/favicon.ico" host=website-classification-d6f350ecbf10.herokuapp.com request_id=edbf5b9f-3df9-409e-a020-d82f1cb57a6c fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:14:02.748667+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/" host=website-classification-d6f350ecbf10.herokuapp.com request_id=7994f77c-d287-45ff-b9af-ea5caab9ce07 fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:14:03.071125+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/favicon.ico" host=website-classification-d6f350ecbf10.herokuapp.com request_id=cecd6dfc-4049-43b8-97e8-32f53ef8967b fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
       
ChatGPT said:
Your Heroku app is still crashing due to a missing system library: libGL.so.1. This is required by OpenCV for image processing.

Fix: Install libGL.so.1 on Heroku
Heroku does not provide system dependencies like libGL.so.1 by default, so you need to install it manually.

Step 1: Create an Aptfile
Heroku supports installing system dependencies using an Aptfile. Run the following command in your project root:

sh
Copy
Edit
echo "libgl1" > Aptfile
Step 2: Add the Apt Buildpack
Run:

sh
Copy
Edit
heroku buildpacks:add --index 1 heroku-community/apt
Step 3: Commit and Deploy
Add and commit the Aptfile:

sh
Copy
Edit
git add Aptfile
git commit -m "Added libGL dependency for OpenCV"
git push heroku main
Step 4: Restart the App
sh
Copy
Edit
heroku restart
Step 5: Check Logs
sh
Copy
Edit
heroku logs --tail
This should resolve the libGL.so.1: cannot open shared object file: No such file or directory error and allow your app to run successfully. Let me know if you need further help! 🚀






You said:
How do I get the heroku website link now
ChatGPT said:
To get the Heroku website link for your deployed app, run:

sh
Copy
Edit
heroku info -s | grep web_url
Alternatively, you can check the link from your Heroku dashboard:

Go to Heroku Dashboard

Select your app (website-classification-d6f350ecbf10)

Look for the Open App button at the top-right corner. The link should be in the format:

arduino
Copy
Edit
https://website-classification-d6f350ecbf10.herokuapp.com/
Let me know if you need any more help! 🚀






You said:
Now I get this error:
2025-02-23T00:18:35.137898+00:00 app[web.1]: [2025-02-23 00:18:35 +0000] [2] [INFO] Using worker: sync
2025-02-23T00:18:35.140432+00:00 app[web.1]: [2025-02-23 00:18:35 +0000] [9] [INFO] Booting worker with pid: 9
2025-02-23T00:18:35.231857+00:00 app[web.1]: [2025-02-23 00:18:35 +0000] [10] [INFO] Booting worker with pid: 10
2025-02-23T00:18:35.254544+00:00 app[web.1]: [2025-02-23 00:18:35 +0000] [11] [INFO] Booting worker with pid: 11
2025-02-23T00:18:35.292304+00:00 app[web.1]: [2025-02-23 00:18:35 +0000] [12] [INFO] Booting worker with pid: 12
2025-02-23T00:18:42.115025+00:00 heroku[web.1]: Error R10 (Boot timeout) -> Web process failed to bind to $PORT within 60 seconds of launch
2025-02-23T00:18:42.129449+00:00 heroku[web.1]: Stopping process with SIGKILL
2025-02-23T00:18:42.188696+00:00 heroku[web.1]: Process exited with status 137
2025-02-23T00:18:59.101245+00:00 heroku[router]: at=error code=H20 desc="App boot timeout" method=GET path="/" host=website-classification-d6f350ecbf10.herokuapp.com request_id=8bbfaa5b-773a-4588-a043-9ba367a21a9b fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
ChatGPT said:
The R10 (Boot timeout) error in Heroku means that your app failed to bind to the correct $PORT within the required 60 seconds. This often happens if:

Your Flask app isn't explicitly set to listen on $PORT.
Your Procfile is incorrect.
The app is stuck on something during startup.
🔧 Fix: Make sure your Flask app is binding to $PORT
Modify your app.py so that it correctly listens on $PORT:

python
Copy
Edit
import os
from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Heroku!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Heroku-assigned port or 5000 locally
    app.run(host="0.0.0.0", port=port)
🔧 Fix: Ensure your Procfile is correct
Your Procfile should look like this (no file extension, just Procfile):

makefile
Copy
Edit
web: gunicorn app:app
If your app.py is inside a folder, adjust accordingly, e.g., web: gunicorn myfolder.app:app.

🔧 Fix: Check Your Dependencies
Ensure gunicorn is in requirements.txt:

txt
Copy
Edit
Flask==3.1.0
gunicorn==23.0.0
Then push changes:

sh
Copy
Edit
git add .
git commit -m "Fixed Heroku port binding issue"
git push heroku main
After redeployment, run:

sh
Copy
Edit
heroku logs --tail
Let me know if you're still seeing issues! 🚀






You said:
Now I get this error:
2025-02-23T00:24:48.701447+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/" host=website-classification-d6f350ecbf10.herokuapp.com request_id=1f326d69-81dc-4837-9443-8a0dce6348a7 fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:24:49.065365+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/favicon.ico" host=website-classification-d6f350ecbf10.herokuapp.com request_id=a9d17c60-977d-4427-b6e0-c7f9c9300f51 fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:24:49.409583+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/" host=website-classification-d6f350ecbf10.herokuapp.com request_id=c6d8b598-bc21-464e-90eb-28e802fd9fc9 fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
2025-02-23T00:24:49.882623+00:00 heroku[router]: at=error code=H10 desc="App crashed" method=GET path="/" host=website-classification-d6f350ecbf10.herokuapp.com request_id=59e79428-36be-4891-81c0-306f65bb3ae4 fwd="140.254.77.225" dyno= connect= service= status=503 bytes= protocol=https
(venv) andrewrosenblum@Andrews-MacBook-Air-5 HackAI Project % 
ChatGPT said:
🔍 Fixing Heroku H10 (App Crashed) Error
The H10 error means your app failed to start. Let's troubleshoot and fix it.

🔎 Step 1: Check Heroku Logs
Run the following command to get more details on why your app crashed:

sh
Copy
Edit
heroku logs --tail
Look for tracebacks or errors that indicate what's failing.

🔧 Step 2: Ensure gunicorn is Installed & Working
Open your requirements.txt and ensure gunicorn is listed:

ini
Copy
Edit
Flask==3.1.0
gunicorn==23.0.0
Then, run:

sh
Copy
Edit
pip install -r requirements.txt
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Ensure gunicorn is installed"
git push heroku main
🔧 Step 3: Fix Procfile
Make sure your Procfile (in the project root, with no file extension) contains:

makefile
Copy
Edit
web: gunicorn app:app
If your app.py is inside a folder, adjust accordingly:

makefile
Copy
Edit
web: gunicorn myfolder.app:app
Commit and push:

sh
Copy
Edit
git add Procfile
git commit -m "Fix Procfile"
git push heroku main
🔧 Step 4: Make Sure Your Flask App Binds to $PORT
Modify app.py to explicitly bind to $PORT:

python
Copy
Edit
import os
from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Heroku!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Heroku-assigned port
    app.run(host="0.0.0.0", port=port)
Then commit & push:

sh
Copy
Edit
git add app.py
git commit -m "Fix port binding"
git push heroku main
🔧 Step 5: Restart & Scale Up
After pushing changes, restart the Heroku dynos:

sh
Copy
Edit
heroku restart
heroku ps:scale web=1
Then check logs again:

sh
Copy
Edit
heroku logs --tail
If you still see H10, copy-paste the full log output here, and I’ll help debug further! 🚀






You said:
Make a readme.md file for this code:
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
    port = int(os.environ.get("PORT", 5000))  # Use Heroku-assigned port or 5000 locally
    app.run(host="0.0.0.0", port=port)
ChatGPT said:
Readme

Answer in chat instead
I've created a README file detailing the setup, usage, and deployment instructions for your Flask-based face recognition app. Let me know if you need any modifications or additional sections! 🚀











Search


ChatGPT can make mistakes. Check important info.
?


Readme







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
