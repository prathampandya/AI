from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import face_recognition

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded images
        image1 = request.files['image1']
        image2 = request.files['image2']

        # Open the images with PIL
        img1 = Image.open(image1)
        img2 = Image.open(image2)

        # Convert the images to numpy arrays
        img1_array = np.array(img1)
        img2_array = np.array(img2)

        # Detect faces and extract feature vectors
        face_locs1 = face_recognition.face_locations(img1_array)
        face_encs1 = face_recognition.face_encodings(img1_array, face_locs1)
        face_locs2 = face_recognition.face_locations(img2_array)
        face_encs2 = face_recognition.face_encodings(img2_array, face_locs2)

        # Compare the feature vectors
        distance = np.linalg.norm(face_encs1[0] - face_encs2[0])

        if(distance>0.37):
            # x=True
            return f"Not Similar"
        else:
            # x=False
            return f"Similar"
    else:
        return render_template('index.html')
    
if __name__ == '__main__':
    app.run(debug=True)