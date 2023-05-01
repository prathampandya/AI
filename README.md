# AI
AI project Sem 6
The code provided is an implementation of face recognition using OpenCV and face_recognition libraries in Python. The main functionality of the code is to detect faces in a video feed and recognize the faces using a pre-trained model.
The code begins by importing the required libraries, including cv2 and face_recognition. Then it loads all images from the database directory and encodes them using face_recognition.face_encodings. The known face encodings and names are stored in two separate lists.
After loading the images and encoding them, the code initializes the video capture object using cv2.VideoCapture. It also initializes the face detection cascade classifier using cv2.CascadeClassifier.
The code then enters an infinite loop that reads frames from the video feed using cap.read(). It converts each frame to grayscale and detects faces in the frame using the face detection cascade classifier. For each detected face, the code extracts the face image, encodes it using face_recognition.face_encodings, and compares the encoding to all known face encodings using face_recognition.compare_faces.
The code then calculates the distance between the face encoding and all known face encodings using face_recognition.face_distance. It finds the index of the best match, gets the name of the best match, and calculates the percentage match.
Finally, the code draws a rectangle around the detected face, displays the name and percentage match, and saves five images of the face with unique names. It also exits the loop if the 'q' key is pressed.
