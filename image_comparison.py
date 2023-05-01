# import cv2
# import face_recognition
#----------------------------------------------------------------OLD CODE
# img = cv2.imread("images/krisha.jpg")
# rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img_encoding = face_recognition.face_encodings(rgb_img)[0]

# img2 = cv2.imread("images/cdp sir.jpg")
# rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

# result = face_recognition.compare_faces([img_encoding], img_encoding2)
# print("Result: ", result)
#------------------------------------------------------------
# cv2.imshow("Img", img)
# cv2.imshow("Img 2", img2)
# cv2.waitKey(0)
#_______________________________________________________________NEW CODE


import cv2
import face_recognition
import os
from PIL import ImageChops
import numpy as np
import cv2
import face_recognition
import os

def TakeImages():
    check_haarcascade_frontalface_default()
    assure_path_exists("TrainingImage/")
    serial = 0
    exists = os.path.isfile("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    sampleNum = 0
    while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                sampleNum = sampleNum + 1
            
                cv2.imwrite("TrainingImage\ " + name + "." + str(serial) + "." + id + '.' + str(sampleNum) + ".jpg",
                            gray[y:y + h, x:x + w])
                
                cv2.imshow('Taking Images', img)
            
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

            cap.release()
            cv2.destroyAllWindows()

            
# Load all images from the database and encode them
database_path = "./images/"
known_face_encodings = []
known_face_names = []

for filename in os.listdir(database_path):
    image_path = os.path.join(database_path, filename)
    image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(os.path.splitext(filename)[0])

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Initialize the face detection cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Loop over the video frames
while True:
    # Capture a frame from the video feed
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using the cascade classifier
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop over the detected faces
    for (x, y, w, h) in faces:
        # Extract the face image from the frame
        face_image = frame[y:y+h, x:x+w]

        # Encode the face image
        face_encodings = face_recognition.face_encodings(face_image)

        if len(face_encodings) > 0:
            # Face encoding(s) were found
            first_face_encoding = face_encodings[0]

            # Compare the face encoding to all known face encodings
            matches = face_recognition.compare_faces(known_face_encodings, first_face_encoding)

            # Calculate the distances between the face encoding and all known face encodings3
            face_distances = face_recognition.face_distance(known_face_encodings, first_face_encoding)

            # Find the index of the best match
            best_match_index = face_distances.argmin()

            # Get the name of the best match
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            else:
                name = "Unknown"

            # Calculate the percentage match
            percent_match = (1 - face_distances[best_match_index]) * 100

            # Draw a rectangle around the detected face and display the name and percentage match
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 2)
            cv2.putText(frame, "{} ({:.2f}%)".format(name, percent_match), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            for i in range(5):  # capture 5 images
                ret, img = cap.read()
                if ret:
                    cv2.imwrite(f'image_{i}.jpg', img)  # save image
                else:
                    print('Error capturing image')
                break

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
