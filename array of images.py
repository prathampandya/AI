import cv2
import face_recognition
import os
# Create a dictionary to store encoding vectors for each image
encodings_dict = {}

# Define the path to the directory containing the images
images_dir = './images/'

# Loop through all the image files in the directory
for filename in os.listdir(images_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        
        # Load the image
        image_path = os.path.join(images_dir, filename)
        image = cv2.imread(image_path)
        
        # Locate the face in the image
        face_locations = face_recognition.face_locations(image)
        
        # Generate the face encoding vector
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        # Store the encoding vector in the dictionary
        encodings_dict[filename] = face_encodings[0]

# Print the encoding vectors
for filename, encoding in encodings_dict.items():
    print(f"{filename}: {encoding}")