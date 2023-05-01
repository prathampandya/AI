# AI
AI project Sem 6

Face Recognition Using OpenCV and Face-Recognition Python Libraries
This is a Python script that uses OpenCV and Face-Recognition libraries to recognize faces from a live video stream or a set of images. The script can be used for various purposes, such as attendance systems or security systems.

Prerequisites
To use this script, you need to have the following installed:

Python 3.6 or higher
OpenCV
Face-Recognition
numpy
Pillow
How to Use
Clone this repository or download the code as a ZIP file.

Navigate to the project directory in your terminal or command prompt.

Run the following command to install the necessary libraries:

pip install opencv-python
pip install numpy
pip install face_recognition

Prepare a set of images of the people you want to recognize and save them in a directory. Make sure each image has only one face visible and the name of the file should be the name of the person.

Run the following command:

python face_recognition.py

This will open a live video stream and start recognizing faces. The name and percentage match of the recognized person will be displayed in a rectangle around their face. You can press the 'q' key to quit the stream.

Additional Features
The script also captures 5 images of the recognized person when a match is found and saves them in the same directory as the script.
The script also allows capturing images of new people by using the TakeImages() function which is currently commented out in the script.

Credits
This script was created with the help of OpenCV and Face-Recognition libraries. Special thanks to the developers of these libraries for making face recognition much simpler and easier.


[https://youtu.be/Zsk2TavQWYk](url)
