import cv2
import argparse
import util
import imutils

# Define input arguments for the script
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image_path", type=str, help="Path to image file")
parser.add_argument("-f", "--hcascade_face_path", type=str, help="Path to 'haarcascade_frontalface_default.xml' file")
parser.add_argument("-e", "--hcascade_eye_path", type=str, help="Path to 'haarcascade_eye_default.xml' file")

# Parse the arguments
args = parser.parse_args()
image_path = args.image_path
cascade_face_path = args.hcascade_face_path
cascade_eye_path = args.hcascade_eye_path

# Validate the input arguments and quit if invalid
util.quit_if_file_arg_is_invalid(image_path, "Must provide a valid image file")
util.quit_if_file_arg_is_invalid(cascade_face_path, "Must provide a valid 'haarcascade_frontalface_default.xml' file")
util.quit_if_file_arg_is_invalid(cascade_eye_path, "Must provide a valid 'haarcascade_eye.xml' file")

# Load the trained cascade xml-files
face_cascade = cv2.CascadeClassifier(cascade_face_path)
eye_cascade = cv2.CascadeClassifier(cascade_eye_path)

# Load the image and convert to grayscale
image = cv2.imread(image_path)
image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces (scaleFactor, minNeighbours)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

eye_rois = []

# Iterate over all found faces
for (f_left, f_top, f_width, f_height) in faces:
    # Create region of interests containing the face (gray for detection, color for drawing)
    roi_gray = gray[f_top:f_top + f_height, f_left:f_left + f_width]
    roi_color = image[f_top:f_top + f_height, f_left:f_left + f_width]

    # Detect the eyes
    eyes = eye_cascade.detectMultiScale(roi_gray)

    # Iterate over all found eyes
    for (e_left, e_top, e_width, e_height) in eyes:
        cv2.rectangle(roi_color, (e_left, e_top), (e_left + e_width, e_top + e_height), (50, 0, 255), 2)
        eye_rois.append(roi_color[e_top:e_top + e_height, e_left:e_left + e_width])

for i, eye_roi in enumerate(eye_rois):
    eye_roi = imutils.resize(eye_roi, height=300)
    util.show_image("Eye {}: ".format(i + 1), eye_roi)
