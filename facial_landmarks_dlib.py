import argparse
import cv2
import dlib
import util


def shape_to_coordinates(shape):
    """ Converts landmark coordinates from dlib-format into (x, y)-tuples. """
    coordinates = []

    for i in range(0, 68):
        coordinates.append((shape.part(i).x, shape.part(i).y))

    return coordinates


# Define input arguments for the script
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image_path", type=str, help="Path to image file")
parser.add_argument("-p", "--shape_predictor", type=str, help="Path to 'shape_predictor_68_face_landmarks.dat' file")

# Parse the arguments
args = parser.parse_args()
image_path = args.image_path
predictor_path = args.shape_predictor

# Validate the input arguments and quit if invalid
util.quit_if_file_arg_is_invalid(image_path, "Must provide a valid image file")
util.quit_if_file_arg_is_invalid(predictor_path, "Must provide a valid 'shape_predictor_68_face_landmarks.dat' file")

# Load image
image = cv2.imread(image_path)

# Load the dlib detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Detect face with dlib
detections = detector(image, 1)
for d in detections:
    # Predict landmark coordinates with dlib
    s = predictor(image, d)

    # Draw a dot for every point found
    image = util.draw_points_copy(image, shape_to_coordinates(s))

util.show_image("Facial landmarks: dlib", image)
