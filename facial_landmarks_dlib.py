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


def draw_facial_landmarks(image, p_path):
    drawn_image = image

    # Load the dlib detector and predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p_path)

    # Detect face with dlib
    detections = detector(image, 1)
    for d in detections:
        # Predict landmark coordinates with dlib
        s = predictor(image, d)

        # Draw a dot for every point found
        drawn_image = util.draw_points_copy(drawn_image, shape_to_coordinates(s))

    return drawn_image


if __name__ == "__main__":
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
    input_image = cv2.imread(image_path)

    util.show_image("Facial landmarks: dlib", draw_facial_landmarks(input_image, predictor_path))
