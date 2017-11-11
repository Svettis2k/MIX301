import cv2
import argparse
import util


def draw_face_rects(image, c_path):
    drawn_image = image

    # Load the trained cascade xml-file
    face_cascade = cv2.CascadeClassifier(c_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces (scaleFactor, minNeighbours)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # For every face found draw a face rectangle
    for (left, top, width, height) in faces:
        drawn_image = util.draw_rect_copy(drawn_image, {"left": left, "top": top, "width": width, "height": height})

    return drawn_image


if __name__ == "__main__":
    # Define input arguments for the script
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", type=str, help="Path to image file")
    parser.add_argument("-c", "--haarcascade_path", type=str, help="Path to 'haarcascade_frontalface_default.xml' file")

    # Parse the arguments
    args = parser.parse_args()
    image_path = args.image_path
    cascade_path = args.haarcascade_path

    # Load image
    input_image = cv2.imread(image_path)

    # Validate the input arguments and quit if invalid
    util.quit_if_file_arg_is_invalid(image_path, "Must provide a valid image file")
    util.quit_if_file_arg_is_invalid(cascade_path, "Must provide a valid 'haarcascade_frontalface_default.xml' file")

    util.show_image("Detect faces: OpenCV", draw_face_rects(input_image, cascade_path))
