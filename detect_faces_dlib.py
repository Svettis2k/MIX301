import cv2
import argparse
import dlib
import util


def draw_face_rects(image):
    # Load the dlib detector
    detector = dlib.get_frontal_face_detector()

    # Detect face with dlib
    detections = detector(image, 1)

    # For every detection draw a face rectangle
    for d in detections:
        top = d.top()
        left = d.left()
        width = d.right() - left
        height = d.bottom() - top
        image = util.draw_rect_copy(image, {"left": left, "top": top, "width": width, "height": height})

    return image


if __name__ == "__main__":
    # Define input arguments for the script
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", type=str, help="Path to image file")

    # Parse the arguments
    args = parser.parse_args()
    image_path = args.image_path

    # Validate the input arguments and quit if invalid
    util.quit_if_file_arg_is_invalid(image_path, "Must provide a valid image file")

    # Load image
    input_image = cv2.imread(image_path)

    util.show_image("Detect faces: dlib", draw_face_rects(input_image))
