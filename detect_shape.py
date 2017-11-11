import cv2
import imutils
import argparse
import util


def draw_largest_shape(image, edges=0):
    # Convert image to grayscale and blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarize the image (black and white)
    binary = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1]

    # Find contours in the binarized image
    (_, contours, _) = cv2.findContours(binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        # Calculate the approximate contour (calculate main edges)
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * perimeter, True)

        if edges == 0 or len(approx) == edges:
            image = util.draw_contours_copy(image, [approx])
            return image

    return image


if __name__ == "__main__":
    # Define input arguments for the script
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", type=str, help="Path to image file")
    parser.add_argument("-e", "--number_of_edges", type=int, default=0, help="Number of edges of the shape to find.")

    # Parse the arguments
    args = parser.parse_args()
    image_path = args.image_path
    number_of_edges = args.number_of_edges

    # Validate the input arguments and quit if invalid
    util.quit_if_file_arg_is_invalid(image_path, "Must provide a valid image file")

    # Load and resize image
    input_image = cv2.imread(image_path)
    input_image = imutils.resize(input_image, height=600)

    util.show_image("Draw shape: ", draw_largest_shape(input_image, number_of_edges))
