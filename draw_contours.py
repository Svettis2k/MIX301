import cv2
import imutils
import argparse
import util

# Define input arguments for the script
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image_path", type=str, help="Path to image file")

# Parse the arguments
args = parser.parse_args()
image_path = args.image_path

# Validate the input arguments and quit if invalid
util.quit_if_file_arg_is_invalid(image_path, "Must provide a valid image file")

# Load image and resize image
image = cv2.imread(image_path)
image = imutils.resize(image, height=400)

# Convert image to grayscale and blur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 7, 15, 15)

# Detect edges
edged = cv2.Canny(gray, 30, 125)

# Find contours in the edge detected image
(_, contours, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours by size
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# For every contour draw the outline of its approximation (calculate main edges)
for c in contours:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * perimeter, True)

    if cv2.contourArea(c) > 200:
        image = util.draw_contours_copy(image, [approx])

print("Test")

util.show_image("Draw contours", image)
