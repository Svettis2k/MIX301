import cv2
import imutils
import argparse
import util
import contour_util as cu

# Define input arguments for the script
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image_path", type=str, help="Path to image file")

# Parse the arguments
args = parser.parse_args()
image_path = args.image_path

# Validate the input arguments and quit if invalid
util.quit_if_file_arg_is_invalid(image_path, "Must provide a valid image file")

# Load and resize image
image = cv2.imread(image_path)
image = imutils.resize(image, height=600)

# Convert image to grayscale and blur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Binarize the image (black and white)
binary = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1]

# Find contours in the binarized image
(_, contours, _) = cv2.findContours(binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours by area
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Define all contour shapes as arrays
triangles = []
quads = []
squares = []
rectangles = []
pentagons = []
hexagons = []
heptagons = []
octagons = []
nonagons = []
decagons = []
circles = []

for c in contours:
    # Ignore small contours
    if cv2.contourArea(c) < 400:
        # We can break the loop since contours are sorted by area
        break

    # Calculate the approximate contour (calculate main edges)
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * perimeter, True)

    # Based on the number of edges we can put the contour into the correct array
    if len(approx) == 3:
        triangles.append(approx)
    elif len(approx) == 4:
        quads.append(approx)
        # In addition to being a quadrilateral, check if square or rectangle
        if cu.is_square(approx):
            squares.append(approx)
        if cu.is_rectangle(approx):
            rectangles.append(approx)
    elif len(approx) == 5:
        pentagons.append(approx)
    elif len(approx) == 6:
        hexagons.append(approx)
    elif len(approx) == 7:
        heptagons.append(approx)
    elif len(approx) == 8:
        octagons.append(approx)
    elif len(approx) == 9:
        nonagons.append(approx)
    elif len(approx) == 10:
        decagons.append(approx)
    else:
        circles.append(approx)

# If the array contains any elements draw the contours on a copy of the image and show
if len(triangles) > 0:
    triangle_image = util.draw_contours_copy(image, triangles)
    util.show_image("Triangles", triangle_image)

if len(quads) > 0:
    quad_image = util.draw_contours_copy(image, quads)
    util.show_image("Quads", quad_image)

if len(squares) > 0:
    square_image = util.draw_contours_copy(image, squares)
    util.show_image("Squares", square_image)

if len(rectangles) > 0:
    rect_image = util.draw_contours_copy(image, rectangles)
    util.show_image("Rectangles", rect_image)

if len(pentagons) > 0:
    pentagon_image = util.draw_contours_copy(image, pentagons)
    util.show_image("Pentagons", pentagon_image)

if len(hexagons) > 0:
    hexagon_image = util.draw_contours_copy(image, hexagons)
    util.show_image("Hexagons", hexagon_image)

if len(heptagons) > 0:
    heptagon_image = util.draw_contours_copy(image, heptagons)
    util.show_image("Heptagons", heptagon_image)

if len(octagons) > 0:
    octagon_image = util.draw_contours_copy(image, octagons)
    util.show_image("Octagons", octagon_image)

if len(nonagons) > 0:
    nonagon_image = util.draw_contours_copy(image, nonagons)
    util.show_image("Nonagons", nonagon_image)

if len(decagons) > 0:
    decagon_image = util.draw_contours_copy(image, decagons)
    util.show_image("Decagons", decagon_image)

if len(circles) > 0:
    circle_image = util.draw_contours_copy(image, circles)
    util.show_image("Circles", circle_image)
