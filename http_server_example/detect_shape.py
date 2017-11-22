import cv2
import io_util as util


def classify_largest_shape(image):
    # Convert image to grayscale and blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarize the image (black and white)
    binary = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1]

    # Find contours in the binarized image
    (_, contours, _) = cv2.findContours(binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    c = contours[0]

    height, width = image.shape[:2]
    if cv2.contourArea(c) > height * width * 0.99:
        c = contours[1]

    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * perimeter, True)

    # image = util.draw_contours_copy(image, [approx])
    # util.show_image("Contour", image)

    return len(approx)
