import cv2
import numpy as np
import os


COLOR = (50, 0, 255)


def quit_if_file_arg_is_invalid(file_path, error_message):
    """ Check if a file exists at path. Quit and provide error message if it does not. """
    if file_path is None or not os.path.exists(file_path):
        print("Input ERROR: " + error_message)
        exit(1)


def show_image(title, img):
    """ Shows an image with a given title. Requires key-press to proceed. """
    cv2.imshow(title, img)
    cv2.waitKey(0)


def draw_contours_copy(image, contours, color=COLOR):
    """ Draws all contours onto a copy of the input image. """
    drawn = np.zeros(shape=image.shape, dtype=image.dtype)
    drawn[:] = image[:]

    cv2.drawContours(drawn, contours, -1, color, 2)

    return drawn


def draw_rect_copy(image, rect, color=COLOR):
    """ Draws a rectangle onto a copy of the input image. Rect is a dictionary like this {left, top, width, height}."""
    drawn = np.zeros(shape=image.shape, dtype=image.dtype)
    drawn[:] = image

    left = rect["left"]
    top = rect["top"]
    right = left + rect["width"]
    bottom = top + rect["height"]

    cv2.rectangle(drawn, (left, top), (right, bottom), color, 2)

    return drawn


def draw_points_copy(image, points, color=COLOR):
    """ Draws all points onto a copy of the input image. Points are (x, y)-tuples. """
    drawn = np.zeros(shape=image.shape, dtype=image.dtype)
    drawn[:] = image

    for p in points:
        cv2.circle(drawn, (p[0], p[1]), 1,  color, -1)

    return drawn


def put_text_copy(image, text, origin, color=COLOR):
    """ Puts text onto a copy of the input image. """
    drawn = np.zeros(shape=image.shape, dtype=image.dtype)
    drawn[:] = image

    cv2.putText(drawn, text, origin, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 1, cv2.LINE_AA)

    return drawn


def combine_2_images(image1, image2):
    """ Combines two images horizontally into a single window. """
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    new_width = width1 + width2

    if height1 > height2:
        new_height = height1
    else:
        new_height = height2

    shape = (new_height, new_width, 3)

    combined = np.zeros(shape=shape, dtype=image1.dtype)
    combined[0: height1, 0:width1] = image1
    combined[0: height2, width1:new_width] = image2

    return combined
