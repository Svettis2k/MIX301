import cv2


def __is_quadrilateral(contour):
    """ Returns True if, and only if, the contour has 4 edges. """

    if len(contour) == 4:
        return True
    else:
        print("Contour is not a quadrilateral")
        return False


def is_square(contour):
    """ Returns True if the contour has an aspect ratio which is approximately 1 (0.85 <= ratio <= 1.15). """

    if not __is_quadrilateral(contour):
        return False

    (left, top, width, height) = cv2.boundingRect(contour)
    ratio = width / float(height)

    return 0.85 <= ratio <= 1.15


def is_rectangle(contour):
    """ Returns True if the contour has an aspect ratio which greatly differs from 1 (0.85 <= ratio >= 1.15). """
    if not __is_quadrilateral(contour):
        return False

    (left, top, width, height) = cv2.boundingRect(contour)
    ratio = width / float(height)

    return 0.85 <= ratio >= 1.15
