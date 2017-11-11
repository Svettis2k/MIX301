

def dict_to_coordinates(dictionary):
    """ Converts landmark coordinates from F++-dictionary into (x, y)-tuples. """
    coordinates = []

    for landmark, coordinate in dictionary.items():
        coordinates.append((coordinate['x'], coordinate['y']))

    return coordinates


def get_mouth(dictionary):
    """ Finds all landmarks which make up the mouth. """
    mouth = []
    for landmark, coordinate in dictionary.items():
        if landmark.startswith("mouth_"):
            mouth.append((coordinate['x'], coordinate['y']))
    return mouth


def get_left_eyebrow(dictionary):
    """ Finds all landmarks which make up the left eyebrow. """
    left_eyebrow = []
    for landmark, coordinate in dictionary.items():
        if landmark.startswith("left_eyebrow_"):
            left_eyebrow.append((coordinate['x'], coordinate['y']))
    return left_eyebrow


def get_left_eye(dictionary):
    """ Finds all landmarks which make up the left eye. """
    left_eye = []
    for landmark, coordinate in dictionary.items():
        if landmark.startswith("left_eye_"):
            left_eye.append((coordinate['x'], coordinate['y']))
    return left_eye


def get_left_eye_corner(dictionary):
    """ Finds the left corner of the left eye. """
    left_eye_left_corner = dictionary['left_eye_left_corner']
    return left_eye_left_corner['x'], left_eye_left_corner['y']


def get_right_eyebrow(dictionary):
    """ Finds all landmarks which make up the right eyebrow. """
    right_eyebrow = []
    for landmark, coordinate in dictionary.items():
        if landmark.startswith("right_eyebrow_"):
            right_eyebrow.append((coordinate['x'], coordinate['y']))
    return right_eyebrow


def get_right_eye(dictionary):
    """ Finds all landmarks which make up the right eye. """
    right_eye = []
    for landmark, coordinate in dictionary.items():
        if landmark.startswith("right_eye_"):
            right_eye.append((coordinate['x'], coordinate['y']))
    return right_eye


def get_right_eye_corner(dictionary):
    """ Finds the right corner of the right eye. """
    right_eye_right_corner = dictionary['right_eye_right_corner']
    return right_eye_right_corner['x'], right_eye_right_corner['y']