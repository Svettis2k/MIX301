import cv2
import argparse
import requests
import util
import simplejson
import numpy as np
import landmark_util as lu


def poly_from_landmarks(landmarks):
    # Find the convex hull indices from landmark coordinates
    hull_indices = cv2.convexHull(np.array(landmarks), returnPoints=False)

    poly = []

    for hull_index in hull_indices:
        index = hull_index[0]
        poly.append(landmarks[index])

    return poly


def calculate_skin_mean(face_image, dictionary):
    """ Finds the mean skin RGB-value of a face given its landmarks. """

    # Create the mask
    mask = np.zeros(face_image.shape[:2], np.uint8)

    # Find the distinct landmarks that make up the relevant facial features
    border_landmarks = lu.dict_to_coordinates(dictionary)
    left_eyebrow_landmarks = lu.get_left_eyebrow(dictionary)
    left_eye_landmarks = lu.get_left_eye(dictionary)
    right_eyebrow_landmarks = lu.get_right_eyebrow(dictionary)
    right_eye_landmarks = lu.get_right_eye(dictionary)
    mouth_landmarks = lu.get_mouth(dictionary)

    # Create lists for convex landmarks for every feature
    border_poly = poly_from_landmarks(border_landmarks)
    left_eyebrow_poly = poly_from_landmarks(left_eyebrow_landmarks)
    left_eye_poly = poly_from_landmarks(left_eye_landmarks)
    right_eyebrow_poly = poly_from_landmarks(right_eyebrow_landmarks)
    right_eye_poly = poly_from_landmarks(right_eye_landmarks)
    mouth_poly = poly_from_landmarks(mouth_landmarks)

    # Fill the mask and remove disturbing features
    cv2.fillConvexPoly(mask, np.int32(border_poly), (255, 255, 255))
    util.show_image("Whole face", mask)
    cv2.fillConvexPoly(mask, np.int32(left_eyebrow_poly), (0, 0, 0))
    util.show_image("Left eyebrow", mask)
    cv2.fillConvexPoly(mask, np.int32(left_eye_poly), (0, 0, 0))
    util.show_image("Left eye", mask)
    cv2.fillConvexPoly(mask, np.int32(right_eyebrow_poly), (0, 0, 0))
    util.show_image("Right eyebrow", mask)
    cv2.fillConvexPoly(mask, np.int32(right_eye_poly), (0, 0, 0))
    util.show_image("Right eye", mask)
    cv2.fillConvexPoly(mask, np.int32(mouth_poly), (0, 0, 0))
    util.show_image("Mask", mask)

    # Calculate the mean from the masked image
    return cv2.mean(face_image, mask)


def draw_facial_landmarks(image, api_key, api_secret):
    # Connect to F++-api and provide key, secret and image file
    fpp_detect_url = "https://api-us.faceplusplus.com/facepp/v3/detect?api_key={k}&api_secret={s}&return_landmark=2"
    r = requests.post(fpp_detect_url.format(k=api_key, s=api_secret), files={"image_file": open(image_path, "rb")})

    # Load the result into a JSON-dictionary
    json = simplejson.loads(r.text)

    # If the JSON only has one key then it failed
    if len(json) is 1:
        # Print error message and quit
        print("Error: " + json["error_message"])
        exit(1)

    for face in json["faces"]:
        landmark_dict = face["landmark"]
        image = util.draw_points_copy(image, lu.dict_to_coordinates(landmark_dict))

        # print(calculate_skin_mean(image, landmark_dict))

    return image


if __name__ == "__main__":
    # Define input arguments for the script
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", type=str, help="Path to image file")
    parser.add_argument("-k", "--api_key", type=str, help="API key for F++")
    parser.add_argument("-s", "--api_secret", type=str, help="API secret F++")

    # Parse the arguments
    args = parser.parse_args()
    image_path = args.image_path
    fpp_key = args.api_key
    fpp_secret = args.api_secret

    # Validate the input arguments and quit if invalid
    util.quit_if_file_arg_is_invalid(image_path, "Must provide a valid image file")

    # Load image
    input_image = cv2.imread(image_path)

    util.show_image("Facial landmarks: Face++", draw_facial_landmarks(input_image, fpp_key, fpp_secret))