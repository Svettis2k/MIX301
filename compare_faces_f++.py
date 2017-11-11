import cv2
import argparse
import requests
import util
import imutils
import simplejson


def compare_faces(image1, image2, api_key, api_secret):
    # Connect to F++-api and provide key, secret and image files
    fpp_detect_url = "https://api-us.faceplusplus.com/facepp/v3/compare?api_key={k}&api_secret={s}"
    images = {"image_file1": open(image_path1, "rb"), "image_file2": open(image_path2, "rb")}
    r = requests.post(fpp_detect_url.format(k=api_key, s=api_secret), files=images)

    # Load the result into a JSON-dictionary
    json = simplejson.loads(r.text)

    # If the JSON only has one key then it failed
    if len(json) is 1:
        # Print error message and quit
        print("Error: " + json["error_message"])
        exit(1)

    # Combine the two images and write the returned confidence value as text on the image
    image = util.combine_2_images(image1, image2)
    image = util.put_text_copy(image, "Confidence: " + str(json["confidence"]), (0, 20))

    return image


if __name__ == "__main__":
    # Define input arguments for the script
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path1", type=str, help="Path to image file")
    parser.add_argument("-j", "--image_path2", type=str, help="Path to image file")
    parser.add_argument("-k", "--api_key", type=str, help="API key for F++")
    parser.add_argument("-s", "--api_secret", type=str, help="API secret for F++")

    # Parse the arguments
    args = parser.parse_args()
    image_path1 = args.image_path1
    image_path2 = args.image_path2
    fpp_key = args.api_key
    fpp_secret = args.api_secret

    # Validate the input arguments and quit if invalid
    util.quit_if_file_arg_is_invalid(image_path1, "Must provide a valid image file (-i)")
    util.quit_if_file_arg_is_invalid(image_path2, "Must provide a valid image file (-j)")

    # Load images and resize to 500 height and x width
    input_image1 = cv2.imread(image_path1)
    input_image2 = cv2.imread(image_path2)
    input_image1 = imutils.resize(input_image1, height=500)
    input_image2 = imutils.resize(input_image2, height=500)

    util.show_image("Face comparison", compare_faces(input_image1, input_image2, fpp_key, fpp_secret))
