import cv2
import argparse
import requests
import util
import simplejson


def get_segmented_image(image, api_key, api_secret, use_base64=False):
    # Connect to F++-api and provide key, secret and image file
    fpp_segment_url = "https://api-us.faceplusplus.com/humanbodypp/beta/segment?api_key={k}&api_secret={s}"
    if use_base64:
        r = requests.post(fpp_segment_url.format(k=api_key, s=api_secret), data={"image_base64": image})
    else:
        files = {"image_file": open(image_path, "rb")}
        r = requests.post(fpp_segment_url.format(k=api_key, s=api_secret), files=files)

    # Load the result into a JSON-dictionary
    json = simplejson.loads(r.text)

    # If the JSON only has one key then it failed
    if "error_message" in json:
        # Print error message and quit
        print("Error: " + json["error_message"])
        exit(1)

    return json["result"]


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

    segmented_image = util.base64_to_image(get_segmented_image(input_image, fpp_key, fpp_secret))

    util.show_image("Body segmentation: F++", segmented_image)
