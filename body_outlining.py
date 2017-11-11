import io
import cv2
import argparse
import requests
import util
import simplejson
import numpy as np
import base64
from PIL import Image


def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    rgb_image = Image.open(io.BytesIO(image_data))
    return np.array(rgb_image)


# Define input arguments for the script
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image_path", type=str, help="Path to image file")
parser.add_argument("-k", "--api_key", type=str, help="API key for F++")
parser.add_argument("-s", "--api_secret", type=str, help="API secret F++")

# Parse the arguments
args = parser.parse_args()
image_path = args.image_path
api_key = args.api_key
api_secret = args.api_secret

# Validate the input arguments and quit if invalid
util.quit_if_file_arg_is_invalid(image_path, "Must provide a valid image file")

# Load image
image = cv2.imread(image_path)

# Connect to F++-api and provide key, secret and image file
fpp_segment_url = "https://api-us.faceplusplus.com/humanbodypp/beta/segment?api_key={k}&api_secret={s}"
r = requests.post(fpp_segment_url.format(k=api_key, s=api_secret), files={"image_file": open(image_path, "rb")})

# Load the result into a JSON-dictionary
json = simplejson.loads(r.text)

# If the JSON only has one key then it failed
if "error_message" in json:
    # Print error message and quit
    print("Error: " + json["error_message"])
    exit(1)

print(json)

outlined_image = base64_to_image(json["result"])

util.show_image("Body outlining: F++", outlined_image)
