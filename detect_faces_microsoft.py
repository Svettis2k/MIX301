import cv2
import argparse
import requests
import util
import simplejson

# Define input arguments for the script
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image_path", type=str, help="Path to image file")
parser.add_argument("-k", "--api_key", type=str, help="API key for Microsoft Azure Face API")

# Parse the arguments
args = parser.parse_args()
image_path = args.image_path
api_key = args.api_key

# Validate the input arguments and quit if invalid
util.quit_if_file_arg_is_invalid(image_path, "Must provide a valid image file")

# Load image
image = cv2.imread(image_path)

# Connect to Microsoft-api and provide key, content-type and image file
microsoft_detect_url = "https://northeurope.api.cognitive.microsoft.com/face/v1.0/detect"
headers = {"Content-Type": "application/octet-stream", "Ocp-Apim-Subscription-Key": api_key}
r = requests.post(microsoft_detect_url, open(image_path, "rb"), headers=headers)

# Load the result into a JSON-dictionary
json = simplejson.loads(r.text)

# If the JSON only has one key then it failed
if len(json) is 1:
    # Print error message and quit
    print("Error: " + json["error"]["message"])
    exit(1)

for face in json:
    image = util.draw_rect_copy(image, face["faceRectangle"])

util.show_image("Detect faces: Microsoft", image)
