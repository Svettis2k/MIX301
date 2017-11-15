import argparse
import cv2
import util
import body_segmentation as bs


def contour_to_coordinates(contour):
    coordinates = []

    for c in contour:
        x = c[0][0]
        y = c[0][1]
        coordinates.append((x, y))

    return coordinates


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

    # Load the image and convert to base64
    input_image = cv2.imread(image_path)
    base64_str = util.image_to_base64(input_image)

    base64_str = bs.get_segmented_image(base64_str, fpp_key, fpp_secret, use_base64=True)

    image = util.base64_to_image(base64_str)
    util.show_image("BS", image)
    binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]
    util.show_image("BS", binary)

    (_, contours, _) = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    perimeter = cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], 0.0001 * perimeter, True)

    image = util.draw_contours_copy(input_image, [approx])
    util.show_image("BS", image)
    image = util.draw_points_copy(input_image, contour_to_coordinates(approx))
    util.show_image("BS", image)
