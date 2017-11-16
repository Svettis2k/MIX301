import argparse
import cv2
import numpy as np
import util
import body_segmentation as bs


def contour_to_coordinates(contour):
    coordinates = []

    for c in contour:
        x = c[0][0]
        y = c[0][1]
        coordinates.append((x, y))

    return coordinates


def find_body_contours(image):
    binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]

    (_, contours1, _) = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours1 = sorted(contours1, key=cv2.contourArea, reverse=True)

    perimeter = cv2.arcLength(contours1[0], True)
    approx = cv2.approxPolyDP(contours1[0], 0.0001 * perimeter, True)

    return approx


if __name__ == "__main__":
    # Define input arguments for the script
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path1", type=str, help="Path to image file")
    parser.add_argument("-j", "--image_path2", type=str, help="Path to image file")
    parser.add_argument("-k", "--api_key", type=str, help="API key for F++")
    parser.add_argument("-s", "--api_secret", type=str, help="API secret F++")

    # Parse the arguments
    args = parser.parse_args()
    image_path1 = args.image_path1
    image_path2 = args.image_path2
    fpp_key = args.api_key
    fpp_secret = args.api_secret

    # Validate the input arguments and quit if invalid
    util.quit_if_file_arg_is_invalid(image_path1, "Must provide a valid image file (-i)")
    util.quit_if_file_arg_is_invalid(image_path2, "Must provide a valid image file (-j)")

    base64_str1 = bs.get_segmented_image(image_path1, fpp_key, fpp_secret)
    base64_str2 = bs.get_segmented_image(image_path2, fpp_key, fpp_secret)

    image1 = util.base64_to_image(base64_str1)
    image2 = util.base64_to_image(base64_str2)

    contour1 = find_body_contours(image1)
    contour2 = find_body_contours(image2)

    # Load images
    input_image1 = cv2.imread(image_path1)
    util.show_image("1", input_image1)
    input_image2 = cv2.imread(image_path2)
    util.show_image("2", input_image2)

    h1, w1 = input_image1.shape[:2]
    h2, w2 = input_image2.shape[:2]

    if w1 > w2:
        w = w1
    else:
        w = w2
    if h1 > h2:
        h = h1
    else:
        h = h2

    cnt_canvas = np.zeros((h, w), np.uint8)

    left1, top1, width1, height1 = cv2.boundingRect(contour1)
    left2, top2, width2, height2 = cv2.boundingRect(contour2)

    # cnt_rect_1 = util.draw_rect_copy(cnt_canvas, {"left": left1, "top": top1, "width": width1, "height": height1})
    # cnt_rect_2 = util.draw_rect_copy(cnt_canvas, {"left": left2, "top": top2, "width": width2, "height": height2})

    src = np.array([
        [left1, top1],
        [left1 + width1, top1],
        [left1 + width1, top1 + height1],
        [left1, top1 + height1]
    ], dtype=np.float32)
    print(src)

    dst = np.array([
        [left2, top2],
        [left2 + width2, top2],
        [left2 + width2, top2 + height2],
        [left2, top2 + height2]
    ], dtype=np.float32)
    print(dst)

    inim1 = util.draw_contours_copy(input_image1, [contour1], (255, 0, 0))

    util.show_image("Contours", inim1)

    inim2 = util.draw_contours_copy(input_image2, [contour2], (255, 0, 0))

    util.show_image("Contours", inim2)

    cnt_canvas = util.draw_contours_copy(cnt_canvas, [contour1])
    M = cv2.estimateRigidTransform(src, dst, True)
    util.show_image("Contours", cnt_canvas)
    cnt_canvas = cv2.warpAffine(cnt_canvas, M, (w, h))
    util.show_image("Contours", cnt_canvas)

    cnt_canvas = util.draw_contours_copy(cnt_canvas, [contour2], (255, 0, 0))

    util.show_image("Contours", cnt_canvas)
