import os
import cv2
import detect_faces_opencv as detect
import body_segmentation as body
import util
import facial_landmarks_dlib as landmarks
from flask import Flask, render_template, request, make_response, send_from_directory


app = Flask(__name__)


@app.route("/")
def index():
    """ Serve index template """
    return render_template("index.html")


@app.route("/detect-faces", methods=["GET"])
def detect_faces():
    return render_template("detect-faces.html")


@app.route("/facial-landmarks", methods=["GET"])
def facial_landmarks():
    return render_template("facial-landmarks.html")


@app.route("/body-segmentation", methods=["GET"])
def body_segmentation():
    return render_template("body-segmentation.html")


@app.route("/favicon.ico", methods=["GET"])
def favicon():
    return send_from_directory(os.path.join(app.root_path, "templates"), 'favicon.ico',
                               mimetype='image/vnd.microsoft.icon')


@app.route("/activate-function", methods=["POST"])
def activate_function():
    data = request.get_json()

    method = data["method"]
    base64_str = data["image"]
    base64_prefix, base64_str = util.strip_base64_prefix(base64_str)
    image = util.base64_to_image(base64_str)

    if method == "detect":
        cascade_path = os.path.join(app.root_path, "data", "haarcascade_frontalface_default.xml")
        drawn_image = detect.draw_face_rects(image, cascade_path)
        drawn_image = cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB)
        base64_str = util.append_base64_prefix(base64_prefix, util.image_to_base64(drawn_image))

    elif method == "landmarks":
        predictor_path = os.path.join(app.root_path, "data", "shape_predictor_68_face_landmarks.dat")
        drawn_image = landmarks.draw_facial_landmarks(image, predictor_path)
        drawn_image = cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB)
        base64_str = util.append_base64_prefix(base64_prefix, util.image_to_base64(drawn_image))

    elif method == "segmentation":
        fpp_key = data["fpp_key"]
        fpp_secret = data["fpp_secret"]
        base64_str = util.append_base64_prefix(
            base64_prefix, body.get_segmented_image(base64_str, fpp_key, fpp_secret, use_base64=True))

    return make_response(base64_str, 200)


if __name__ == "__main__":
    app.run()
