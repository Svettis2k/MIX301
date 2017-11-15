import os
import aubio
import cv2
import numpy as np
import detect_faces_opencv as detect
import body_segmentation as body
import facial_landmarks_dlib as landmarks
import aubio_test
import util
from threading import Lock
from flask import Flask, render_template, request, make_response, send_from_directory
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config["SECRET_KEY"] = "mixmaster301"
socketio = SocketIO(app)

is_pitch_detecting = False
p_detection_thread = None
thread_lock = Lock()


def emit_pitch_detections():
    stream, p_detection = aubio_test.setup_detection()

    global is_pitch_detecting
    while is_pitch_detecting:
        data = stream.read(1024)
        samples = np.fromstring(data, dtype=aubio.float_type)
        pitch = p_detection(samples)[0]

        # Compute the energy (volume) of the current frame.
        volume = np.sum(samples ** 2) / len(samples)

        # Format the volume output so that at most it has six decimal numbers.
        volume = "{:.6f}".format(volume)

        note = str(aubio_test.frequency_to_note(pitch))
        pitch = str(pitch)
        volume = str(volume)

        detection = {"note": note, "pitch": pitch, "volume": volume}

        socketio.emit("pitch_detection_response", detection, namespace="/test")
        socketio.sleep(0.01)


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


@app.route("/pitch-detection", methods=["GET"])
def pitch_detection():
    return render_template("pitch-detection.html")


@app.route("/favicon.ico", methods=["GET"])
def favicon():
    directory = os.path.join(app.root_path, "templates")
    return send_from_directory(directory, 'favicon.ico', mimetype='image/vnd.microsoft.icon')


@socketio.on("start_pitch_detection", "/test")
def start_pitch_detection():
    global is_pitch_detecting
    is_pitch_detecting = True

    global p_detection_thread
    if p_detection_thread is None:
        p_detection_thread = socketio.start_background_task(emit_pitch_detections)


@socketio.on("stop_pitch_detection", "/test")
def stop_pitch_detection():
    with thread_lock:
        global is_pitch_detecting
        is_pitch_detecting = False
        global p_detection_thread
        p_detection_thread = None

        emit("pitch_detection_stopped", {"message": "Stopped..."})


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
    print("Starting http server...")
    socketio.run(app, host='localhost', port=8080, debug=True)
