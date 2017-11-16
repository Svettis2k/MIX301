import aubio
import numpy as np
import aubio_util
from threading import Lock
from flask import Flask, render_template
from flask_socketio import SocketIO, emit


app = Flask(__name__)
app.config["SECRET_KEY"] = "mixmaster301"
socketio = SocketIO(app)

is_pitch_detecting = False
p_detection_thread = None
thread_lock = Lock()


def emit_pitch_detections():
    stream, p_detection = aubio_util.setup_detection()

    global is_pitch_detecting
    while is_pitch_detecting:
        data = stream.read(1024)
        samples = np.fromstring(data, dtype=aubio.float_type)
        pitch = p_detection(samples)[0]

        # Compute the energy (volume) of the current frame.
        volume = np.sum(samples ** 2) / len(samples)

        # Format the volume output so that at most it has six decimal numbers.
        volume = "{:.6f}".format(volume)

        note = str(aubio_util.frequency_to_note(pitch))
        pitch = str(pitch)
        volume = str(volume)

        detection = {"note": note, "pitch": pitch, "volume": volume}

        socketio.emit("pitch_detection_response", detection, namespace="/test")
        socketio.sleep(0.01)


@app.route("/")
def index():
    """ Serve index template """
    return render_template("index.html")


@app.route("/pitch-detection", methods=["GET"])
def pitch_detection():
    return render_template("pitch-detection.html")


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


if __name__ == "__main__":
    print("Starting http server...")
    socketio.run(app, host='localhost', port=8080, debug=True)
