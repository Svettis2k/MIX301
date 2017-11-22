import io_util as util
import detect_shape as ds
from flask import Flask, render_template, make_response, request


app = Flask(__name__)
app.config["SECRET_KEY"] = "mixmaster301"


@app.route("/")
def index():
    """ Serve index template """
    return render_template("index.html")


@app.route("/gjettelek", methods=["GET"])
def gjettelek():
    return render_template("gjettelek.html")


@app.route("/detect-shape", methods=["POST"])
def detect_shape():
    data = request.get_json()
    base64_str = data["image"]

    base64_prefix, base64_str = util.strip_base64_prefix(base64_str)
    image = util.base64_to_image(base64_str)

    edges = ds.classify_largest_shape(image)

    return make_response(str(edges), 200)


if __name__ == "__main__":
    print("Starting http server...")
    app.run()
