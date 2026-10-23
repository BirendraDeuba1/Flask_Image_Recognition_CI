"""Flask web application that predicts hand-sign digits from uploaded images."""
import os

from flask import Flask, render_template, request
from PIL import UnidentifiedImageError

from model import preprocess_img, predict_result

# Instantiating flask app
app = Flask(__name__)

# Reject uploads larger than 16 MB before they reach the model (Flask returns 413).
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


@app.route("/")
def main():
    """Render the home page with the image upload form."""
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def predict_image_file():
    """Preprocess the uploaded image and render the predicted digit.

    Returns HTTP 400 with an error message when no file is sent or the file
    is not a readable image.
    """
    uploaded = request.files.get("file")
    if uploaded is None or uploaded.filename == "":
        return render_template("result.html", err="No file was uploaded."), 400

    try:
        img = preprocess_img(uploaded.stream)
    except (UnidentifiedImageError, OSError, ValueError):
        return render_template("result.html", err="File cannot be processed."), 400

    pred = predict_result(img)
    return render_template("result.html", predictions=str(pred))


# Driver code
if __name__ == "__main__":
    # Debug mode is opt-in (FLASK_DEBUG=1) so the debugger is never exposed by default.
    app.run(port=9000, debug=os.environ.get("FLASK_DEBUG") == "1")
