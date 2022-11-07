from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PIL import Image

from models import statics

app = Flask(__name__)
CORS(app)

@app.route("/encode-text", methods=["POST", "GET"])
def encode_text():
    # Get text from request body / query string
    if request.method == "POST":
        text = request.json.get("text")
    else:
        text = request.args.get("text")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Encode text
        embedding = statics.model.encode(text).tolist()
        return jsonify(embedding)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/encode-image-url", methods=["POST", "GET"])
def encode_image_url():
    # Get image URL from request body / query string
    if request.method == "POST":
        url = request.json.get("url")
    else:
        url = request.args.get("url")

    if not url:
        return jsonify({"error": "No URL provided", "code" : 0}), 400

    # Try to download image
    try:
        image = Image.open(requests.get(url, stream=True).raw)
    except Exception as e:
        return jsonify({"error": str(e), "code" : 1}), 500

    # Try to encode image
    try:
        embedding = statics.model.encode(image, convert_to_tensor=True).tolist()
        return jsonify(embedding)
    except Exception as e:
        return jsonify({"error": str(e), "code" : 2}), 500

@app.route("/encode-image-file", methods=["POST"])
def encode_image_file():
    # Get image file from request body
    if "file" not in request.files:
        return jsonify({"error": "No file provided", "code" : 0}), 400

    # Try to download image
    try:
        image = Image.open(request.files["file"])
    except Exception as e:
        return jsonify({"error": str(e), "code" : 1}), 500

    # Try to encode image
    try:
        embedding = statics.model.encode(image, convert_to_tensor=True).tolist()
        return jsonify(embedding)
    except Exception as e:
        return jsonify({"error": str(e), "code" : 2}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)