from flask import Flask, request, jsonify
from PIL import Image
from tests.test_python import image

import utils
from ultralytics import YOLO
import threading
from flask_cors import CORS

from utils import run_example, plot_bbox

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load YOLO model (use a pre-trained model or specify your custom model path)
model = YOLO("FastSAM-s.pt")


task = utils.TaskType.OBJECT_DETECTION

@app.route('/detect', methods=['POST'])
def detect_objects():
    # Ensure an image is uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Read the image file
    file = request.files['image']
    img = Image.open(file.stream)


    result = run_example(task, img)
    plot_bbox(result[task], img)




    # Run YOLO detection
    results = model(img)

    # Extract bounding box data in a vectorized way
    # boxes = results[0].boxes.xywhn  # Tensor with shape [N, 4] (N boxes with x_min_n, y_min_n, width_n, height_n)
    boxes = results[0].boxes.xyxy  # Tensor with shape [N, 4] (N boxes with x_min_n, y_min_n, width_n, height_n)
    classes = results[0].boxes.cls  # Tensor with shape [N] (N class IDs)
    confidences = results[0].boxes.conf  # Tensor with shape [N] (N confidence scores)

    # Prepare the detections as a list of dictionaries
    detections = [
        {
            "class": int(classes[i].item()),       # Class ID
            "confidence": float(confidences[i].item()),  # Confidence score
            "bbox": [
                float(boxes[i, 0].item()),        # x_centre_n
                float(boxes[i, 1].item()),        # y_centre_n
                float(boxes[i, 2].item()),        # width_n
                float(boxes[i, 3].item())         # height_n
            ],
            "original shape":results[0].boxes.orig_shape
        }
        for i in range(len(classes))
    ]

    return jsonify(detections)

# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
