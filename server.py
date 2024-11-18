import torch
from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from flask_cors import CORS
from helpers.utils import set_model_info, run_example, plot_bbox, TaskType
from infer import run_inference
from collections import OrderedDict

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load YOLO model (use a pre-trained model or specify your custom model path)
model_id = 'microsoft/Florence-2-base'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(DEVICE)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
set_model_info(model, processor)


task = TaskType.OBJECT_DETECTION

@app.route('/detect', methods=['POST'])
def detect_objects():
    # Ensure an image is uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Read the image file
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')

    result = run_inference(input_img=img)


    # result = run_example(task, img)
    # plot_bbox(result[task], img)

    label_and_loc = result["<OD>"]

    saved_annotation_box = [None] * len(label_and_loc['labels'])

    # Extract bboxes and their labels, place into a list of tuples, one for each annotation
    for idx, box_coordinates in enumerate(label_and_loc['bboxes']):
        label = label_and_loc['labels'][idx]
        saved_annotation_box[idx] = (label, box_coordinates)



    # Run YOLO detection
    #results = model(img)

    # Extract bounding box data in a vectorized way
    # boxes = results[0].boxes.xywhn  # Tensor with shape [N, 4] (N boxes with x_min_n, y_min_n, width_n, height_n)
    # boxes = results[0].boxes.xyxy  # Tensor with shape [N, 4] (N boxes with x_min_n, y_min_n, width_n, height_n)
    # classes = results[0].boxes.cls  # Tensor with shape [N] (N class IDs)
    # confidences = results[0].boxes.conf  # Tensor with shape [N] (N confidence scores)

    # Prepare the detections as a list of dictionaries
    detections = [
        {
            "class": str(saved_annotation_box[i][0]),       # Class ID
            "bbox": [
                float(saved_annotation_box[i][1][0]),        #
                float(saved_annotation_box[i][1][1]),        #
                float(saved_annotation_box[i][1][2]),        #
                float(saved_annotation_box[i][1][3])         #
            ],
            "original shape":img.size
        }
        for i in range(len(saved_annotation_box))
    ]
    print(detections)
    return jsonify(detections)

# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
