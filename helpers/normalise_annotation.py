import os
import json
from PIL import Image


def normalize_bbox(x, y, width, height, img_width, img_height):
    """
    Normalize the bounding box from (x, y, width, height) to (x1, y1, x2, y2)
    and scale it to 0-1000.
    """
    # Calculate x1, y1, x2, y2 from the input format
    x1 = x
    y1 = y
    x2 = x + width
    y2 = y + height

    # Normalize to the range [0, 1000]
    x1_norm = (x1 / img_width) * 1000
    y1_norm = (y1 / img_height) * 1000
    x2_norm = (x2 / img_width) * 1000
    y2_norm = (y2 / img_height) * 1000

    return x1_norm, y1_norm, x2_norm, y2_norm


def transform_annotations(input_json_dir, output_json_dir):
    """
    Transform image annotations to the Florence 2 dataset format.
    """
    # Iterate over all annotation files in the input directory
    for filename in os.listdir(input_json_dir):
        if filename.endswith('.json'):
            input_filepath = os.path.join(input_json_dir, filename)
            with open(input_filepath, 'r') as f:
                annotations_data = json.load(f)
            annotations = annotations_data['images']
            for img_annotation in annotations:
                image_name = img_annotation["image"]
                img_path = os.path.join(input_json_dir, image_name)

                # Open the image to get its width and height
                img = Image.open(img_path)
                img_width, img_height = img.size

                # Prepare the transformed annotations
                prefix = "<OD>"
                suffix_parts = []

                # Loop through all annotations
                for annotation in img_annotation["annotations"]:
                    class_name = annotation["class"]
                    bbox = annotation["boundingBox"]
                    x, y, width, height = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

                    # Normalize the bounding box
                    x1, y1, x2, y2 = normalize_bbox(x, y, width, height, img_width, img_height)

                    # Append the transformed annotation in the required format
                    suffix_parts.append(
                        f"{class_name}<loc_{int(x1):03d}><loc_{int(y1):03d}><loc_{int(x2):03d}><loc_{int(y2):03d}>")

                # Create the final output dictionary
                output_data = {
                    "image": image_name,
                    "prefix": prefix,
                    "suffix": "".join(suffix_parts)
                }

                # Write the output data to the output directory
                output_filepath = os.path.join(output_json_dir, filename.replace('.json', '.jsonl'))
                with open(output_filepath, 'a') as f:  # Open in append mode
                    json.dump(output_data, f)
                    f.write("//n")  # Write each JSON object on a new line

                print(f"Processed {filename} and saved to {output_filepath}")


# Example usage:
input_json_dir = "C://transmetric//dev//python//AI camera//trial//fine-tuned-Florence-2//Australia-vehicles-classification//dataset//train//raw_annotations"
output_json_dir = "C://transmetric//dev//python//AI camera//trial//fine-tuned-Florence-2//Australia-vehicles-classification//dataset//train//output_annotations"

transform_annotations(input_json_dir, output_json_dir)
