import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor
from PIL import Image
import os
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Set up the model with custom number of classes
def load_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(r"C:\Graduation_Project\Train\Faster_rcnn\fastercnn_state_dict.pth"))
    model.eval()
    model.cuda()
    return model

# Transform image for model input
def transform(image):
    image = to_tensor(image).cuda()
    return image

# Generate detections in COCO format
def get_coco_predictions(model, directory):
    results = []
    image_id = 0
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # check for image files
            path = os.path.join(directory, filename)
            image = Image.open(path).convert("RGB")
            image = transform(image)
            with torch.no_grad():
                prediction = model([image])
            for idx in range(len(prediction[0]['labels'])):
                box = prediction[0]['boxes'][idx].cpu().numpy()
                score = prediction[0]['scores'][idx].item()
                label = prediction[0]['labels'][idx].item()
                results.append({
                        "image_id": image_id,
                        "category_id": label,
                        "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],  # convert to Python float
                        "score": float(score)  # convert to Python float
                    })
            image_id += 1
    return results

# Evaluate predictions using COCO API
def evaluate_coco_format(ground_truths, detections):
    with open('detections.json', 'w') as f:
        json.dump(detections, f)
    cocoGt = COCO(ground_truths)  # Load ground truth annotations
    cocoDt = cocoGt.loadRes('detections.json')  # Load detection results
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats

# Main execution
if __name__ == "__main__":
    num_classes = 6  # Set the number of classes
    model = load_model(num_classes)  # Load the model with trained weights
    
    # Directory containing validation images
    validation_directory = r"C:\Graduation_Project\Train\Faster_rcnn\final_dataset_coco\valid"
    
    # Get predictions in COCO format
    predictions = get_coco_predictions(model, validation_directory)
    
    # Path to your COCO format ground truth annotations
    ground_truth_annotations = r"C:\Graduation_Project\Train\Faster_rcnn\final_dataset_coco\valid\_annotations.coco.json"
    
    # Evaluate predictions
    evaluation_results = evaluate_coco_format(ground_truth_annotations, predictions)
    print("COCO Evaluation Results:", evaluation_results)
