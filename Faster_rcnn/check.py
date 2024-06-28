import numpy as np
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from model import get_model
from dataset import CustomCSVDataset, CustomTransform

def calculate_metrics(data_loader, class_mapping):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=len(class_mapping) + 1)
    model.load_state_dict(torch.load(r"Train\Faster_rcnn\fastercnn_state_dict.pth"))
    model.to(device)
    model.eval()

    # Metrics initialization
    TP = np.zeros(len(class_mapping) + 1, dtype=int)  # True Positives
    FP = np.zeros(len(class_mapping) + 1, dtype=int)  # False Positives
    FN = np.zeros(len(class_mapping) + 1, dtype=int)  # False Negatives

    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                # Convert tensors to numpy arrays for easier manipulation
                pred_boxes = output['boxes'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()

                target_boxes = targets[i]['boxes'].cpu().numpy()
                target_labels = targets[i]['labels'].cpu().numpy()

                # Assume you have a function that calculates TP, FP, FN for each sample
                sample_TP, sample_FP, sample_FN = evaluate_sample(target_boxes, target_labels, pred_boxes, pred_labels, pred_scores)
                TP += sample_TP
                FP += sample_FP
                FN += sample_FN

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * (precision * recall) / (precision + recall)

        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1_score)

# Helper function to evaluate single image predictions
def evaluate_sample(target_boxes, target_labels, pred_boxes, pred_labels, pred_scores, iou_threshold=0.5):
    # Initialize counters
    TP = 0
    FP = 0
    FN = 0

    # Thresholding predictions by score
    threshold_indices = pred_scores > iou_threshold
    pred_boxes = pred_boxes[threshold_indices]
    pred_labels = pred_labels[threshold_indices]
    pred_scores = pred_scores[threshold_indices]

    # True positives calculation requires matching
    matched = set()  # To keep track of which ground-truth boxes have been matched

    # Iterate over all predicted boxes
    for i, p_box in enumerate(pred_boxes):
        p_label = pred_labels[i]

        # For each predicted box, compare with all ground-truth boxes
        for j, t_box in enumerate(target_boxes):
            t_label = target_labels[j]

            # Check for label match
            if p_label == t_label:
                # Calculate IoU
                iou = compute_iou(p_box, t_box)

                # If IoU exceeds the threshold and target box not already matched
                if iou > iou_threshold and j not in matched:
                    TP += 1
                    matched.add(j)
                    break
        else:
            # If no match found, it's a false positive
            FP += 1

    # All unmatched ground-truth boxes are false negatives
    FN = len(target_boxes) - len(matched)

    return TP, FP, FN

def compute_iou(box1, box2):
    """Compute the intersection over union of two boxes."""
    # Coordinates of the intersection box
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Union area
    union_area = box1_area + box2_area - intersection_area

    # Compute the IoU
    iou = intersection_area / union_area

    return iou
# Dataset and DataLoader setup
image_path_valid = r"C:\Graduation_Project\Train\Faster_rcnn\Final-dataset-grad-milestone3.v20i.tensorflow\valid"
image_path_valid_annotations = r"C:\Graduation_Project\Train\Faster_rcnn\Final-dataset-grad-milestone3.v20i.tensorflow\valid\_annotations.csv"
transformations = CustomTransform(transforms.Compose([
    transforms.ToTensor()
]))
class_mapping = {
    'Car': 1,
    'Truck': 2,
    'Bus': 3,
    'MotorCycle': 4,
    'Toktok': 5,
}

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)

validation_dataset = CustomCSVDataset(root=image_path_valid, annotation_file=image_path_valid_annotations, transforms=transformations, class_mapping=class_mapping)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

calculate_metrics(validation_loader, class_mapping)
