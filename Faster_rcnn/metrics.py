# metrics.py

import numpy as np
from bbox_utils import bbox_iou, nms
import torch
from iou import evaluate_detection,calculate_metrics
def calculate_precision_recall_f1(gt_boxes, pred_boxes, pred_scores, pred_labels, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # print(f"gt_boxes shape: {gt_boxes.shape}")
    # print(f"pred_boxes shape: {pred_boxes.shape}")

    # if gt_boxes.shape[1] != 5 or pred_boxes.shape[1] != 4:
    #     print("Invalid box dimensions")
    #     return 0, 0, 0
    if gt_boxes.shape[1] != 5 or pred_boxes.shape[1] != 4:
        print("Invalid box dimensions")
        return 0, 0, 0
    for i, gt_box in enumerate(gt_boxes):
        matched = False
        for j, pred_box in enumerate(pred_boxes):
            iou = bbox_iou(np.expand_dims(gt_box[:4], 0), np.expand_dims(pred_box[:4], 0))
            if (iou >= iou_threshold).any() and pred_labels[j] == gt_box[4]:
                if pred_scores[j] >= 0.5:
                    true_positives += 1
                    matched = True
                    break
        if not matched:
            false_negatives += 1

    for j, pred_box in enumerate(pred_boxes):
        match_found = False
        for i, gt_box in enumerate(gt_boxes):
            iou = bbox_iou(np.expand_dims(pred_box, 0), np.expand_dims(gt_box, 0))
            if iou >= iou_threshold and pred_labels[j] == gt_box[4]:
                match_found = True
                break
        if not match_found:
            false_positives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

def compute_ap(recall, precision):
    recall = np.concatenate(([0.], recall, [1.]))
    precision = np.concatenate(([0.], precision, [0.]))

    for i in range(len(precision) - 1, 0, -1):
        precision[i-1] = np.maximum(precision[i-1], precision[i])

    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])

    return ap

def calculate_map(all_gt_boxes, all_pred_boxes, all_pred_scores, all_pred_labels, iou_threshold=0.5):
    all_gt_boxes = [box for sublist in all_gt_boxes for box in sublist]
    all_pred_boxes = [box for sublist in all_pred_boxes for box in sublist]
    all_pred_scores = [score for sublist in all_pred_scores for score in sublist]
    all_pred_labels = [label for sublist in all_pred_labels for label in sublist]

    unique_classes = np.unique(all_gt_boxes[:, 4])
    aps = []

    for cls in unique_classes:
        cls_gt_boxes = all_gt_boxes[all_gt_boxes[:, 4] == cls]
        cls_indices = [i for i, l in enumerate(all_pred_labels) if l == cls]
        cls_pred_boxes = np.array(all_pred_boxes)[cls_indices]
        cls_pred_scores = np.array(all_pred_scores)[cls_indices]

        sorted_indices = np.argsort(-cls_pred_scores)
        cls_pred_boxes = cls_pred_boxes[sorted_indices]
        cls_pred_scores = cls_pred_scores[sorted_indices]

        keep_indices = nms(cls_pred_boxes, cls_pred_scores, iou_threshold)
        cls_pred_boxes = cls_pred_boxes[keep_indices]
        cls_pred_scores = cls_pred_scores[keep_indices]

        precision, recall, _ = calculate_precision_recall_f1(cls_gt_boxes, cls_pred_boxes, cls_pred_scores, [cls]*len(cls_pred_scores))
        ap = compute_ap(recall, precision)
        aps.append(ap)

    return np.mean(aps)

def evaluate(model, data_loader, device):
    model.eval()
    all_gt_boxes = []
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            outputs = model(images)

            for i, output in enumerate(outputs):
                pred_boxes = output['boxes'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()
                high_score_indices = pred_scores > 0.5
                pred_boxes = pred_boxes[high_score_indices]
                pred_scores = pred_scores[high_score_indices]
                pred_labels = pred_labels[high_score_indices]

                pred_boxes = [[box[0], box[1], box[2], box[3], label] for box, label in zip(pred_boxes, pred_labels)]
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                gt_labels = targets[i]['labels'].cpu().numpy()
                gt_boxes_labeled = np.hstack((gt_boxes, np.expand_dims(gt_labels, axis=1)))
                gt_boxes = [[box[0], box[1], box[2], box[3], label] for box, label in zip(gt_boxes, gt_labels)]
                
                # print(f"gt_boxes_labeled shape: {gt_boxes_labeled.shape}")
                # print(f"pred_boxes shape: {pred_boxes.shape}")
                
                
                all_pred_scores.append(evaluate_detection(gt_boxes, pred_boxes))

    calculate_metrics(all_pred_scores)
        
        
    
    all_gt_boxes = np.array(all_gt_boxes)
    all_pred_boxes = np.array(all_pred_boxes)
    all_pred_scores = np.array(all_pred_scores)
    all_pred_labels = np.array(all_pred_labels)

   # precision, recall, f1_score = calculate_precision_recall_f1(all_gt_boxes, all_pred_boxes, all_pred_scores, all_pred_labels)
    #mAP = calculate_map(all_gt_boxes, all_pred_boxes, all_pred_scores, all_pred_labels)
#     results, mAP = evaluate_detection(gt_boxes, pred_boxes)
#     print(results)
#     print(mAP)
#    # return precision, recall, f1_score, mAP
# import torch
# import numpy as np
# from collections import defaultdict
# from sklearn.metrics import precision_recall_fscore_support

# def iou(box1, box2):
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])

#     inter_area = max(0, x2 - x1) * max(0, y2 - y1)
#     box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
#     iou = inter_area / float(box1_area + box2_area - inter_area)
#     return iou

# def evaluate(model, data_loader, device, num_classes=5, iou_threshold=0.5):
#     model.eval()
#     all_gt_labels = []
#     all_pred_labels = []
#     all_pred_scores = []

#     with torch.no_grad():
#         for images, targets in data_loader:
#             images = list(image.to(device) for image in images)
#             outputs = model(images)

#             for i, output in enumerate(outputs):
#                 pred_boxes = output['boxes'].cpu().numpy()
#                 pred_scores = output['scores'].cpu().numpy()
#                 pred_labels = output['labels'].cpu().numpy()

#                 gt_boxes = targets[i]['boxes'].cpu().numpy()
#                 gt_labels = targets[i]['labels'].cpu().numpy()

#                 for j, gt_box in enumerate(gt_boxes):
#                     best_iou = 0
#                     best_pred_idx = -1
#                     for k, pred_box in enumerate(pred_boxes):
#                         current_iou = iou(gt_box, pred_box)
#                         if current_iou > best_iou:
#                             best_iou = current_iou
#                             best_pred_idx = k
                    
#                     if best_iou >= iou_threshold:
#                         all_pred_labels.append(pred_labels[best_pred_idx])
#                         all_gt_labels.append(gt_labels[j])
#                         all_pred_scores.append(pred_scores[best_pred_idx])
#                     else:
#                         # False negative (missed ground truth)
#                         all_gt_labels.append(gt_labels[j])
#                         all_pred_labels.append(-1)  # -1 indicates no prediction

#                 # False positives (predictions with no matching ground truth)
#                 unmatched_preds = set(range(len(pred_boxes))) - {best_pred_idx}
#                 for idx in unmatched_preds:
#                     all_pred_labels.append(pred_labels[idx])
#                     all_gt_labels.append(-1)  # -1 indicates no ground truth

#     all_pred_labels = torch.tensor(all_pred_labels)
#     all_gt_labels = torch.tensor(all_gt_labels)

#     # Calculate metrics
#     precision, recall, f1, _ = precision_recall_fscore_support(all_gt_labels, all_pred_labels, average='weighted', labels=list(range(num_classes)))
#     accuracy = (all_pred_labels == all_gt_labels).float().mean().item()

#     return {
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1_score': f1
#     }

# Example usage
# Assuming you have a trained model, data_loader, and device (e.g., 'cuda' or 'cpu')
#results = evaluate(model, data_loader, device)

