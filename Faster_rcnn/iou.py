import numpy as np
import cv2


def calculate_metrics(all_pred_scores):
    class_metrics = {}

    # Process each image's results
    for results, mAP in all_pred_scores:
        for class_id, metrics in results.items():
            if class_id not in class_metrics:
                class_metrics[class_id] = {
                    'precision_sum': 0, 
                    'recall_sum': 0, 
                    'f1_score_sum': 0, 
                    'count': 0, 
                    'mAP_sum': 0
                }
            
            # Sum up the metrics for each class
            class_metrics[class_id]['precision_sum'] += metrics['precision']
            class_metrics[class_id]['recall_sum'] += metrics['recall']
            class_metrics[class_id]['f1_score_sum'] += metrics['f1_score']
            class_metrics[class_id]['mAP_sum'] += mAP  # Assuming mAP is per class; adjust if mAP is global
            class_metrics[class_id]['count'] += 1

    # Calculate and print the average metrics for each class
    #metrics['count']
    for class_id, metrics in class_metrics.items():
        avg_precision = metrics['precision_sum'] / metrics['count']
        avg_recall = metrics['recall_sum'] /metrics['count']
        avg_f1_score = metrics['f1_score_sum'] /metrics['count']
        avg_mAP = metrics['mAP_sum'] / metrics['count'] # Assuming mAP is per class; adjust if mAP is global
        
        print(f"Class {class_id}: Average Precision: {avg_precision:.2f}, Average Recall: {avg_recall:.2f}, Average F1-Score: {avg_f1_score:.2f}, Average mAP: {avg_mAP:.2f}")

# Example of how to use the function
# all_pred_scores = [...]  # Your list of prediction scores
# calculate_metrics(all_pred_scores)









def calculate_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area

# def evaluate_detection(gt_boxes, pred_boxes, iou_threshold=0.5):
#     class_ids = set([box[4] for box in gt_boxes] + [box[4] for box in pred_boxes])
#     results = {}
#     for class_id in class_ids:
#         gt_class_boxes = [box for box in gt_boxes if box[4] == class_id]
#         pred_class_boxes = [box for box in pred_boxes if box[4] == class_id]

#         tp = 0
#         matched_gt_indices = set()
#         for pred_box in pred_class_boxes:
#             has_match = False
#             for idx, gt_box in enumerate(gt_class_boxes):
#                 iou = calculate_iou(pred_box, gt_box)
#                 if iou >= iou_threshold and idx not in matched_gt_indices:
#                     tp += 1
#                     matched_gt_indices.add(idx)
#                     has_match = True
#                     break
#             if not has_match:
#                 tp += 0  # False positive case

#         fp = len(pred_class_boxes) - tp
#         fn = len(gt_class_boxes) - len(matched_gt_indices)

#         precision = tp / (tp + fp) if tp + fp > 0 else 0
#         recall = tp / (tp + fn) if tp + fn > 0 else 0
#         f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

#         results[class_id] = {
#             'precision': precision,
#             'recall': recall,
#             'f1_score': f1_score,
#             'TP': tp,
#             'FP': fp,
#             'FN': fn
#         }

#     # Calculating mAP
#     average_precisions = [result['precision'] for result in results.values()]
#     mAP = np.mean(average_precisions) if average_precisions else 0

#     return results, mAP


def evaluate_detection(gt_boxes, pred_boxes, iou_threshold=0.5):
    # Collect all unique class IDs from both ground truth and predicted boxes
    class_ids = set([box[4] for box in gt_boxes] + [box[4] for box in pred_boxes])
    results = {}

    for class_id in class_ids:
        # Filter ground truth and predicted boxes by class ID
        gt_class_boxes = [box for box in gt_boxes if box[4] == class_id]
        pred_class_boxes = [box for box in pred_boxes if box[4] == class_id]

        tp = 0
        fp = 0
        matched_pred_indices = set()

        # Evaluate for precision (correct detections)
        for pred_idx, pred_box in enumerate(pred_class_boxes):
            for gt_box in gt_class_boxes:
                iou = calculate_iou(pred_box, gt_box)
                if iou >= iou_threshold:
                    tp += 1
                    matched_pred_indices.add(pred_idx)
                    break
        fp = len(pred_class_boxes) - len(matched_pred_indices)

        # Evaluate for recall (missed detections)
        matched_gt_indices = set()
        for gt_box in gt_class_boxes:
            for pred_idx, pred_box in enumerate(pred_class_boxes):
                iou = calculate_iou(gt_box, pred_box)
                if iou >= iou_threshold:
                    matched_gt_indices.add(pred_idx)
                    break
        fn = len(gt_class_boxes) - len(matched_gt_indices)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results[class_id] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'TP': tp,
            'FP': fp,
            'FN': fn
        }

    # Calculate mean Average Precision (mAP) across all classes
    average_precisions = [result['precision'] for result in results.values()]
    mAP = np.mean(average_precisions) if average_precisions else 0

    return results, mAP