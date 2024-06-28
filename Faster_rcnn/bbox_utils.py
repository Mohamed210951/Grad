import torch
import numpy as np
def bbox_iou(box1, box2):
    """
    Compute the IoU of two bounding boxes.
    """
    box1 = torch.tensor(box1, dtype=torch.float32)
    box2 = torch.tensor(box2, dtype=torch.float32)
    
    if box1.shape[-1] == 5:
        box1 = box1[..., :4]
    if box2.shape[-1] == 5:
        box2 = box2[..., :4]
    # box1=box1[:4]
    # box2=box2[:4]
    if box1.shape[1] != 4 or box2.shape[1] != 4:
       print(f"Invalid box dimensions: box1 shape {box1.shape}, box2 shape {box2.shape}")
       return np.array([0.0])
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
  
    # Get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    union_area = b1_area + b2_area - inter_area

    # Compute the IoU
    iou = inter_area / union_area

    return iou
def nms(boxes, scores, iou_threshold):
    indices = torch.ops.torchvision.nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold)
    return indices.numpy()