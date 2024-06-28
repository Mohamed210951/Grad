import os
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
#import pytorch_lightning as pl
import json
#import albumentations as A
from torchvision.models import resnet50
#from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from cococococo import CocoDetectionWithTransforms




def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    bboxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return images,bboxes, labels






model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 6)
model.load_state_dict(torch.load(r"C:\Graduation_Project\Train\Faster_rcnn\fastercnn_state_dict.pth"))

model.eval()
model = model.cuda()  

# Prepare the validation dataset and dataloader
val=r"C:\Graduation_Project\Train\Faster_rcnn\final_dataset_coco\valid"
VAL_DATASET = CocoDetectionWithTransforms(image_directory_path=val, transform=None, train=False)
VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, collate_fn=collate_fn, batch_size=4)

# Perform inference on the validation set and collect predictions
all_preds = []
all_labels = []

for batch_idx, (images, bboxes, labels) in enumerate(VAL_DATALOADER):
    images = images.cuda()  # Move images to GPU
    bboxes = [b.cuda() for b in bboxes]  # Move bboxes to GPU
    with torch.no_grad():
        num_queries = max([len(b) for b in bboxes]) 
        outputs= model(images)
    
    # outputs_class = outputs_class.softmax(-1)
    # outputs_coord = outputs_coord.cpu()
    # outputs_class = outputs_class.cpu()
    
    for img_idx, img_pred in enumerate(zip(outputs)):
        img_id = VAL_DATALOADER.dataset.ids[batch_idx * VAL_DATALOADER.batch_size + img_idx]  
        for bbox_idx,pred  in enumerate(img_pred):
           # score, label = torch.max(cls, dim=0)
            pred_box = pred['boxes'].cpu().numpy()
            pred_scorre = pred['scores'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()
            high_score_indices = pred_scorre > 0.5
            pred_boxes = pred_box[high_score_indices]
            pred_scores = pred_scorre[high_score_indices]
            pred_labels = pred_labels[high_score_indices]
            for i in range(len(pred_boxes)):
                    bbox = pred_box[i]
                    x_min, y_min, x_max, y_max = bbox
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    # COCO format expects [x_min, y_min, width, height]
                    pred2 = {
                        "image_id": img_id,
                        "category_id": int(pred_labels[i]),
                        "bbox": [x_min, y_min, x_max, y_max],
                        "score": float(pred_scores[i])
                    }
                    all_preds.append(pred2)
    all_labels.append(labels)

# Convert the predictions and labels to COCO format
VAL_DIRECTORY=r"C:\Graduation_Project\Train\Faster_rcnn\final_dataset_coco\valid"
ANNOTATION_FILE_NAME="_annotations.coco.json"
coco_gt = COCO(os.path.join(VAL_DIRECTORY, ANNOTATION_FILE_NAME))
coco_pred = coco_gt.loadRes(all_preds)

# Evaluate the predictions using COCO API
coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()