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




class CocoDetectionWithTransforms(CocoDetection):
    
    def __init__(self, image_directory_path: str, transform=None, train: bool = True):
        annotation_file_path = os.path.join(image_directory_path, "_annotations.coco.json")
        assert os.path.exists(annotation_file_path), f"Annotation file not found: {annotation_file_path}"
        with open(annotation_file_path, 'r') as f:
            annotations = json.load(f)
            if 'annotations' not in annotations:
                raise KeyError(f"The key 'annotations' is missing in the annotation file: {annotation_file_path}")
        super(CocoDetectionWithTransforms, self).__init__(image_directory_path, annotation_file_path)
        self.transform = transform

    def __getitem__(self, idx):
        images, annotations = super(CocoDetectionWithTransforms, self).__getitem__(idx)
        image_id = self.ids[idx]
        if not annotations:
            annotations = []
        annotations = {'image_id': image_id, 'annotations': annotations}

        images = np.array(images) / 255.0  # Scale image to [0, 1]
        images = images.astype(np.float32)  # Ensure the image is float32

        if self.transform:
            augmented = self.transform(image=images)
            images = augmented['image']

        images = torch.tensor(images).permute(2, 0, 1)

        bboxes = []
        labels = []
        for ann in annotations['annotations']:
            bbox = ann['bbox']
            bbox = [
                bbox[0] / images.shape[2],  # Normalize x_min
                bbox[1] / images.shape[1],  # Normalize y_min
                bbox[2] / images.shape[2],  # Normalize width
                bbox[3] / images.shape[1]   # Normalize height
            ]
            label = ann['category_id']
            bboxes.append(bbox)
            labels.append(label)
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)  # Ensure labels are long

        return images, bboxes, labels

