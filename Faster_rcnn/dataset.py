# import torch
# from torchvision.datasets import CocoDetection
# from torchvision.transforms import functional as F

# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, root, annotation, transforms=None):
#         self.root = root
#         self.transforms = transforms
#         self.coco = CocoDetection(root, annotation)

#     def __getitem__(self, idx):
#         img, target = self.coco[idx]
#         img = F.to_tensor(img)
#         target = {k: v for k, v in target.items() if k in ['boxes', 'labels']}
#         if self.transforms:
#             img, target = self.transforms(img, target)
#         return img, target

#     def __len__(self):
#         return len(self.coco)
class CustomTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        img = self.transforms(img)
        return img, target
    
import os
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import functional as F


class CustomCSVDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, class_mapping, transforms=None):
        self.root = root
        self.transforms = transforms
        self.class_mapping = class_mapping
        self.df = pd.read_csv(os.path.join(root, annotation_file))
        self.df['filename'] = self.df['filename'].astype(str)
        
        # Map class names to integers
        self.df['class'] = self.df['class'].map(self.class_mapping)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.df.iloc[idx]['filename'])
        img = Image.open(img_path).convert("RGB")
        
        # Extract the bounding boxes and labels for the current image
        boxes = self.df[self.df['filename'] == self.df.iloc[idx]['filename']]
        box_coords = boxes[['xmin', 'ymin', 'xmax', 'ymax']].values
        labels = boxes['class'].values
        
        # Debug: Check the class labels
        #print(f"Original labels: {labels}")

        # Check for any invalid labels
        if any(labels < 0):
            raise ValueError(f"Found invalid class labels: {labels}")
        
        box_coords = torch.tensor(box_coords, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        if self.transforms:
            img, target = self.transforms(img, {'boxes': box_coords, 'labels': labels})
        else:
            img = F.to_tensor(img)
            target = {'boxes': box_coords, 'labels': labels}
        
        return img, target

    def __len__(self):
        return self.df['filename'].nunique()