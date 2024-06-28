import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
import time

from dataset import CustomCSVDataset, CustomTransform
from model import get_model
from metrics import evaluate
from torchvision.transforms import transforms
def debug_and_validate(dataloader, num_batches=20):
    for i, (images, targets) in enumerate(dataloader):
        if i >= num_batches:
            break
        print(f"Batch {i + 1}")
        print(f"Images shape: {images.shape}")
        for j, target in enumerate(targets):
            print(f"  Target {j + 1}")
            print(f"    Boxes: {target['boxes'].shape}")
            print(f"    Labels: {target['labels'].shape}")
            print(f"    Boxes: {target['boxes']}")
            print(f"    Labels: {target['labels']}")
        print("=" * 50)
def train(num_epochs, dataset_path, annotation_path, validation_path, validation_annotation):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define the class mapping
    class_mapping = {
        'Car': 1,
        'Truck': 2,
        'Bus': 3,
        'MotorCycle': 4,
        'Toktok': 5,
    }

    # Initialize dataset and dataloader for training and validation
    transformations = CustomTransform(transforms.Compose([
        transforms.ToTensor()
    ]))
    
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, list(targets)
    
    # Initialize dataset and dataloader for training and validation
    train_dataset = CustomCSVDataset(root=dataset_path, annotation_file=annotation_path, transforms=transformations, class_mapping=class_mapping)
    train_loader = DataLoader(train_dataset, batch_size=22, shuffle=True, collate_fn=collate_fn)
    validation_dataset = CustomCSVDataset(root=validation_path, annotation_file=validation_annotation, transforms=transformations, class_mapping=class_mapping)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    # print("Validating training data loader")
    # debug_and_validate(train_loader)
    # print("Validating validation data loader")
    # debug_and_validate(validation_loader)
    # Initialize the model
    model = get_model(num_classes=len(class_mapping) + 1)  # Adjust num_classes as needed
    model.to(device)
    optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
            #precision, recall, f1, mAP = evaluate(model, validation_loader, device)
            
            # Print progress
            elapsed_time = time.time() - start_time
            batches_left = len(train_loader) - (i + 1)
            time_per_batch = elapsed_time / (i + 1)
            remaining_time = batches_left * time_per_batch
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {losses.item():.4f}, Remaining time: {remaining_time:.2f}s")

        # Evaluate on the validation set after each epoch
        # precision, recall, f1, mAP = evaluate(model, validation_loader, device)
        # print(f"Epoch: {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, mAP: {mAP:.4f}")
    torch.save(model.state_dict(),r"C:\Graduation_Project\Train\Faster_rcnn\fastercnn_state_dict.pth")
    torch.save(model,r"C:\Graduation_Project\Train\Faster_rcnn\fastercnn.pth")
    

if __name__ == "__main__":
    iamge_path_train = r'C:\Users\Mohamed Ayman\Downloads\Final-dataset-grad-milestone3.v20i.tensorflow\train'
    iamge_path_train2 = r"C:\Users\Mohamed Ayman\Downloads\Final-dataset-grad-milestone3.v20i.tensorflow\train\_annotations.csv"
    iamge_path_valid = r"C:\Users\Mohamed Ayman\Downloads\Final-dataset-grad-milestone3.v20i.tensorflow\valid"
    iamge_path_valid2 = r"C:\Users\Mohamed Ayman\Downloads\Final-dataset-grad-milestone3.v20i.tensorflow\valid\_annotations.csv"
    
    train(100, iamge_path_train, iamge_path_train2, iamge_path_valid, iamge_path_valid2)
    
