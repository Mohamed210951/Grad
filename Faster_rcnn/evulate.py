import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from dataset import CustomCSVDataset, CustomTransform
from model import get_model
from metrics import evaluate

def evaluate_model(dataset_path, annotation_path, class_mapping):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize dataset and dataloader for validation
    transformations = CustomTransform(transforms.Compose([
        transforms.ToTensor()
    ]))
    
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, list(targets)
    
    validation_dataset = CustomCSVDataset(root=dataset_path, annotation_file=annotation_path, transforms=transformations, class_mapping=class_mapping)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Load the trained model
    model = get_model(num_classes=len(class_mapping) + 1)  # Adjust num_classes as needed
    model.load_state_dict(torch.load(r"C:\Graduation_Project\Train\Faster_rcnn\fastercnn_state_dict.pth"))
    model.to(device)
    model.eval()

    evaluate(model, validation_loader, device)
    # print("Accuracy:", results['accuracy'])
    # print("Precision:", results['precision'])
    # print("Recall:", results['recall'])
    # print("F1 Score:", results['f1'])
    #print(f"Evaluation Results - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, mAP: {mAP:.4f}")

if __name__ == "__main__":
    iamge_path_valid = r"F:\final_Grad\output"
    iamge_path_valid2 = r"F:\final_Grad\output\_annotations.csv"
    
    # Define the class mapping
    class_mapping = {
        'Car': 1,
        'Truck': 2,
        'Bus': 3,
        'MotorCycle': 4,
        'Toktok': 5,
    }

    results=evaluate_model(iamge_path_valid, iamge_path_valid2, class_mapping)
    print("Accuracy:", results['accuracy'])
    print("Precision:", results['precision'])
    print("Recall:", results['recall'])
    print("F1 Score:", results['f1_score'])
    # print(precision)
    # print(recall)
    # print(f1_score)
    # print(mAP)
