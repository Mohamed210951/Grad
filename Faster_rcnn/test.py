import torch
import torchvision
import cv2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Function to load the model with a customized number of classes
def get_model(num_classes):
    # Load a pre-trained Faster R-CNN model with a ResNet-50 backbone
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Replace the classifier with a new one that matches the number of classes (including background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)  # +1 for the background

    return model

# Function to perform detection on an image
def detect_objects(model, image_path, device='cuda'):
    # Load an image and convert it to RGB
    image = cv2.imread(image_path)
    image=cv2.resize(image,(640,640))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to a PyTorch tensor and add a batch dimension
    image_tensor = torchvision.transforms.functional.to_tensor(image).unsqueeze(0).to(device)

    # Set the model to evaluation mode and move it to the specified device
    model.eval()
    model.to(device)

    with torch.no_grad():
        predictions = model(image_tensor)

    return image, predictions

# Function to draw bounding boxes around detected objects
def draw_boxes(image, predictions, classes):
    # Loop through predictions, and draw bounding boxes and labels for high-confidence detections
    for box, score, label in zip(predictions[0]['boxes'], predictions[0]['scores'], predictions[0]['labels']):
        if score > 0.5:  # Filter out detections with low confidence
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a green rectangle
            cv2.putText(image, f'{classes[label]}: {score:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Convert RGB back to BGR for displaying with OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Detected Objects', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage of the functions
if __name__ == "__main__":
    num_classes = 5  # Number of classes, excluding the background
    model = get_model(num_classes)
    model.load_state_dict(torch.load(r"C:\Graduation_Project\Train\Faster_rcnn\fastercnn_state_dict.pth"))

    image_path = r"F:\final_Grad\output\image_412_jpg.rf.017bcd81bfe2f10612b4a89f926a5a20.jpg"  # Specify the path to your image
    classes = ['Background', 'Car', 'Truck', 'bus', 'Bike','toktok']  # List of class labels including background
    image, predictions = detect_objects(model, image_path)
    draw_boxes(image, predictions, classes)