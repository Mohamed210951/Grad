import torch
import torchvision
import cv2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tkinter import filedialog


# Function to load the model with a customized number of classes
def get_model(num_classes):
    # Load a pre-trained Faster R-CNN model with a ResNet-50 backbone
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)  # +1 for the background
    return model

# Function to process each frame for object detection
def process_frame(model, frame, device='cuda'):
    model.eval()
    model.to(device)
    # Convert frame to tensor and add batch dimension
    frame_tensor = torchvision.transforms.functional.to_tensor(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(frame_tensor)
    return predictions

# Function to draw bounding boxes and labels
def draw_boxes(frame, predictions, classes):
    for box, score, label in zip(predictions[0]['boxes'], predictions[0]['scores'], predictions[0]['labels']):
        if score > 0.5:  # Filter low confidence
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{classes[label]}: {score:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame

# Main function to process a video
def process_video(video_path,model, classes):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Optional: Define the codec and create VideoWriter object to save the output video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    #out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame through the model
        frame=cv2.resize(frame,(640,640))
        predictions = process_frame(model, frame, device)
        # Draw bounding boxes on the frame
        output_frame = draw_boxes(frame, predictions, classes)

        # Display the resulting frame
        cv2.imshow('Video', output_frame)
        # Write the frame into the file 'output.avi'
        #out.write(output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything when job is finished
    cap.release()
    #out.release()
    cv2.destroyAllWindows()

def run_faster():
    num_classes = 5  
    model = get_model(num_classes)
    model.load_state_dict(torch.load(r"moduels\Faster_rcnn\fastercnn_state_dict.pth"))
    
    classes = ['Background', 'Car', 'Truck', 'bus', 'Bike','toktok']  
    
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    process_video(video_path,model, classes)
