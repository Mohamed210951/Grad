import os
import cv2
import re
import cv2
from tkinter import filedialog
import numpy as np
from ultralytics import YOLO,RTDETR
from View import *
import os

import tkinter.filedialog






global AllVechileslocation, OneMailDraft, AllVechilesClasss, AllVechilesConf
global instance_for_car
global instance_for_Truck
global instance_for_bus
global instance_for_motorcycle
global instance_for_riskwash
global MyForm
global ct
instance_for_car=0
instance_for_Truck=0
instance_for_bus=0
instance_for_motorcycle=0
instance_for_riskwash=0

model_RTDETR=YOLO(r"moduels\yolov8l-final4\weights\best.pt")
model_YOLO=YOLO(r"moduels\yolov9m-seg-final5\weights\best.pt")




def calculate_iou(box1, box2):
    # Unpack the coordinates and convert them to integers
    x1_min, y1_min, x1_max, y1_max = map(int, map(float, box1[:4]))
    x2_min, y2_min, x2_max, y2_max = map(int, map(float, box2[:4]))

    # Calculate the (x, y)-coordinates of the intersection rectangle
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    # Compute the area of intersection rectangle
    inter_area = max(0, x_inter_max - x_inter_min) * max(0, y_inter_max - y_inter_min)

    # Compute the area of both the prediction and ground-truth rectangles
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Compute the union area by using the formula: union_area = box1_area + box2_area - inter_area
    union_area = box1_area + box2_area - inter_area

    # Compute the IoU by dividing the intersection area by the union area
    iou = inter_area / union_area if union_area > 0 else 0

    return iou









 
def Extract_Vehicle_Box(Image,filename):
    global model
    global ct
    global AllVechileslocation, OneMailDraft, AllVechilesClasss,AllVechilesConf
    global instance_for_car
    global instance_for_Truck
    global instance_for_bus
    global instance_for_motorcycle
    global instance_for_riskwash

    # Read image
    #img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_shape = Image.shape
    iamge2=Image
    Image = cv2.resize(Image, (640, 640))
    # Perform prediction
    
    model_RTDETR_Result = model_RTDETR.predict(source=Image, conf=0.3,device=0)
    model_YOLO_result = model_YOLO.predict(source=Image, conf=0.3,device=0)
    model_RTDETR_Result2=model_RTDETR_Result
    
    AllVechileslocation = []
    AllVechilesClasss = []
    AllVechilesConf=[]
    OneMailDraft = []
    Allvechile=[]
    AllvechileClass=[]
    to_put=0
    for data in model_RTDETR_Result:
        if data.boxes:
            for x, c,c2 in zip(data.boxes.xyxy, data.boxes.cls,data.boxes.conf):
                        instance_for_riskwash+=1
                        x_str = str(x)
                        to_put =1
                        numerical_values = re.findall(r'[-+]?\d*\.\d+|\d+', x_str)
                        OneMailDraft.append([float(value) for value in numerical_values])
                        AllVechileslocation.append(OneMailDraft)
                        AllVechilesClasss.append(int(c))
                        AllVechilesConf.append(float(c2))
                        OneMailDraft=[]
    for data in model_YOLO_result:
        if data.boxes:
            for x, c,c2 in zip(data.boxes.xyxy, data.boxes.cls,data.boxes.conf):
                        instance_for_riskwash+=1
                        x_str = str(x)
                        to_put =1
                        numerical_values = re.findall(r'[-+]?\d*\.\d+|\d+', x_str)
                        OneMailDraft.append([float(value) for value in numerical_values])
                        AllVechileslocation.append(OneMailDraft)
                        AllVechilesClasss.append(int(c))
                        AllVechilesConf.append(float(c2))
                        OneMailDraft=[]
    index2=None
    index_to_remove=[]
    for i in range(len(AllVechilesClasss)):
        maxx=-9999999
        index2=None
        for k in range(len(AllVechilesClasss)):
            if k!=i :
                same=calculate_iou(AllVechileslocation[i][0],AllVechileslocation[k][0])
                if same > maxx:
                    maxx=same
                    index2=k
        if maxx==0.0:
            #print("iou = "+str(maxx))
            pass
        elif maxx>=0.7 and index2!=None:
            if AllVechilesConf[i]>AllVechilesConf[index2]:
              index_to_remove.append(index2)
            else:
                index_to_remove.append(i)
    if len(index_to_remove)!=0:
     index_to_remove = list(set(index_to_remove))           
    index_to_remove.sort(reverse=True)
    for index in index_to_remove:
        AllVechilesConf.pop(index)
        AllVechileslocation.pop(index)
        AllVechilesClasss.pop(index)  
          
    output_lines = []
    for i in range(len(AllVechileslocation)):
        if AllVechilesClasss[i]==0: 
            label_name="Bus"
        elif AllVechilesClasss[i]==1: 
            label_name="Car"
        elif AllVechilesClasss[i]==2:
            label_name="MotorCycle"
        elif AllVechilesClasss[i]==3:
            label_name="Toktok"
        elif AllVechilesClasss[i]==4:
            label_name="Truck"
       
        xmin, ymin, xmax, ymax = AllVechileslocation[i][0][:4]
        output_lines.append(f"{label_name} {AllVechilesConf[i]} {int(xmin)} {int(ymin)} {int(xmax)} {int(ymax)}")
    filename2="detection-results"
    with open(f"mAP\input\detection-results\\{filename}.txt", 'w') as file:
        for line in output_lines:
            file.write(line+'\n')
    
            

                
                        
                        
    
    
                        
                        
    
    return Image
                        
                        
        

# Example usage:
image_path = "path/to/your/image.jpg"
output_folder = r"C:\Graduation_Project\Ctreat_data\Seg_Dataset"
output_folder2 = r"C:\Graduation_Project\Ctreat_data\Obj_Dataset"




    
   
from collections import defaultdict
import cv2
import numpy as np
#
import time
import math
# Load the YOLOv8 model
ct=5000
def start_ins():
    global instance_for_car
    global instance_for_Truck
    global instance_for_bus
    global instance_for_motorcycle
    global instance_for_riskwash
    global ct
    # Ask user to select a folder containing video files

    #filetypes=[("Video files", "*.mp4;*.avi;*.mkv;*.mov;*.jpg;*.png")]
    folder_path = filedialog.askdirectory(title="Select Folder containing images")
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4') or f.endswith('.mov') or  f.endswith('.webm') or  f.endswith('.jpg')]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        cap = cv2.VideoCapture(image_path)

        # Check if the video capture is successful
        if not cap.isOpened():
            print("Error: Could not open video file ", image_path)
        else:
        
    # Get total number of frames and frame rate
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Calculate frame interval for 5 seconds
            frame_interval = int(fps * 1)

            # Initialize variables
            frame_count = 0

            while frame_count < total_frames:
                # Set the position in the video to read a frame every 5 seconds
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

                # Read the frame
                success, frame = cap.read()
                ct+=1
                if success:
                    # Process the frame using Extract_Vehicle_Box_intance function
                    #names: ['Rickshaw', 'Truck', 'bus', 'car', 'motorcycle']
                    filename=image_path[69:-4]
                    new_image = Extract_Vehicle_Box(frame,filename)
                    for i in range(len(AllVechilesClasss)):
                        if AllVechilesClasss[i]==0:
                            color = (0, 255, 0)  
                            name="Bus"
                        elif AllVechilesClasss[i]==1:
                            color = (255, 255, 0)  
                            name="Car"
                        elif AllVechilesClasss[i]==2:
                            color = (0, 255, 255)  
                            name="MotorCycle"
                        elif AllVechilesClasss[i]==3:
                            color = (0, 0, 255)  
                            name="Toktok"
                        elif AllVechilesClasss[i]==4:
                            color = (255, 0, 255)  
                            name="Truck"
                        

                        conf = AllVechilesConf[i]
                        cv2.rectangle(new_image, (int(AllVechileslocation[i][0][0]), int(AllVechileslocation[i][0][1])), (int(AllVechileslocation[i][0][2]), int(AllVechileslocation[i][0][3])), color, 2)
                        cv2.putText(new_image, f'{name}: {conf:.2f}', (int(AllVechileslocation[i][0][0]), int(AllVechileslocation[i][0][1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.imshow('Image with bounding boxes', new_image)  
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break                
                    
                    # Update instance variables as needed
                    # ...
                    
                    # print("Frame captured from video file:", file_path)
                    # print("Frame number:", frame_count)
                    
                    # Increment frame count by frame_interval
                    frame_count += 1

                else:
                    break

            # Release the video capture object
            cap.release()

            # print("car = "+str(instance_for_car))
            # print("Truck = "+str(instance_for_Truck))
            # print("Bus = "+str(instance_for_bus))
            # print("motorcycle = "+str(instance_for_motorcycle))
            # print("riskwash = "+str(instance_for_riskwash))






MyForm = Form()

MyForm.AddLable("Lable For Constraints", X=950, Y=100, Value="Menu")
MyForm.AddButton("Login", X=900, Y=500, Value="Select Folder",Width=30, Heigth=5,OnClick=start_ins)


MyForm.Display()

