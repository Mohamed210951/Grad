from ultralytics import YOLO,RTDETR
import os
import cv2
from tkinter import filedialog
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#os.environ['TORCH_USE_CUDA_DSA']='True'


# if __name__ == '__main__':
    
def run_rt():
        model=RTDETR(r"moduels\rtdetr4\weights\best.pt")
        
        
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        
        cap = cv2.VideoCapture(file_path)




        while cap.isOpened():
            success, frame = cap.read()

            if success:
                frame_cropped= cv2.resize(frame,(640,640))
                #frame_cropped=cv2.cvtColor(frame_cropped,cv2.COLOR_BGR2RGB)
                results = model.predict(frame,conf=0.5,device="cuda:0")
                boxes = results[0].boxes.xywh.cpu()
                if results[0].boxes:
                    classes=results[0].boxes.cls
                    annotated_frame = results[0].plot()          
                    annotated_frame= cv2.resize(annotated_frame,(640,640))
                    cv2.imshow("RTDETR Tracking", annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
             break
        
        cap.release()
        cv2.destroyAllWindows()


