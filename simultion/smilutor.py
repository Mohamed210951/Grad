# LAG
# NO. OF VEHICLES IN SIGNAL CLASS
# stops not used
# DISTRIBUTION
# BUS TOUCHING ON TURNS
# Distribution using python class



from collections import defaultdict
import cv2
import os
import numpy as np
from ultralytics import YOLO,RTDETR
import time
import math
import json
os.environ['KMP_DUPLICATE_LIB_OK']='True'#
import pandas as pd








# *** IMAGE XY COOD IS TOP LEFT
import random
import math
import time
import threading
# from vehicle_detection import detection
import pygame
import sys
import os
import json

# options={
#    'model':'./cfg/yolo.cfg',     #specifying the path of model
#    'load':'./bin/yolov2.weights',   #weights
#    'threshold':0.3     #minimum confidence factor to create a box, greater than 0.3 good
# }

# tfnet=TFNet(options)    #READ ABOUT TFNET

# Default values of signal times
defaultRed = 150
defaultYellow = 5
defaultGreen = 20
defaultMinimum = 10
defaultMaximum = 60
to_laod=None 
to_laod=1

count_for_each_intersextion=None
count_for_each_intersextion=[]
intersection_time=None
intersection_time=[]
# intersection_time.append(20)

signals = []
noOfSignals = 4
simTime = 300  # change this to change time of simulation
timeElapsed = 0

currentGreen = 0   # Indicates which signal is green
nextGreen = (currentGreen+1)%noOfSignals
currentYellow = 0   # Indicates whether yellow signal is on or off 

# Average times for vehicles to pass the intersection
carTime = 1.5
bikeTime = 1
rickshawTime = 1 
busTime = 2.5
truckTime = 2.5

# Count of cars at a traffic signal
noOfCars = 0
noOfBikes = 0
noOfBuses =0
noOfTrucks = 0
noOfRickshaws = 0
noOfLanes = 2

# Red signal time at which cars will be detected at a signal
detectionTime = 5

speeds = {'car':2.25, 'bus':1.8, 'truck':1.8, 'rickshaw':2, 'bike':2.5}  # average speeds of vehicles

# Coordinates of start
x = {'right':[0,0,0], 'down':[755,727,697], 'left':[1400,1400,1400], 'up':[602,627,657]}    
y = {'right':[348,370,398], 'down':[0,0,0], 'left':[498,466,436], 'up':[800,800,800]}

vehicles = {'right': {0:[], 1:[], 2:[], 'crossed':0}, 'down': {0:[], 1:[], 2:[], 'crossed':0}, 'left': {0:[], 1:[], 2:[], 'crossed':0}, 'up': {0:[], 1:[], 2:[], 'crossed':0}}
vehicleTypes = {0:'rickshaw', 1:'truck', 2:'bus', 3:'car', 4:'bike'}
directionNumbers = {0:'right', 1:'down', 2:'left', 3:'up'}

# Coordinates of signal image, timer, and vehicle count
signalCoods = [(530,230),(810,230),(810,570),(530,570)]
signalTimerCoods = [(530,210),(810,210),(810,550),(530,550)]
vehicleCountCoods = [(480,210),(880,210),(880,550),(480,550)]
vehicleCountTexts = ["0", "0", "0", "0"]

# Coordinates of stop lines
stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
defaultStop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}
stops = {'right': [580,580,580], 'down': [320,320,320], 'left': [810,810,810], 'up': [545,545,545]}

mid = {'right': {'x':705, 'y':445}, 'down': {'x':695, 'y':450}, 'left': {'x':695, 'y':425}, 'up': {'x':695, 'y':400}}
rotationAngle = 3



array_of_round_robbin=None
array_of_round_robbin=[]
araay_pri=None
araay_pri=[0,1,2,3]







# Gap between vehicles
gap = 15    # stopping gap
gap2 = 15   # moving gap


model=YOLO(r"moduels\yolov9m-final3\weights\best.pt")
#model=YOLO(r"C:\Graduation_Project\runs\segment\finatx-yolo8x_seg_Ammar\weights\best.pt")
#model=RTDETR(r"C:\Graduation_Project\runs\detect\finatl-rtdetrl_Ammar\weights\best.pt")


pygame.init()
simulation = pygame.sprite.Group()
def detect():
    path_traffic_light = r"F:\final_Grad\Data_Set\data_not_from_iphone\pexels-mike-bird-15610894.jpg"
    # path_near = r"C:\Graduation_Project\simultion\grad-intersection\img5.jpg"
    # path4=r"C:\Users\Mohamed Ayman\Downloads\pexels-bébé-ehiem-8608890.jpg"
    # path7=r"C:\Graduation_Project\simultion\grad-intersection\GECG43Na8AA4e3P.jpg"
    vehicle_counts = {
        "rickshaw": 0,
        "truck": 0,
        "bus": 0,
        "car": 0,
        "motorcycle": 0
    }

    video_paths = [path_traffic_light]
    ct=1
    all_intersectin_data=[]

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            success, frame = cap.read()
            
            if success:
                frame= cv2.resize(frame,(640,640))
                results = model.predict(frame,conf=0.5)
                boxes = results[0].boxes.xywh.cpu()
                if results[0].boxes:
                    vehicle_counts["rickshaw"] =0
                    vehicle_counts["truck"] =0
                    vehicle_counts["bus"] =0
                    vehicle_counts["car"] =0
                    vehicle_counts["motorcycle"] =0
                    #track_ids = results[0].boxes.id.int().cpu().tolist()
                    classes=results[0].boxes.cls
                    annotated_frame = results[0].plot()          
                    for box, classesss in zip(boxes,classes):
                        klass=int(classesss)
                        if klass == 0:
                            vehicle_counts["rickshaw"] += 1
                        elif klass == 1:
                            vehicle_counts["truck"] += 1
                        elif klass == 2:
                            vehicle_counts["bus"] += 1
                        elif klass == 3:
                            vehicle_counts["car"] += 1
                        elif klass == 4:
                            vehicle_counts["motorcycle"] += 1
                        #track = track_history[track_id]             
                        #annotated_frame= cv2.resize(annotated_frame,(640,640))
                        cv2.imshow("YOLOv8 Tracking", annotated_frame)
                    
                if cv2.waitKey(1) & 0xFF == ord("q"):
                  
                    break
                intersection_data = {
                "intersection_id": ct,
                "vehicles": [
                    {"type": "0", "count": vehicle_counts["rickshaw"]},
                    {"type": "1", "count": vehicle_counts["truck"]},
                    {"type": "2", "count": vehicle_counts["bus"]},
                    {"type": "3", "count": vehicle_counts["car"]},
                    {"type": "4", "count": vehicle_counts["motorcycle"]}
                 ]
                }
                all_intersectin_data.append(intersection_data)
                ct+=1
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
    json_filename = "trail.json"

    with open(json_filename, 'w') as json_file:
        json.dump(all_intersectin_data, json_file, indent=2)
class TrafficSignal:
    def __init__(self, red, yellow, green, minimum, maximum):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.minimum = minimum
        self.maximum = maximum
        self.signalText = "30"
        self.totalGreenTime = 0
        
class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction, will_turn):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        self.willTurn = will_turn
        self.turned = 0
        self.rotateAngle = 0
        vehicles[direction][lane].append(self)
        # self.stop = stops[direction][lane]
        self.index = len(vehicles[direction][lane]) - 1
        path = "simultion/images/" + direction + "/" + vehicleClass + ".png"
        self.originalImage = pygame.image.load(path)
        self.currentImage = pygame.image.load(path)

    
        if(direction=='right'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):    # if more than 1 vehicle in the lane of vehicle before it has crossed stop line
                self.stop = vehicles[direction][lane][self.index-1].stop - vehicles[direction][lane][self.index-1].currentImage.get_rect().width - gap         # setting stop coordinate as: stop coordinate of next vehicle - width of next vehicle - gap
            else:
                self.stop = defaultStop[direction]
            # Set new starting and stopping coordinate
            temp = self.currentImage.get_rect().width + gap    
            x[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif(direction=='left'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):
                self.stop = vehicles[direction][lane][self.index-1].stop + vehicles[direction][lane][self.index-1].currentImage.get_rect().width + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap
            x[direction][lane] += temp
            stops[direction][lane] += temp
        elif(direction=='down'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):
                self.stop = vehicles[direction][lane][self.index-1].stop - vehicles[direction][lane][self.index-1].currentImage.get_rect().height - gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif(direction=='up'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):
                self.stop = vehicles[direction][lane][self.index-1].stop + vehicles[direction][lane][self.index-1].currentImage.get_rect().height + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] += temp
            stops[direction][lane] += temp
        simulation.add(self)

    def render(self, screen):
        screen.blit(self.currentImage, (self.x, self.y))

    def move(self):
        if(self.direction=='right'):
            if(self.crossed==0 and self.x+self.currentImage.get_rect().width>stopLines[self.direction]):   # if the image has crossed stop line now
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if(self.willTurn==1):
                if(self.crossed==0 or self.x+self.currentImage.get_rect().width<mid[self.direction]['x']):
                    if((self.x+self.currentImage.get_rect().width<=self.stop or (currentGreen==0 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.x+self.currentImage.get_rect().width<(vehicles[self.direction][self.lane][self.index-1].x - gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):                
                        self.x += self.speed
                else:   
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 2
                        self.y += 1.8
                        if(self.rotateAngle==90):
                            self.turned = 1
                            # path = "images/" + directionNumbers[((self.direction_number+1)%noOfSignals)] + "/" + self.vehicleClass + ".png"
                            # self.x = mid[self.direction]['x']
                            # self.y = mid[self.direction]['y']
                            # self.image = pygame.image.load(path)
                    else:
                        if(self.index==0 or self.y+self.currentImage.get_rect().height<(vehicles[self.direction][self.lane][self.index-1].y - gap2) or self.x+self.currentImage.get_rect().width<(vehicles[self.direction][self.lane][self.index-1].x - gap2)):
                            self.y += self.speed
            else: 
                if((self.x+self.currentImage.get_rect().width<=self.stop or self.crossed == 1 or (currentGreen==0 and currentYellow==0)) and (self.index==0 or self.x+self.currentImage.get_rect().width<(vehicles[self.direction][self.lane][self.index-1].x - gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                # (if the image has not reached its stop coordinate or has crossed stop line or has green signal) and (it is either the first vehicle in that lane or it is has enough gap to the next vehicle in that lane)
                    self.x += self.speed  # move the vehicle



        elif(self.direction=='down'):
            if(self.crossed==0 and self.y+self.currentImage.get_rect().height>stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if(self.willTurn==1):
                if(self.crossed==0 or self.y+self.currentImage.get_rect().height<mid[self.direction]['y']):
                    if((self.y+self.currentImage.get_rect().height<=self.stop or (currentGreen==1 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.y+self.currentImage.get_rect().height<(vehicles[self.direction][self.lane][self.index-1].y - gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):                
                        self.y += self.speed
                else:   
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 2.5
                        self.y += 2
                        if(self.rotateAngle==90):
                            self.turned = 1
                    else:
                        if(self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or self.y<(vehicles[self.direction][self.lane][self.index-1].y - gap2)):
                            self.x -= self.speed
            else: 
                if((self.y+self.currentImage.get_rect().height<=self.stop or self.crossed == 1 or (currentGreen==1 and currentYellow==0)) and (self.index==0 or self.y+self.currentImage.get_rect().height<(vehicles[self.direction][self.lane][self.index-1].y - gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                    self.y += self.speed
            
        elif(self.direction=='left'):
            if(self.crossed==0 and self.x<stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if(self.willTurn==1):
                if(self.crossed==0 or self.x>mid[self.direction]['x']):
                    if((self.x>=self.stop or (currentGreen==2 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):                
                        self.x -= self.speed
                else: 
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 1.8
                        self.y -= 2.5
                        if(self.rotateAngle==90):
                            self.turned = 1
                            # path = "images/" + directionNumbers[((self.direction_number+1)%noOfSignals)] + "/" + self.vehicleClass + ".png"
                            # self.x = mid[self.direction]['x']
                            # self.y = mid[self.direction]['y']
                            # self.currentImage = pygame.image.load(path)
                    else:
                        if(self.index==0 or self.y>(vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height +  gap2) or self.x>(vehicles[self.direction][self.lane][self.index-1].x + gap2)):
                            self.y -= self.speed
            else: 
                if((self.x>=self.stop or self.crossed == 1 or (currentGreen==2 and currentYellow==0)) and (self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                # (if the image has not reached its stop coordinate or has crossed stop line or has green signal) and (it is either the first vehicle in that lane or it is has enough gap to the next vehicle in that lane)
                    self.x -= self.speed  # move the vehicle    
            # if((self.x>=self.stop or self.crossed == 1 or (currentGreen==2 and currentYellow==0)) and (self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2))):                
            #     self.x -= self.speed
        elif(self.direction=='up'):
            if(self.crossed==0 and self.y<stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if(self.willTurn==1):
                if(self.crossed==0 or self.y>mid[self.direction]['y']):
                    if((self.y>=self.stop or (currentGreen==3 and currentYellow==0) or self.crossed == 1) and (self.index==0 or self.y>(vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height +  gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):
                        self.y -= self.speed
                else:   
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 1
                        self.y -= 1
                        if(self.rotateAngle==90):
                            self.turned = 1
                    else:
                        if(self.index==0 or self.x<(vehicles[self.direction][self.lane][self.index-1].x - vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width - gap2) or self.y>(vehicles[self.direction][self.lane][self.index-1].y + gap2)):
                            self.x += self.speed
            else: 
                if((self.y>=self.stop or self.crossed == 1 or (currentGreen==3 and currentYellow==0)) and (self.index==0 or self.y>(vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height + gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                    self.y -= self.speed

# Initialization of signals with default values
def initialize():
    ts1 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts1)
    ts2 = TrafficSignal(ts1.red+ts1.yellow+ts1.green, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts4)
    #repeat()

# Set time according to formula
def setTime(round_robbin_now):
    global noOfCars, noOfBikes, noOfBuses, noOfTrucks, noOfRickshaws, noOfLanes
    global carTime, busTime, truckTime, rickshawTime, bikeTime
    os.system("say detecting vehicles, "+directionNumbers[(currentGreen+1)%noOfSignals])
#    detection_result=detection(currentGreen,tfnet)
#    greenTime = math.ceil(((noOfCars*carTime) + (noOfRickshaws*rickshawTime) + (noOfBuses*busTime) + (noOfBikes*bikeTime))/(noOfLanes+1))
#    if(greenTime<defaultMinimum):
#       greenTime = defaultMinimum
#    elif(greenTime>defaultMaximum):
#       greenTime = defaultMaximum
    # greenTime = len(vehicles[currentGreen][0])+len(vehicles[currentGreen][1])+len(vehicles[currentGreen][2])
    # noOfVehicles = len(vehicles[directionNumbers[nextGreen]][1])+len(vehicles[directionNumbers[nextGreen]][2])-vehicles[directionNumbers[nextGreen]]['crossed']
    # print("no. of vehicles = ",noOfVehicles)
    noOfCars, noOfBuses, noOfTrucks, noOfRickshaws, noOfBikes = 0,0,0,0,0
    
    
    for j in range(len(vehicles[directionNumbers[round_robbin_now]][0])):
        vehicle = vehicles[directionNumbers[round_robbin_now]][0][j]
        if(vehicle.crossed==0):
            vclass = vehicle.vehicleClass
            # print(vclass)
            noOfBikes += 1
    for i in range(1,3):
        for j in range(len(vehicles[directionNumbers[round_robbin_now]][i])):
            vehicle = vehicles[directionNumbers[round_robbin_now]][i][j]
            if(vehicle.crossed==0):
                vclass = vehicle.vehicleClass
                # print(vclass)
                if(vclass=='car'):
                    noOfCars += 1
                elif(vclass=='bus'):
                    noOfBuses += 1
                elif(vclass=='truck'):
                    noOfTrucks += 1
                elif(vclass=='rickshaw'):
                    noOfRickshaws += 1
    # print(noOfCars)
    greenTime = math.ceil(((noOfCars*carTime) + (noOfRickshaws*rickshawTime) + (noOfBuses*busTime) + (noOfTrucks*truckTime)+ (noOfBikes*bikeTime))/(noOfLanes+1))
    # greenTime = math.ceil((noOfVehicles)/noOfLanes) 
    print('Green Time: ',greenTime)
    if(greenTime<defaultMinimum):
        greenTime = defaultMinimum
    elif(greenTime>defaultMaximum):
        greenTime = defaultMaximum
    # greenTime = random.randint(15,50)
    
    #signals[(currentGreen+1)%(noOfSignals)].green = greenTime
    #signals[(currentGreen+1)%(noOfSignals)].green = 20
    
    #intersection_time.append(greenTime)
    #if len(intersection_time) >=4:
    return greenTime
             
def calclue_for_round_robbin():
    global array_of_round_robbin,araay_pri
    #time.sleep(3)
    array_of_round_robbin=[]
    if len(araay_pri)==0:
        araay_pri=[0,1,2,3]
    for i in range(0,len(araay_pri)):
            time2=setTime(araay_pri[i])
            array_of_round_robbin.append([araay_pri[i],time2])  
           
    # 1 2 3 4 
    #     |
   
def repeat():
    global currentGreen, currentYellow, nextGreen,array_of_round_robbin,araay_pri
    currentGreen = -1
    calclue_for_round_robbin()
    maximun=-99999999
    for i in range(len(array_of_round_robbin)):
        for k in range(len(array_of_round_robbin[i])):
            if(maximun<array_of_round_robbin[i][1]):
                maximun=array_of_round_robbin[i][1]
                currentGreen=array_of_round_robbin[i][0]
    araay_pri.remove(currentGreen)
    signals[currentGreen].green=maximun
    while(signals[currentGreen].green>0):   # while the timer of current green signal is not zero
        printStatus()
        updateValues()
        if(signals[(currentGreen+1)%(noOfSignals)].red==detectionTime):    # set time of next green signal 
            thread = threading.Thread(name="detection",target=setTime, args=())
            thread.daemon = True
            thread.start()
            # setTime()
        time.sleep(1)
    currentYellow = 1   # set yellow signal on
    vehicleCountTexts[currentGreen] = "0"
    # reset stop coordinates of lanes and vehicles 
    for i in range(0,3):
        stops[directionNumbers[currentGreen]][i] = defaultStop[directionNumbers[currentGreen]]
        for vehicle in vehicles[directionNumbers[currentGreen]][i]:
            vehicle.stop = defaultStop[directionNumbers[currentGreen]]
    while(signals[currentGreen].yellow>0):  # while the timer of current yellow signal is not zero
        printStatus()
        updateValues()
        time.sleep(1)
    currentYellow = 0   # set yellow signal off
    
    # reset all signal times of current signal to default times
    signals[currentGreen].green = defaultGreen
    signals[currentGreen].yellow = defaultYellow
    signals[currentGreen].red = defaultRed
       
    currentGreen = nextGreen # set next signal as green signal
    nextGreen = (currentGreen+1)%noOfSignals    # set next green signal
    signals[nextGreen].red = signals[currentGreen].yellow+signals[currentGreen].green    # set the red time of next to next signal as (yellow time + green time) of next signal
    repeat()     

# Print the signal timers on cmd
def printStatus():                                                                                           
	for i in range(0, noOfSignals):
		if(i==currentGreen):
			if(currentYellow==0):
				print(" GREEN TS",i+1,"-> r:",signals[i].red," y:",signals[i].yellow," g:",signals[i].green)
			else:
				print("YELLOW TS",i+1,"-> r:",signals[i].red," y:",signals[i].yellow," g:",signals[i].green)
		else:
			print("   RED TS",i+1,"-> r:",signals[i].red," y:",signals[i].yellow," g:",signals[i].green)
	print()

# Update values of the signal timers after every second
def updateValues():
    for i in range(0, noOfSignals):
        if(i==currentGreen):
            if(currentYellow==0):
                signals[i].green-=1
                signals[i].totalGreenTime+=1
            else:
                signals[i].yellow-=1
        else:
            signals[i].red-=1

global Which_Signal
Which_Signal=0
# Generating vehicles in the simulation
def generateVehicles():
 global Which_Signal
 global timeElapsed
 global to_laod
 to_laod=1
 number_of=0
 while True:  # Infinite loop
    # while to_laod!=1:
    #     time.sleep(15) 
    # if to_laod==1:
    #     detect()
    if to_laod == 1:  # Check if to_load is 1
            timeElapsed=0
            to_laod=0
        #for i in range(4,9):  # Loop through scenario files
            json_filename = f"simultion//senario{number_of+1}.json"  # Construct the filename
            #json_filename = f"simultion\trail.json"  # Construct the filename
           
            with open(json_filename, 'r') as json_file:
                intersection_data_loaded = json.load(json_file)
                
            for intersection_info in intersection_data_loaded:
                intersection_id = intersection_info["intersection_id"]
                vehicles = intersection_info["vehicles"]
                count22 = 0
                for vehicle_info in vehicles:
                    vehicle_type = int(vehicle_info["type"])
                    count = vehicle_info["count"]
                    count22 += count
                    if vehicle_type == 4:
                        count_for_each_intersextion.append(count22)

                    for _ in range(count):
                        lane_number = random.randint(0, 1) + 1 if vehicle_type != 4 else 0

                        will_turn = 0
                        if lane_number == 2:
                            temp = random.randint(0, 4)
                            will_turn = 1 if temp <= 2 else 0

                        temp = random.randint(0, 999)
                        direction_number = 0
                        if intersection_id == 1:
                            direction_number = 0
                        elif intersection_id == 2:
                            direction_number = 1
                        elif intersection_id == 3:
                            direction_number = 2
                        elif intersection_id == 4:
                            direction_number = 3
                        Vehicle(lane_number, vehicleTypes[vehicle_type], direction_number,
                                directionNumbers[direction_number], will_turn)
            #to_laod = 0  # Set to_load to 0 to stop the loop
            number_of+=1
            repeat()
            break
            if number_of == 11:
                break
        
    
    

def simulationTime():
    global timeElapsed, simTime
    while(True):
        timeElapsed += 1
        time.sleep(1)
        if(timeElapsed==simTime):
            totalVehicles = 0
            print('Lane-wise Vehicle Counts')
            for i in range(noOfSignals):
                print('Lane',i+1,':',vehicles[directionNumbers[i]]['crossed'])
                totalVehicles += vehicles[directionNumbers[i]]['crossed']
            print('Total vehicles passed: ',totalVehicles)
            print('Total time passed: ',timeElapsed)
            print('No. of vehicles passed per unit time: ',(float(totalVehicles)/float(timeElapsed)))
            os._exit(1)
    

class Main:
    global to_laod
    thread4 = threading.Thread(name="simulationTime",target=simulationTime, args=()) 
    thread4.daemon = True
    thread4.start()

    thread2 = threading.Thread(name="initialization",target=initialize, args=())    # initialization
    thread2.daemon = True
    thread2.start()

    # Colours 
    black = (0, 0, 0)
    white = (255, 255, 255)

    # Screensize 
    screenWidth = 1400
    screenHeight = 800
    screenSize = (screenWidth, screenHeight)

    # Setting background image i.e. image of intersection
    background = pygame.image.load(r'simultion\images\mod_int.png')

    screen = pygame.display.set_mode(screenSize)
    pygame.display.set_caption("SIMULATION")

    # Loading signal images and font
    redSignal = pygame.image.load(r'simultion\images\signals\red.png')
    yellowSignal = pygame.image.load(r'simultion\images\signals\yellow.png')
    greenSignal = pygame.image.load(r'simultion\images\signals\green.png')
    font = pygame.font.Font(None, 30)
    
    thread3 = threading.Thread(name="generateVehicles",target=generateVehicles, args=())    # Generating vehicles
    thread3.daemon = True
    thread3.start()
    turn=0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        screen.blit(background,(0,0))   # display background in simulation
        
        
        number_out=0
        if len(count_for_each_intersextion)>3:
            if turn==0:
                intersection_time.append(signals[currentGreen].green)
                signals[currentGreen].green
                vehicles_in_lane = vehicles['right']
                if len(vehicles_in_lane)>0 :
                    for i in range(len(vehicles_in_lane)-1) :
                        g=len(vehicles_in_lane[i])
                        for k in range(g):
                            if vehicles_in_lane[i][k].crossed==1:
                                number_out+=1
                            
                    print(number_out)
                    if number_out==count_for_each_intersextion[0]:
                        signals[currentGreen].green
                        turn+=1
                        number_out=0
                        
                        
            elif turn==1:                
                vehicles_in_lane = vehicles['down']
                intersection_time.append(signals[currentGreen].green)
                if len(vehicles_in_lane)>0 :
                    for i in range(len(vehicles_in_lane)-1) :
                        g=len(vehicles_in_lane[i])
                        for k in range(g):
                            if vehicles_in_lane[i][k].crossed==1:
                                number_out+=1
                            
                    print(number_out)
                    if number_out==count_for_each_intersextion[1]:
                        signals[currentGreen].green
                        turn+=1
                        number_out=0
                    
                    
            elif turn==2:
                vehicles_in_lane = vehicles['left']
                intersection_time.append(signals[currentGreen].green)
                if len(vehicles_in_lane)>0 :
                    for i in range(len(vehicles_in_lane)-1) :
                        g=len(vehicles_in_lane[i])
                        for k in range(g):
                            if vehicles_in_lane[i][k].crossed==1:
                                number_out+=1
                            
                    print(number_out)
                    if number_out==count_for_each_intersextion[2]:
                        signals[currentGreen].green
                        turn+=1
                        number_out=0
                    
                    
            elif turn==3:        
                vehicles_in_lane = vehicles['up']
                intersection_time.append(signals[currentGreen].green)
                if len(vehicles_in_lane)>0 :
                    for i in range(len(vehicles_in_lane)-1) :
                        g=len(vehicles_in_lane[i])
                        for k in range(g):
                            if vehicles_in_lane[i][k].crossed==1:
                                number_out+=1
                            
                    print(number_out)
                    if number_out==count_for_each_intersextion[3]:
                        signals[currentGreen].green
                        turn+=1
                        number_out=0

                        new_data = {
                            'Distribution 1': [count_for_each_intersextion[0]],
                            'Distribution 2': [count_for_each_intersextion[1]],
                            'Distribution 3': [count_for_each_intersextion[2]],
                            'Distribution 4': [count_for_each_intersextion[3]],
                            'Total Time': [timeElapsed]
                        }

                          # Create a DataFrame from the new data
                        new_df = pd.DataFrame(new_data)

                        # Specify the file path
                        file_path = 'static.xlsx'

                        # Check if the file already exists
                        try:
                            # Read existing Excel file into DataFrame
                            existing_df = pd.read_excel(file_path)
                            
                            # Append new data to existing DataFrame
                            updated_df = existing_df._append(new_df, ignore_index=True)
                            
                            # Save the updated DataFrame to Excel
                            updated_df.to_excel(file_path, index=False)
                            
                            print(f"Data appended to {file_path}")
                            to_laod = 1
                            
                        except FileNotFoundError:
                            # If the file doesn't exist, write the new DataFrame directly
                            new_df.to_excel(file_path, index=False)
                            print(f"Data saved to {file_path}")

                                      
                  

  
        
        
        for i in range(0,noOfSignals):  # display signal and set timer according to current status: green, yello, or red
            if(i==currentGreen):
                if(currentYellow==1):
                    if(signals[i].yellow==0):
                        signals[i].signalText = "STOP"
                    else:
                        signals[i].signalText = signals[i].yellow
                    screen.blit(yellowSignal, signalCoods[i])
                else:
                    if(signals[i].green==0):
                        signals[i].signalText = "SLOW"
                    else:
                        signals[i].signalText = signals[i].green
                    screen.blit(greenSignal, signalCoods[i])
            else:
                if(signals[i].red<=10):
                    if(signals[i].red==0):
                        signals[i].signalText = "GO"
                    else:
                        signals[i].signalText = signals[i].red
                else:
                    signals[i].signalText = "---"
                screen.blit(redSignal, signalCoods[i])
        signalTexts = ["","","",""]

        # display signal timer and vehicle count
        for i in range(0,noOfSignals):  
            signalTexts[i] = font.render(str(signals[i].signalText), True, white, black)
            screen.blit(signalTexts[i],signalTimerCoods[i]) 
            displayText = vehicles[directionNumbers[i]]['crossed']
            vehicleCountTexts[i] = font.render(str(displayText), True, black, white)
            screen.blit(vehicleCountTexts[i],vehicleCountCoods[i])
        
        timeElapsedText = font.render(("Time Elapsed: "+str(timeElapsed)), True, black, white)
        screen.blit(timeElapsedText,(1100,50))

        # display the vehicles
        for vehicle in simulation:  
            screen.blit(vehicle.currentImage, [vehicle.x, vehicle.y])
            # vehicle.render(screen)
            vehicle.move()
        pygame.display.update()
        
        






Main()  