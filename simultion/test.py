from collections import defaultdict
import cv2
import os
import numpy as np
from ultralytics import YOLO
import time
import math
import json
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pandas as pd
import random
import threading
import pygame
import sys
from datetime import datetime, timedelta

# Default values of signal times
defaultRed = 150
defaultYellow = 5
defaultGreen = 20
defaultMinimum = 10
defaultMaximum = 60

count_for_each_intersextion = []
intersection_time = []

signals = []
noOfSignals = 4
simTime = 300  # change this to change time of simulation
timeElapsed = 0

currentGreen = 0   # Indicates which signal is green
nextGreen = (currentGreen + 1) % noOfSignals
currentYellow = 0   # Indicates whether yellow signal is on or off 

# Average times for vehicles to pass the intersection
carTime = 2
bikeTime = 1
rickshawTime = 2.25 
busTime = 2.5
truckTime = 2.5

# Count of cars at a traffic signal
noOfCars = 0
noOfBikes = 0
noOfBuses = 0
noOfTrucks = 0
noOfRickshaws = 0
noOfLanes = 2

# Red signal time at which cars will be detected at a signal
detectionTime = 5

speeds = {'car': 2.25, 'bus': 1.8, 'truck': 1.8, 'rickshaw': 2, 'bike': 2.5}  # average speeds of vehicles

# Coordinates of start
x = {'right': [0, 0, 0], 'down': [755, 727, 697], 'left': [1400, 1400, 1400], 'up': [602, 627, 657]}    
y = {'right': [348, 370, 398], 'down': [0, 0, 0], 'left': [498, 466, 436], 'up': [800, 800, 800]}

vehicles = {'right': {0: [], 1: [], 2: [], 'crossed': 0}, 'down': {0: [], 1: [], 2: [], 'crossed': 0}, 'left': {0: [], 1: [], 2: [], 'crossed': 0}, 'up': {0: [], 1: [], 2: [], 'crossed': 0}}
vehicleTypes = {0: 'rickshaw', 1: 'truck', 2: 'bus', 3: 'car', 4: 'bike'}
directionNumbers = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

# Coordinates of signal image, timer, and vehicle count
signalCoods = [(530, 230), (810, 230), (810, 570), (530, 570)]
signalTimerCoods = [(530, 210), (810, 210), (810, 550), (530, 550)]
vehicleCountCoods = [(480, 210), (880, 210), (880, 550), (480, 550)]
vehicleCountTexts = ["0", "0", "0", "0"]

# Coordinates of stop lines
stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
defaultStop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}
stops = {'right': [580, 580, 580], 'down': [320, 320, 320], 'left': [810, 810, 810], 'up': [545, 545, 545]}

mid = {'right': {'x': 705, 'y': 445}, 'down': {'x': 695, 'y': 450}, 'left': {'x': 695, 'y': 425}, 'up': {'x': 695, 'y': 400}}
rotationAngle = 3

array_of_round_robbin = []
araay_pri = [0, 1, 2, 3]

gap = 15    # stopping gap
gap2 = 15   # moving gap

model = YOLO(r"D:\Folder-Runs\finatl-fm3-data-set-yolov8l\weights\best.pt")

pygame.init()
simulation = pygame.sprite.Group()

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
        self.index = len(vehicles[direction][lane]) - 1
        path = "simultion/images/" + direction + "/" + vehicleClass + ".png"
        self.originalImage = pygame.image.load(path)
        self.currentImage = pygame.image.load(path)

        if direction == 'right':
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index - 1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index - 1].stop - vehicles[direction][lane][self.index - 1].currentImage.get_rect().width - gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap    
            x[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif direction == 'left':
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index - 1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index - 1].stop + vehicles[direction][lane][self.index - 1].currentImage.get_rect().width + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap
            x[direction][lane] += temp
            stops[direction][lane] += temp
        elif direction == 'down':
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index - 1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index - 1].stop - vehicles[direction][lane][self.index - 1].currentImage.get_rect().height - gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif direction == 'up':
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index - 1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index - 1].stop + vehicles[direction][lane][self.index - 1].currentImage.get_rect().height + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] += temp
            stops[direction][lane] += temp
        simulation.add(self)

    def render(self, screen):
        screen.blit(self.currentImage, (self.x, self.y))

    def move(self):
        if self.direction == 'right':
            if self.crossed == 0 and self.x + self.currentImage.get_rect().width > stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if self.willTurn == 1:
                if self.crossed == 0 or self.x + self.currentImage.get_rect().width < mid[self.direction]['x']:
                    if (self.x + self.currentImage.get_rect().width <= self.stop or (currentGreen == 0 and currentYellow == 0) or self.crossed == 1) and (self.index == 0 or self.x + self.currentImage.get_rect().width < (vehicles[self.direction][self.lane][self.index - 1].x - gap2) or vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                        self.x += self.speed
                else:   
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 2
                        self.y += 1.8
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if self.index == 0 or self.y + self.currentImage.get_rect().height < (vehicles[self.direction][self.lane][self.index - 1].y - gap2) or self.x + self.currentImage.get_rect().width < (vehicles[self.direction][self.lane][self.index - 1].x - gap2):
                            self.y += self.speed
            else: 
                if (self.x + self.currentImage.get_rect().width <= self.stop or self.crossed == 1 or (currentGreen == 0 and currentYellow == 0)) and (self.index == 0 or self.x + self.currentImage.get_rect().width < (vehicles[self.direction][self.lane][self.index - 1].x - gap2) or vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                    self.x += self.speed

        elif self.direction == 'down':
            if self.crossed == 0 and self.y + self.currentImage.get_rect().height > stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if self.willTurn == 1:
                if self.crossed == 0 or self.y + self.currentImage.get_rect().height < mid[self.direction]['y']:
                    if (self.y + self.currentImage.get_rect().height <= self.stop or (currentGreen == 1 and currentYellow == 0) or self.crossed == 1) and (self.index == 0 or self.y + self.currentImage.get_rect().height < (vehicles[self.direction][self.lane][self.index - 1].y - gap2) or vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                        self.y += self.speed
                else:   
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 2.5
                        self.y += 2
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index - 1].x + vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().width + gap2) or self.y < (vehicles[self.direction][self.lane][self.index - 1].y - gap2):
                            self.x -= self.speed
            else: 
                if (self.y + self.currentImage.get_rect().height <= self.stop or self.crossed == 1 or (currentGreen == 1 and currentYellow == 0)) and (self.index == 0 or self.y + self.currentImage.get_rect().height < (vehicles[self.direction][self.lane][self.index - 1].y - gap2) or vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                    self.y += self.speed
            
        elif self.direction == 'left':
            if self.crossed == 0 and self.x < stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if self.willTurn == 1:
                if self.crossed == 0 or self.x > mid[self.direction]['x']:
                    if (self.x >= self.stop or (currentGreen == 2 and currentYellow == 0) or self.crossed == 1) and (self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index - 1].x + vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().width + gap2) or vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                        self.x -= self.speed
                else: 
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 1.8
                        self.y -= 2.5
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index - 1].y + vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().height + gap2) or self.x > (vehicles[self.direction][self.lane][self.index - 1].x + gap2):
                            self.y -= self.speed
            else: 
                if (self.x >= self.stop or self.crossed == 1 or (currentGreen == 2 and currentYellow == 0)) and (self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index - 1].x + vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().width + gap2) or vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                    self.x -= self.speed

        elif self.direction == 'up':
            if self.crossed == 0 and self.y < stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if self.willTurn == 1:
                if self.crossed == 0 or self.y > mid[self.direction]['y']:
                    if (self.y >= self.stop or (currentGreen == 3 and currentYellow == 0) or self.crossed == 1) and (self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index - 1].y + vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().height + gap2) or vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                        self.y -= self.speed
                else:   
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 1
                        self.y -= 1
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if self.index == 0 or self.x < (vehicles[self.direction][self.lane][self.index - 1].x - vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().width - gap2) or self.y > (vehicles[self.direction][self.lane][self.index - 1].y + gap2):
                            self.x += self.speed
            else: 
                if (self.y >= self.stop or self.crossed == 1 or (currentGreen == 3 and currentYellow == 0)) and (self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index - 1].y + vehicles[self.direction][self.lane][self.index - 1].currentImage.get_rect().height + gap2) or vehicles[self.direction][self.lane][self.index - 1].turned == 1):
                    self.y -= self.speed

# Initialization of signals with default values
def initialize():
    ts1 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts1)
    ts2 = TrafficSignal(ts1.red + ts1.yellow + ts1.green, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts4)

def setTime(round_robbin_now):
    global noOfCars, noOfBikes, noOfBuses, noOfTrucks, noOfRickshaws, noOfLanes
    global carTime, busTime, truckTime, rickshawTime, bikeTime

    noOfCars, noOfBuses, noOfTrucks, noOfRickshaws, noOfBikes = 0, 0, 0, 0, 0

    for j in range(len(vehicles[directionNumbers[round_robbin_now]][0])):
        vehicle = vehicles[directionNumbers[round_robbin_now]][0][j]
        if vehicle.crossed == 0:
            noOfBikes += 1

    for i in range(1, 3):
        for j in range(len(vehicles[directionNumbers[round_robbin_now]][i])):
            vehicle = vehicles[directionNumbers[round_robbin_now]][i][j]
            if vehicle.crossed == 0:
                vclass = vehicle.vehicleClass
                if vclass == 'car':
                    noOfCars += 1
                elif vclass == 'bus':
                    noOfBuses += 1
                elif vclass == 'truck':
                    noOfTrucks += 1
                elif vclass == 'rickshaw':
                    noOfRickshaws += 1

    greenTime = math.ceil(((noOfCars * carTime) + (noOfRickshaws * rickshawTime) + (noOfBuses * busTime) + (noOfTrucks * truckTime) + (noOfBikes * bikeTime)) / (noOfLanes + 1))
    if greenTime < defaultMinimum:
        greenTime = defaultMinimum
    elif greenTime > defaultMaximum:
        greenTime = defaultMaximum
    return greenTime

def calclue_for_round_robbin():
    global array_of_round_robbin, araay_pri
    array_of_round_robbin = []
    if len(araay_pri) == 0:
        araay_pri = [0, 1, 2, 3]
    for i in range(len(araay_pri)):
        time2 = setTime(araay_pri[i])
        array_of_round_robbin.append([araay_pri[i], time2])

def repeat():
    global currentGreen, currentYellow, nextGreen, array_of_round_robbin, araay_pri
    currentGreen = -1
    calclue_for_round_robbin()
    maximun = -99999999
    for i in range(len(array_of_round_robbin)):
        if maximun < array_of_round_robbin[i][1]:
            maximun = array_of_round_robbin[i][1]
            currentGreen = array_of_round_robbin[i][0]
    araay_pri.remove(currentGreen)
    signals[currentGreen].green = maximun
    while signals[currentGreen].green > 0:
        printStatus()
        updateValues()
        if signals[(currentGreen + 1) % noOfSignals].red == detectionTime:
            thread = threading.Thread(name="detection", target=setTime, args=())
            thread.daemon = True
            thread.start()
        time.sleep(1)
    currentYellow = 1
    vehicleCountTexts[currentGreen] = "0"
    for i in range(0, 3):
        stops[directionNumbers[currentGreen]][i] = defaultStop[directionNumbers[currentGreen]]
        for vehicle in vehicles[directionNumbers[currentGreen]][i]:
            vehicle.stop = defaultStop[directionNumbers[currentGreen]]
    while signals[currentGreen].yellow > 0:
        printStatus()
        updateValues()
        time.sleep(1)
    currentYellow = 0
    signals[currentGreen].green = defaultGreen
    signals[currentGreen].yellow = defaultYellow
    signals[currentGreen].red = defaultRed
    currentGreen = nextGreen
    nextGreen = (currentGreen + 1) % noOfSignals
    signals[nextGreen].red = signals[currentGreen].yellow + signals[currentGreen].green
    repeat()

def printStatus():                                                                                           
    for i in range(noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                print(" GREEN TS", i + 1, "-> r:", signals[i].red, " y:", signals[i].yellow, " g:", signals[i].green)
            else:
                print("YELLOW TS", i + 1, "-> r:", signals[i].red, " y:", signals[i].yellow, " g:", signals[i].green)
        else:
            print("   RED TS", i + 1, "-> r:", signals[i].red, " y:", signals[i].yellow, " g:", signals[i].green)
    print()

def updateValues():
    for i in range(noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                signals[i].green -= 1
                signals[i].totalGreenTime += 1
            else:
                signals[i].yellow -= 1
        else:
            signals[i].red -= 1

global Which_Signal
Which_Signal = 0

def generate_random_vehicles():
    vehicle_counts = {
        "rickshaw": random.randint(0, 10),
        "truck": random.randint(0, 10),
        "bus": random.randint(0, 10),
        "car": random.randint(0, 10),
        "motorcycle": random.randint(0, 10)
    }

    all_intersection_data = []

    for intersection_id in range(1, 5):
        intersection_data = {
            "intersection_id": intersection_id,
            "vehicles": [
                {"type": "0", "count": vehicle_counts["rickshaw"]},
                {"type": "1", "count": vehicle_counts["truck"]},
                {"type": "2", "count": vehicle_counts["bus"]},
                {"type": "3", "count": vehicle_counts["car"]},
                {"type": "4", "count": vehicle_counts["motorcycle"]}
            ]
        }
        all_intersection_data.append(intersection_data)
    
    return all_intersection_data

def generateVehicles():
    global Which_Signal
    global timeElapsed
    global to_laod
    to_laod = 1
    while True:  # Infinite loop
        while to_laod != 1:
            time.sleep(15)
        if to_laod == 1:  # Check if to_load is 1
            timeElapsed = 0
            to_laod = 0
            intersection_data_loaded = generate_random_vehicles()
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
                        direction_number = intersection_id - 1
                        Vehicle(lane_number, vehicleTypes[vehicle_type], direction_number,
                                directionNumbers[direction_number], will_turn)
            to_laod = 0  # Set to_load to 0 to stop the loop
            repeat()
            break

def simulationTime():
    global timeElapsed, simTime
    while True:
        timeElapsed += 1
        time.sleep(1)
        if timeElapsed == simTime:
            totalVehicles = 0
            print('Lane-wise Vehicle Counts')
            for i in range(noOfSignals):
                print('Lane', i + 1, ':', vehicles[directionNumbers[i]]['crossed'])
                totalVehicles += vehicles[directionNumbers[i]]['crossed']
            print('Total vehicles passed: ', totalVehicles)
            print('Total time passed: ', timeElapsed)
            print('No. of vehicles passed per unit time: ', (float(totalVehicles) / float(timeElapsed)))
            os._exit(1)

def save_data_to_csv(data, file_path='data.csv'):
    df = pd.DataFrame(data)
    if not os.path.isfile(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)

def main_loop(screen, redSignal, yellowSignal, greenSignal, background, font):
    global timeElapsed, currentGreen, currentYellow
    data_collection_interval = timedelta(minutes=5)
    last_collection_time = datetime.now()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        screen.blit(background, (0, 0))

        current_time = datetime.now()
        if current_time - last_collection_time >= data_collection_interval:
            data = []
            for i in range(noOfSignals):
                direction = directionNumbers[i]
                for lane in range(3):
                    for vehicle in vehicles[direction][lane]:
                        if vehicle.crossed == 1:
                            data.append({
                                "time": current_time.strftime('%Y-%m-%d %H:%M:%S'),
                                "intersection": i,
                                "vehicle_type": vehicle.vehicleClass
                            })
            save_data_to_csv(data)
            last_collection_time = current_time

        for i in range(noOfSignals):
            if i == currentGreen:
                if currentYellow == 1:
                    screen.blit(yellowSignal, signalCoods[i])
                else:
                    screen.blit(greenSignal, signalCoods[i])
            else:
                screen.blit(redSignal, signalCoods[i])

        signalTexts = ["", "", "", ""]
        for i in range(noOfSignals):
            signalTexts[i] = font.render(str(signals[i].signalText), True, (255, 255, 255), (0, 0, 0))
            screen.blit(signalTexts[i], signalTimerCoods[i])
            displayText = vehicles[directionNumbers[i]]['crossed']
            vehicleCountTexts[i] = font.render(str(displayText), True, (0, 0, 0), (255, 255, 255))
            screen.blit(vehicleCountTexts[i], vehicleCountCoods[i])

        timeElapsedText = font.render(("Time Elapsed: " + str(timeElapsed)), True, (0, 0, 0), (255, 255, 255))
        screen.blit(timeElapsedText, (1100, 50))

        for vehicle in simulation:
            screen.blit(vehicle.currentImage, [vehicle.x, vehicle.y])
            vehicle.move()
        pygame.display.update()

class Main:
    global to_laod
    thread4 = threading.Thread(name="simulationTime", target=simulationTime, args=())
    thread4.daemon = True
    thread4.start()

    thread2 = threading.Thread(name="initialization", target=initialize, args=())
    thread2.daemon = True
    thread2.start()

    black = (0, 0, 0)
    white = (255, 255, 255)

    screenWidth = 1400
    screenHeight = 800
    screenSize = (screenWidth, screenHeight)

    background = pygame.image.load(r'simultion\images\mod_int.png')

    screen = pygame.display.set_mode(screenSize)
    pygame.display.set_caption("SIMULATION")

    redSignal = pygame.image.load(r'simultion\images\signals\red.png')
    yellowSignal = pygame.image.load(r'simultion\images\signals\yellow.png')
    greenSignal = pygame.image.load(r'simultion\images\signals\green.png')
    font = pygame.font.Font(None, 30)

    thread3 = threading.Thread(name="generateVehicles", target=generateVehicles, args=())
    thread3.daemon = True
    thread3.start()

    main_loop(screen, redSignal, yellowSignal, greenSignal, background, font)

Main()
