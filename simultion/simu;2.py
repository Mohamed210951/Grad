import pygame
import sys
import os
import time
import random
import json
import threading
import pandas as pd

pygame.init()

# Define the Vehicle class
class Vehicle:
    def __init__(self, lane, vehicleType, directionNumber, direction, willTurn):
        self.x = 0
        self.y = 0
        self.lane = lane
        self.direction = direction
        self.directionNumber = directionNumber
        self.vehicleType = vehicleType
        self.willTurn = willTurn
        self.crossed = 0
        if(lane==0):
            if(direction==0):
                self.x = 600
                self.y = 50
            elif(direction==1):
                self.x = 350
                self.y = 600
            elif(direction==2):
                self.x = 800
                self.y = 350
            elif(direction==3):
                self.x = 50
                self.y = 350
        elif(lane==1):
            if(direction==0):
                self.x = 600
                self.y = 100
            elif(direction==1):
                self.x = 300
                self.y = 600
            elif(direction==2):
                self.x = 800
                self.y = 400
            elif(direction==3):
                self.x = 100
                self.y = 400
        else:
            if(direction==0):
                self.x = 600
                self.y = 150
            elif(direction==1):
                self.x = 250
                self.y = 600
            elif(direction==2):
                self.x = 800
                self.y = 450
            elif(direction==3):
                self.x = 150
                self.y = 450

        # Load the vehicle image based on type
        if(self.vehicleType == 'car'):
            self.currentImage = pygame.image.load(r'simultion\images\vehicles\car.png')
        elif(self.vehicleType == 'bus'):
            self.currentImage = pygame.image.load(r'simultion\images\vehicles\bus.png')
        elif(self.vehicleType == 'bike'):
            self.currentImage = pygame.image.load(r'simultion\images\vehicles\bike.png')
        elif(self.vehicleType == 'truck'):
            self.currentImage = pygame.image.load(r'simultion\images\vehicles\truck.png')
        self.currentImage = pygame.transform.scale(self.currentImage, (50, 30))

    # Move the vehicle
    def move(self):
        if(self.direction==0):
            self.y -= 5
        elif(self.direction==1):
            self.x += 5
        elif(self.direction==2):
            self.y += 5
        elif(self.direction==3):
            self.x -= 5
        if(self.y<-50 or self.y>850 or self.x<-50 or self.x>1450):
            self.crossed = 1

# Define the Signal class
class Signal:
    def __init__(self):
        self.red = defaultRed
        self.yellow = defaultYellow
        self.green = defaultGreen
        self.signalText = self.red
        self.totalGreenTime = 0

# Define global variables
noOfSignals = 4
currentGreen = 0
currentYellow = 0
nextGreen = 1
timeElapsed = 0
simTime = 5000
signalCoods = [[500, 50], [1050, 500], [500, 950], [50, 500]]
signalTimerCoods = [[520, 10], [1070, 460], [520, 1010], [70, 460]]
vehicleCountCoods = [[550, 10], [1100, 460], [550, 1010], [100, 460]]
vehicleTypes = ['car', 'bus', 'bike', 'truck']
defaultRed = 20
defaultYellow = 4
defaultGreen = 20
simulation = []
signal = []
signals = [Signal() for i in range(noOfSignals)]
vehicles = [{},{},{},{}]
stops = [[900, 0], [0, 600], [900, 1200], [1800, 600]]
defaultStop = [[900, 0], [0, 600], [900, 1200], [1800, 600]]
directionNumbers = [0,1,2,3]
count_for_each_intersextion = []
intersection_time = []
to_laod=1


# Initialization
def initialize():
    for i in range(0, noOfSignals):
        simulation.append(Signal())
        signal.append(Signal())
        signal[i] = simulation[i]
    thread1 = threading.Thread(name="repeat",target=repeat, args=())
    thread1.daemon = True
    thread1.start()


def repeat():
    global currentGreen, currentYellow, nextGreen, currentIntersection
    while(signals[currentGreen].green>0):
        printStatus()
        updateValues()
        if(signals[(currentGreen+1)%(noOfSignals)].red==0):
            thread = threading.Thread(name="detection",target=setTime, args=())
            thread.daemon = True
            thread.start()
        time.sleep(1)
    currentYellow = 1
    vehicleCountTexts[currentGreen] = "0"
    for i in range(0,3):
        stops[directionNumbers[currentGreen]][i] = defaultStop[directionNumbers[currentGreen]]
        for vehicle in vehicles[directionNumbers[currentGreen]][i]:
            vehicle.stop = defaultStop[directionNumbers[currentGreen]]
    while(signals[currentGreen].yellow>0):
        printStatus()
        updateValues()
        time.sleep(1)
    currentYellow = 0
    signals[currentGreen].green = defaultGreen
    signals[currentGreen].yellow = defaultYellow
    signals[currentGreen].red = defaultRed
    currentGreen = nextGreen
    nextGreen = (currentGreen+1)%noOfSignals
    signals[nextGreen].red = signals[currentGreen].yellow+signals[currentGreen].green
    repeat()


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


def generateVehicles():
 global Which_Signal
 global timeElapsed
 global to_laod
 to_laod=1
 number_of=3
 while True:  # Infinite loop
    while to_laod!=1:
        time.sleep(15) 
    if to_laod == 1:  # Check if to_load is 1
            timeElapsed=0
            to_laod=0
        #for i in range(4,9):  # Loop through scenario files
            json_filename = f"simultion//senario{number_of+1}.json"  # Construct the filename
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
            to_laod = 0  # Set to_load to 0 to stop the loop
            number_of+=1
            if number_of == 11:
                break
    # global timeElapsed, to_laod
    # while True:
    #     while to_laod != 1:
    #         time.sleep(15)
    #     if to_laod == 1:
    #         timeElapsed = 0
    #         to_laod = 0
    #         json_filename = f"simultion//senario{currentIntersection}.json"
    #         with open(json_filename, 'r') as json_file:
    #             intersection_data_loaded = json.load(json_file)

    #         for intersection_info in intersection_data_loaded:
    #             intersection_id = intersection_info["intersection_id"]
    #             vehicles = intersection_info["vehicles"]
    #             count22 = 0
    #             for vehicle_info in vehicles:
    #                 vehicle_type = int(vehicle_info["type"])
    #                 count = vehicle_info["count"]
    #                 count22 += count
    #                 if vehicle_type == 4:
    #                     count_for_each_intersextion.append(count22)

    #                 for _ in range(count):
    #                     lane_number = random.randint(0, 1) + 1 if vehicle_type != 4 else 0

    #                     will_turn = 0
    #                     if lane_number == 2:
    #                         temp = random.randint(0, 4)
    #                         will_turn = 1 if temp <= 2 else 0

    #                     temp = random.randint(0, 999)
    #                     direction_number = 0
    #                     if intersection_id == 1:
    #                         direction_number = 0
    #                     elif intersection_id == 2:
    #                         direction_number = 1
    #                     elif intersection_id == 3:
    #                         direction_number = 2
    #                     elif intersection_id == 4:
    #                         direction_number = 3
    #                     Vehicle(lane_number, vehicleTypes[vehicle_type], direction_number,
    #                             directionNumbers[direction_number], will_turn)
    #         to_laod = 0
    #         currentIntersection = (currentIntersection % 4) + 1
    #         if currentIntersection == 1:
    #             break


def simulationTime():
    global timeElapsed, simTime
    while True:
        timeElapsed += 1
        time.sleep(1)
        if timeElapsed == simTime:
            totalVehicles = 0
            print('Lane-wise Vehicle Counts')
            for i in range(noOfSignals):
                print('Lane', i+1, ':', vehicles[directionNumbers[i]]['crossed'])
                totalVehicles += vehicles[directionNumbers[i]]['crossed']
            print('Total vehicles passed: ', totalVehicles)
            print('Total time passed: ', timeElapsed)
            print('No. of vehicles passed per unit time: ', (float(totalVehicles)/float(timeElapsed)))
            os._exit(1)


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
    turn = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        screen.blit(background, (0, 0))

        number_out = 0
        if len(count_for_each_intersextion) > 3:
            if turn == 0:
                intersection_time.append(signals[currentGreen].green)
                signals[currentGreen].green
                vehicles_in_lane = vehicles['right']
                if len(vehicles_in_lane) > 0:
                    for i in range(len(vehicles_in_lane) - 1):
                        g = len(vehicles_in_lane[i])
                        for k in range(g):
                            if vehicles_in_lane[i][k].crossed == 1:
                                number_out += 1

                    print(number_out)
                    if number_out == count_for_each_intersextion[0]:
                        signals[currentGreen].green
                        turn += 1
                        number_out = 0

            elif turn == 1:
                vehicles_in_lane = vehicles['down']
                intersection_time.append(signals[currentGreen].green)
                if len(vehicles_in_lane) > 0:
                    for i in range(len(vehicles_in_lane) - 1):
                        g = len(vehicles_in_lane[i])
                        for k in range(g):
                            if vehicles_in_lane[i][k].crossed == 1:
                                number_out += 1

                    print(number_out)
                    if number_out == count_for_each_intersextion[1]:
                        signals[currentGreen].green
                        turn += 1
                        number_out = 0

            elif turn == 2:
                vehicles_in_lane = vehicles['left']
                intersection_time.append(signals[currentGreen].green)
                if len(vehicles_in_lane) > 0:
                    for i in range(len(vehicles_in_lane) - 1):
                        g = len(vehicles_in_lane[i])
                        for k in range(g):
                            if vehicles_in_lane[i][k].crossed == 1:
                                number_out += 1

                    print(number_out)
                    if number_out == count_for_each_intersextion[2]:
                        signals[currentGreen].green
                        turn += 1
                        number_out = 0

            elif turn == 3:
                vehicles_in_lane = vehicles['up']
                intersection_time.append(signals[currentGreen].green)
                if len(vehicles_in_lane) > 0:
                    for i in range(len(vehicles_in_lane) - 1):
                        g = len(vehicles_in_lane[i])
                        for k in range(g):
                            if vehicles_in_lane[i][k].crossed == 1:
                                number_out += 1

                    print(number_out)
                    if number_out == count_for_each_intersextion[3]:
                        signals[currentGreen].green
                        turn += 1
                        number_out = 0

                        new_data = {
                            'Distribution 1': [count_for_each_intersextion[0]],
                            'Distribution 2': [count_for_each_intersextion[1]],
                            'Distribution 3': [count_for_each_intersextion[2]],
                            'Distribution 4': [count_for_each_intersextion[3]],
                            'Total Time': [timeElapsed]
                        }

                        new_df = pd.DataFrame(new_data)

                        file_path = 'static.xlsx'

                        try:
                            existing_df = pd.read_excel(file_path)
                            updated_df = existing_df._append(new_df, ignore_index=True)
                            updated_df.to_excel(file_path, index=False)
                            print(f"Data appended to {file_path}")
                            to_laod = 1
                        except FileNotFoundError:
                            new_df.to_excel(file_path, index=False)
                            print(f"Data saved to {file_path}")

        for i in range(0, noOfSignals):
            if(i == currentGreen):
                if(currentYellow == 0):
                    print(" GREEN TS", i+1, "-> r:", signals[i].red, " y:", signals[i].yellow, " g:", signals[i].green)
                else:
                    print("YELLOW TS", i+1, "-> r:", signals[i].red, " y:", signals[i].yellow, " g:", signals[i].green)
            else:
                print("   RED TS", i+1, "-> r:", signals[i].red, " y:", signals[i].yellow, " g:", signals[i].green)
        print()

        for i in range(0, noOfSignals):
            if(i == currentGreen):
                if(currentYellow == 0):
                    signals[i].signalText = signals[i].green
                    screen.blit(greenSignal, signalCoods[i])
                else:
                    signals[i].signalText = signals[i].yellow
                    screen.blit(yellowSignal, signalCoods[i])
            else:
                if(signals[i].red <= 10):
                    if(signals[i].red == 0):
                        signals[i].signalText = "GO"
                    else:
                        signals[i].signalText = signals[i].red
                else:
                    signals[i].signalText = "---"
                screen.blit(redSignal, signalCoods[i])

        signalTexts = ["", "", "", ""]
        for i in range(0, noOfSignals):
            signalTexts[i] = font.render(str(signals[i].signalText), True, white, black)
            screen.blit(signalTexts[i], signalTimerCoods[i])
            displayText = vehicles[directionNumbers[i]]['crossed']
            vehicleCountTexts[i] = font.render(str(displayText), True, black, white)
            screen.blit(vehicleCountTexts[i], vehicleCountCoods[i])

        timeElapsedText = font.render(("Time Elapsed: "+str(timeElapsed)), True, black, white)
        screen.blit(timeElapsedText, (1100, 50))

        for vehicle in simulation:
            screen.blit(vehicle.currentImage, [vehicle.x, vehicle.y])
            vehicle.move()
        pygame.display.update()


Main()
