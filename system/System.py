import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'inscompel')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'system')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'inscompel')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'inscompel')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'inscompel')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'inscompel')))

import code_ins   # type:ignore
from Yolo9seg import *
from Yolo8 import *
from RTDTR import *
from yolo3 import *
from yolo9 import *
from yolov8seg import *
from faster_test import *
def set_background():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image = image.resize((root.winfo_width(), root.winfo_height()), Image.Resampling.LANCZOS)
        bg_image = ImageTk.PhotoImage(image)
        bg_label.config(image=bg_image)
        bg_label.image = bg_image

def button_action1():
    run_8()
    
def button_action2():
    run_9()
    print("Button 2 pressed")

def button_action3(): 
    print("Button 3 pressed")
    run_3()

def button_action4():
    run_seg8()
    print("Button 4 pressed")

def button_action5():
    run_seg9()
    print("Button 4 pressed")

def button_action6():
   run_rt()
   print("Button 6 pressed")


def button_action7(): 
    run_faster()
    print("Button 1 pressed")


def button_action8():
    code_ins.start_ins()
    print("Button 1 pressed")



root = tk.Tk()
root.title("Background and Buttons App")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}")
bg_label = tk.Label(root)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)
file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Set Background", command=set_background)

button_texts = ["Yolov8", "Yolov9", "Yolov3", "Yolov8-seg", "Yolov9-seg","RtDetr", "Faster_Rcnn","Ensamble"]
button_actions = [button_action1, button_action2, button_action3, button_action4,button_action5,button_action6,button_action7,button_action8]
buttons = []
button_width = 100
total_width = button_width * len(button_texts)
start_x = (screen_width - total_width) / 2

for i, (text, action) in enumerate(zip(button_texts, button_actions)):
    button = tk.Button(root, text=text, command=action)
    button.place(x=start_x + i * button_width, y=screen_height/2, width=button_width, height=40)
    buttons.append(button)

root.mainloop()