import tkinter as Tk
import tkinter.font as font
from tkinter.ttk import *


class Form:
    Elements = {}
    root = None

    def __init__(self):
        self.root = Tk.Tk()
        self.root.geometry("1920x1080")

    def AddText(self, Name, X=None, Y=None, Value=None, Height=None, Width=None, FontSize=17):
        NewText = Tk.Text(self.root, height=Height,
                          width=Width, font=font.Font(size=FontSize))
        
        if Value != None:
            NewText.insert(Tk.END, Value)
        if X != None and Y != None:
            NewText.place(x=X, y=Y)
        self.Elements[Name] = NewText

    def GetTextValue(self, TextName):
        return self.Elements[TextName].get("1.0", 'end-1c')

    def AddButton(self, Name, X=None, Y=None, Value=None, OnClick=None, Heigth=None, Width=None):
        NewBotton = Tk.Button(self.root, text=Value,
                              command=OnClick, height=Heigth, width=Width)
        if X != None and Y != None:
            NewBotton.place(x=X, y=Y)
        self.Elements[Name] = NewBotton

    def AddLable(self, Name, X=None, Y=None, Value=None, FontSize=14, FontFamily="Courier"):
        NewLable = Tk.Label(self.root, text=Value)
        if X != None and Y != None:
            NewLable.place(x=X, y=Y)
        NewLable.config(font=(FontFamily, FontSize))
        self.Elements[Name] = NewLable

    def AddComboBox(self, Name, X=None, Y=None, Values=[], SelectedIndex=None, Width=None, Height=None):
        NewComboBox = Combobox(self.root, width=Width, height=Height)
        if X != None and Y != None:
            NewComboBox.place(x=X, y=Y)
        NewComboBox['values'] = (Values)
        NewComboBox.current(SelectedIndex)
        self.Elements[Name] = NewComboBox

    def GetComboBoxValue(self, Name):
        return self.Elements[Name].get()

    def Distroy(self, Name):
        if Name in self.Elements:
            self.Elements[Name].destroy()


    def ChangeLableText(self, Name, NewText):
        self.Elements[Name].config(text=NewText)
        
    def ChangeTextValue(self, Name, NewValue):
        self.Elements[Name].delete(1.0, "end-1c")
        self.Elements[Name].insert("end-1c", NewValue)
        
    def chnageButtonValue(self, Name, NewText):
        if Name in self.Elements:
            self.Elements[Name].value=NewText

    
        
    def Display(self):
        Tk.mainloop()

