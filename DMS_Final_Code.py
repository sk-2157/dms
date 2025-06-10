#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import os
import winsound
import csv    
import datetime



def start():
    
    global status
    status = True
    
    def eye_aspect_ratio(eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])

        ear = (A + B) / (2.0 * C)
        return ear

    def mouth_aspect_ratio(mouth):
        A = distance.euclidean(mouth[2], mouth[10])
        B = distance.euclidean(mouth[4], mouth[8])
        C = distance.euclidean(mouth[0], mouth[6]) 

        mar = (A + B) / (2.0 * C)
        return mar
    

    ear_thresh = 0.25
    mar_thresh = 0.7
    frame_check = 20

    detect = dlib.get_frontal_face_detector()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "shape_predictor_68_face_landmarks.dat")
    predict = dlib.shape_predictor(model_path)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    global cap
    cap=cv2.VideoCapture(0)

    flag=0
    yawn=0
    ear_mar = []

    while status:
        ret, frame=cap.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape) 

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            ear_disp = "EAR: {:.2f}".format(ear)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)        

            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            cv2.putText(frame,ear_disp , (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)    

            if ear < ear_thresh:
                flag += 1            
                if flag >= frame_check:
                    cv2.putText(frame, "***Eyes Closed***", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "***ALERT!***", (10,325),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                    freq , dur = 900,100
                    winsound.Beep(freq, dur)
                   
            else:
                flag = 0    


            mouth = shape[mStart:mEnd]
            mar = mouth_aspect_ratio(mouth)
            mar_disp = "MAR: {:.2f}".format(mar)
            mouthHull = cv2.convexHull(mouth)

            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

            cv2.putText(frame,mar_disp , (310, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)            


            if mar > mar_thresh:            
                yawn += 1            
                if yawn > 5:            
                    cv2.putText(frame, "***Yawn Detected***", (220, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "***ALERT!***", (250,325),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                    freq , dur = 900,100
                    winsound.Beep(freq, dur)
                    
            else:
                yawn = 0

            ct = datetime.datetime.now()
            e = float("{:.2f}".format(ear))
            m = float("{:.2f}".format(mar))
            ear_mar.append([ct,e,m])


        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            cv2.destroyAllWindows()
            cap.release()            
            break
            

    fname = "User_Data/{}.csv".format(username)
    with open(fname, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(ear_mar)
        
        
    fname = "User_Data_Graph/{}.csv".format(username)
    with open(fname, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(ear_mar)
              


# In[2]:


def display_graph():
    get_ipython().run_line_magic('matplotlib', 'tk')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    colnames = ["Timestamp", "EAR","MAR"]
    df = pd.read_csv("User_Data_Graph/{}.csv".format(username),names=colnames, header=None)
    dates = df["Timestamp"]
    A = df["EAR"]
    B = df["MAR"]
    

    date_range = pd.date_range(start = dates.iloc[0], end = dates.iloc[-1] ,periods = 2)

    xaxis_label = []

    for i in date_range:
        xaxis_label.append(str(i))
        


    # Create the figure and subplots
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,5))
    
   
    # Plot the A variable on the first subplot

    ax[0].plot(dates, A)
    ax[0].set_title('EAR Graph')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('EAR')
    ax[0].set_xticks(xaxis_label)
    ax[0].axhline(y = 0.25, color = 'r', linestyle = '-')


    # Plot the B variable on the second subplot
    ax[1].plot(dates, B)
    ax[1].set_title('MAR Graph')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('MAR')
    ax[1].set_xticks(xaxis_label)
    ax[1].axhline(y = 0.7, color = 'r', linestyle = '-')

    # Show the plot in a separate window
    fig.tight_layout()
    plt.show()


# In[3]:


from tkinter import *
from PIL import ImageTk,Image
from tkinter import messagebox
import tkinter as tk
import tkinter.font as font
import ast

global w
w=Tk()
w.geometry('900x500')
w.resizable(0,0)
w.title('Driver Monitoring System Using Machine Learning')
w.configure(bg='#ff4f5a')
w.minsize(900,500)


def cam_page():       
    
    f1=Frame(w,width=900,height=500,bg='white')
    f1.place(x=0,y=0)   
   

    f1.pack(fill='both', expand=True)
    
    
    global image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "images", "cam_bg.png")
    image = Image.open(image_path)
    image = image.resize((900, 500), Image.LANCZOS)
    
    global background_image
    background_image = ImageTk.PhotoImage(image)

    background_label = tk.Label(f1, image=background_image)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
    
    
    Label(f1,text="Driver Monitoring System",font=('Segoe Print',28, 'bold'),fg="navy",bg='white').place(x=200,y=40)
    
    #Yu Gothic UI Semibold
    buttonFont = font.Font(family='Yu Gothic UI Semibold', size=12, weight='bold')   
    
    Button(f1, text='Start Camera',bg='aquamarine1',command=start,width=25,height=2, font=buttonFont).place(x=320,y=140,)
    Button(f1, text='Display Graph',width=25,height=2,bg='aquamarine1',command = display_graph, font=buttonFont).place(x=320,y=230)
    Button(f1, text='Logout',bg='OrangeRed2',command=exit_code, width=8,height=1, font=buttonFont).place(x=390,y=320)
    

   
def signin():
    signin_win=Frame(w,width=900,height=500,bg='white')
    signin_win.place(x=0,y=0)
    f1=Frame(signin_win,width=350,height=350,bg='white')
    f1.place(x=480,y=100)
    
    global img1
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "images", "DMS.png")
    img1 = ImageTk.PhotoImage(Image.open(image_path))
    Label(signin_win,image=img1,border=0,bg='white').place(x=50,y=80)
    
    Label(signin_win,text="Driver Monitoring System",font=('Comic Sans MS',26),fg="maroon",bg='white').place(x=25,y=25)
    

    l2=Label(signin_win,text="Driver Sign In",fg='#ff4f5a',bg='white')
    l2.config(font=('Microsoft YaHei UI',22))
    l2.place(x=530,y=110)



    user_label = Label(f1,text="Username",bg='white',fg='black')
    user_label.config(font=('Microsoft YaHei UI Light',13,'bold'))
    user_label.place(x=60,y=80)
    e1 =Entry(f1,width=25,fg='black',bd=1,bg='white')
    e1.config(font=('Microsoft YaHei UI Light',11, ))
    e1.place(x=60,y=110)
    
    pass_label = Label(f1,text="Password",bg='white',fg='black')
    pass_label.config(font=('Microsoft YaHei UI Light',13,'bold'))
    pass_label.place(x=60,y=160)
    e2 =Entry(f1,width=25,fg='black',bd=1,bg='white',show='*')
    e2.config(font=('Microsoft YaHei UI Light',11, ))
    e2.place(x=60,y=190)

    #-mech------------------------------------------------
    def signin_cmd():
        script_dir = os.path.dirname(os.path.abspath(__file__))
        datasheet_path = os.path.join(script_dir,'datasheet.txt')
        file=open(datasheet_path,'r')
        d=file.read()
        r=ast.literal_eval(d)
        file.close()
        
        global username
        username = e1.get()
        key=e1.get()
        value=e2.get()
        
        
        if key in r.keys() and value==r[key]: 
            cam_page()
        else:
            messagebox.showwarning('try again', 'invalid username or password')


    #------------------------------------------------------
    Button(f1,width=35,pady=7,text='Sign In',bg='#ff4f5a',fg='white',border=0,command=signin_cmd).place(x=35,y=250)
    l1=Label(f1,text="Don't have an account?",fg="black",bg='white')
    l1.config(font=('Yu Gothic Medium',10, ))
    l1.place(x=65,y=300)

    b2=Button(f1,width=6,text='Sign Up',border=0,bg='white',fg='#ff4f5a',command=signup)
    b2.place(x=215,y=300)




def signup():
    signup_win=Frame(w,width=900,height=550,bg='white')
    signup_win.place(x=0,y=0)
    f1=Frame(signup_win,width=350,height=400,bg='white')
    f1.place(x=480,y=70)

    
    global img2
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "images", "DMS.png")
    img2 = ImageTk.PhotoImage(Image.open(image_path))
    Label(signup_win,image=img2,border=0,bg='white').place(x=50,y=80)
    
    Label(signup_win,text="Driver Monitoring System",font=('Comic Sans MS',26),fg="maroon",bg='white').place(x=25,y=25)
    

    l2=Label(signup_win,text="Driver Sign Up",fg='#ff4f5a',bg='white')
    l2.config(font=('Microsoft YaHei UI',22))
    l2.place(x=530,y=110)

    def on_enter(e):
        e1.delete(0,'end')    
    def on_leave(e):
        if e1.get()=='':   
            e1.insert(0,'Username')

    
    e1 =Entry(f1,width=25,fg='black',border=0,bg='white')
    e1.config(font=('Microsoft YaHei UI Light',11, ))
    e1.bind("<FocusIn>", on_enter)
    e1.bind("<FocusOut>", on_leave)
    e1.insert(0,'Username')
    e1.place(x=30,y=110)

    Frame(f1,width=295,height=2,bg='black').place(x=25,y=137)

    #------------------------------------------------------

    def on_enter(e):
        e2.delete(0,'end')    
    def on_leave(e):
        if e2.get()=='':   
            e2.insert(0,'Password')

    
    e2 =Entry(f1,width=21,fg='black',border=0,bg='white')
    e2.config(font=('Microsoft YaHei UI Light',11, ))
    e2.bind("<FocusIn>", on_enter)
    e2.bind("<FocusOut>", on_leave)
    e2.insert(0,'Password')
    e2.place(x=30,y=180)

    Frame(f1,width=295,height=2,bg='black').place(x=25,y=207)

    def on_enter(e):
        e3.delete(0,'end')    
    def on_leave(e):
        if e3.get()=='':   
            e3.insert(0,'Confirm Password')

    
    e3 =Entry(f1,width=21,fg='black',border=0,bg='white')
    e3.config(font=('Microsoft YaHei UI Light',11, ))
    e3.bind("<FocusIn>", on_enter)
    e3.bind("<FocusOut>", on_leave)
    e3.insert(0,'Confirm Password')
    e3.place(x=30,y=130+70+50)

    Frame(f1,width=295,height=2,bg='black').place(x=25,y=157+70+50)    

    
    #Mechenism------------------------------------------------
    
    def signup_cmd():
        key=e1.get()
        value=e2.get()
        value2=e3.get()
        
        if len(key) < 5:
            messagebox.showwarning('try again', 'username should be more than 5 characters')
        elif len(value) < 8:
            messagebox.showwarning('try again', 'password should be more than 8 characters')
        else:
            if value==value2:
                file=open('datasheet.txt','r+')
                d=file.read()
                r=ast.literal_eval(d)    
                dict2={key:value}            
                r.update(dict2)            
                file.truncate(0)
                file.close()

                file=open('datasheet.txt','w')
                w=file.write(str(r))
                file.close()

                messagebox.showinfo("","     successfully signed up     ")
                signin()

            else:
                messagebox.showwarning('try again', 'password should match ')


    #-------------------------------------------------------
    Button(f1,width=39,pady=7,text='Sign Up',bg='#ff4f5a',fg='white',border=0,command=signup_cmd).place(x=35,y=204+60+50)
    l1=Label(f1,text="Already have an account?",fg="black",bg='white')
    l1.config(font=('Yu Gothic Medium',10, ))
    l1.place(x=50,y=250+63+50)

    b2=Button(f1,width=6,text='Sign In',border=0,bg='white',fg='#ff4f5a',command=signin)
    b2.place(x=210,y=250+63+50)

    
def exit_code():
    w.destroy()
    

signin() #default screen

w.mainloop()


# In[ ]:





# In[ ]:




