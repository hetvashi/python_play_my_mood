#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
import webbrowser 
import pywhatkit as kit
from playsound import playsound


# In[2]:


img = cv2.imread('boy.jfif')


# In[3]:


plt.imshow(img)


# In[4]:


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[9]:


video_capture = cv2.VideoCapture(1)
if not video_capture.isOpened():
    video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    raise IOError("Cannot open webcam")


# In[11]:


while True: 
    ret, img = video_capture.read()
    result = DeepFace.analyze(img , actions =['emotion'])
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.1,4)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,result['dominant_emotion'],(50,50),font,3,(0,0,255),2,cv2.LINE_4)
    cv2.imshow('video', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break


plt.imshow(img)
video_capture.release()
cv2.destroyAllWindows()


# In[7]:


if(result['dominant_emotion']=='happy'):
    webbrowser.open("https://www.youtube.com/watch?v=A-sfd1J8yX4")
elif(result['dominant_emotion']=='sad'):
    webbrowser.open("https://www.youtube.com/watch?v=i_k3K772Zyk")
elif(result['dominant_emotion']=='angry'):
    webbrowser.open("https://www.youtube.com/watch?v=Ux-BoW8h6BA")
elif(result['dominant_emotion']=='energetic'):
    webbrowser.open("https://www.youtube.com/watch?v=n1oaPb_UTxs")
elif(result['dominant_emotion']=='neutral'):
    webbrowser.open("https://www.youtube.com/watch?v=g3M10O_eGV4")
else:
    print("No songs found");


# In[12]:


if(result['dominant_emotion']=='happy'):
    kit.playonyt("happy"+"songs")
elif(result['dominant_emotion']=='sad'):
     kit.playonyt("sad"+"songs")
elif(result['dominant_emotion']=='angry'):
     kit.playonyt("relaxing"+"songs")
elif(result['dominant_emotion']=='energetic'):
     kit.playonyt("energetic"+"songs")
elif(result['dominant_emotion']=='neutral'):
      kit.playonyt("songs")
else:
    print("No songs found");


# In[ ]:


if(result['dominant_emotion']=='happy'):
    playsound("happy.mp3")
elif(result['dominant_emotion']=='sad'):
     playsound("sad.mpeg")
elif(result['dominant_emotion']=='angry'):
     playsound("relaxing.mpeg")
elif(result['dominant_emotion']=='energetic'):
     playsound("energetic.mp3")
elif(result['dominant_emotion']=='neutral'):
      playsound("songs.mpeg")
else:
    print("No songs found")


# In[ ]:





# In[ ]:




