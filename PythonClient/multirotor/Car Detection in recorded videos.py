# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 11:37:27 2019

@author: vchan
"""

# OpenCV Python program to detect cars in video frame
# import libraries of python OpenCV
import cv2
client = airsim.MultirotorClient()
responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPlanner,
pixels_as_float=True, compress=False),
airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
color = responses[1]
imgcolor = np.fromstring(color.image_data_uint8, dtype=np.uint8)
imgcolor = imgcolor.reshape(responses[1].height, responses[1].width, -1)
if imgcolor.shape[2] == 4:
imgcolor = cv2.cvtColor(imgcolor,cv2.COLOR_RGBA2BGR)
image = Image.fromarray(imgcolor)
#image = my_yolo.detect_image(image)
#result = np.asarray(image)
#cv2.imshow("result", result)

# capture frames from a video
#cap = cv2.VideoCapture( r'C:\Users\kartikeya singh\Desktop\yoyoyo\Computer-Vision---Object-Detection-in-Python-master\n.3GPP')

# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier( r'C:\Users\kartikeya singh\Desktop\yoyoyo\Computer-Vision---Object-Detection-in-Python-master\xml files\cars.xml')

# loop runs if capturing has been initialized.
while True:
    # reads frames from a video
    ret, frames = image.read()
    # convert to gray scale of each frames
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    # Detects cars of different sizes in the input image
    cars = car_cascade.detectMultiScale( gray, 1.1, 1)
    # To draw a rectangle in each cars
    for (x,y,w,h) in cars:
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frames, 'Car', (x + 6, y - 6), font, 0.5, (0, 0, 255), 1)
        # Display frames in a window
        cv2.imshow('Car Detection', frames)
    # Wait for Enter key to stop
    if cv2.waitKey(33) == 13:
        break

cap.release()
cv2.destroyAllWindows()
