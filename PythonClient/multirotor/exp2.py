
from __future__ import print_function
import os
import neat
import visualize
# 2-input XOR inputs and expected outputs.
# In settings.json first activate computer vision mode: 
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode
import setup_path 
import airsim
import numpy as np
import cv2
import time
import sys
import multiprocessing
import os
import tempfile
import pprint
import pickle
import neat
pp = pprint.PrettyPrinter(indent=4)

def printUsage():
   print("Usage: python camera.py [depth|segmentation|scene]")
cameraType = "scene"
for arg in sys.argv[1:]:
  cameraType = arg.lower()
cameraTypeMap = { 
 "depth": airsim.ImageType.DepthVis,
 "segmentation": airsim.ImageType.Segmentation,
 "seg": airsim.ImageType.Segmentation,
 "scene": airsim.ImageType.Scene,
 "disparity": airsim.ImageType.DisparityNormalized,
 "normals": airsim.ImageType.SurfaceNormals
}
if (not cameraType in cameraTypeMap):
  printUsage()
  sys.exit(0)
print (cameraTypeMap[cameraType])
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
car_cascade = cv2.CascadeClassifier( r'C:\Users\kartikeya singh\Desktop\myplate_cas\classifier\cascade.xml')

airsim.wait_key('Press any key to get camera parameters')
for camera_id in range(2):
    camera_info = client.simGetCameraInfo(str(camera_id))
    print("CameraInfo %d: %s" % (camera_id, pp.pprint(camera_info)))

airsim.wait_key('Press any key to get images')
tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
print ("Saving images to %s" % tmp_dir)
try:
    for n in range(3):
        os.makedirs(os.path.join(tmp_dir, str(n)))
except OSError:
    if not os.path.isdir(tmp_dir):
        raise
yo=0
global point_a
global point_b
global point_c
global point_d
global point_e
global point_f
global point_g
global point_h
       
for i in range(1, 100):
         
            client.moveToPositionAsync(-5-yo,5,-9, 1).join()
            yo=yo+1
            rawImage = client.simGetImage("0", cameraTypeMap[cameraType])
            if (rawImage == None):
                print("Camera is not returning image, please check airsim for error messages")
                sys.exit(0)
            else:

                png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
                gray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
                cars = car_cascade.detectMultiScale(gray, 1.01, 1)
                for (x, y, w, h) in cars:
                    cv2.rectangle(png, (x,y), (x+w,y+h), (255,0,0), 2)
                    cv2.rectangle(png, (2, 2) , (316, 177) , (0,255,0),2)
                cv2.imshow("View", png)
            key = cv2.waitKey(1) & 0xFF;
            if (key == 27 or key == ord('q') or key == ord('x')):
                  break;
      
    

            
            
            
  
           