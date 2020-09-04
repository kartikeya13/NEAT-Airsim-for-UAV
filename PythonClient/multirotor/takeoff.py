import setup_path 
import airsim
import numpy as np
import cv2 
import pprint
import tempfile
import os
import time

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
start_point = (5, 5) 
  
# Ending coordinate, here (220, 220) 
# represents the bottom right corner of rectangle 
end_point = (220, 220) 
  
client.armDisarm(True)
pp = pprint.PrettyPrinter(indent=4)
CAMERA_NAME = '0'
IMAGE_TYPE = airsim.ImageType.Scene
DECODE_EXTENSION = '.jpg'
# Trained XML classifiers describes some features of some object we want to detect 
car_cascade = cv2.CascadeClassifier(r'C:\Users\kartikeya singh\Desktop\movies\Computer-Vision-Tutorial-master\Haarcascades/haarcascade_car.xml') 

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

landed = client.getMultirotorState().landed_state

if landed == airsim.LandedState.Landed:
       print("taking off...") 
       client.takeoffAsync().join()
for x in range(50): # do few times

   
        response_image = client.simGetImage(CAMERA_NAME, IMAGE_TYPE)
        int.from_bytes(response_image, byteorder='big')
        print(type(response_image))
        cap = cv2.VideoCapture(0)
        
        ret, frames = cap.read() 
	            # convert to gray scale of each frames 
        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY) 
	            # Detects cars of different sizes in the input image 
        cars = car_cascade.detectMultiScale(gray, 1.1, 1) 
	            # To draw a rectangle in each cars 
        for (x,y,w,h) in cars: 
               cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2) 
               cv2.rectangle(frames, start_point, end_point, (0,0,255),2) 
               cv2.imshow('video2', frames) 

        for i, response in enumerate(responses):
            if response.pixels_as_float:
                print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_float), pprint.pformat(response.camera_position)))
                airsim.write_pfm(os.path.normpath(os.path.join(tmp_dir, str(x) + "_" + str(i) + '.pfm')), airsim.get_pfm_array(response))
                
            else:
                print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
                airsim.write_file(os.path.normpath(os.path.join(tmp_dir, str(i), str(x) + "_" + str(i) + '.png')), response.image_data_uint8)
                pose = client.simGetVehiclePose()
                pp.pprint(pose)

else:
   print("already flying...")
   client.hoverAsync().join()
cv2.destroyAllWindows() 