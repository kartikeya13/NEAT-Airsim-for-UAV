

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
import pickle
import neat
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
car_cascade = cv2.CascadeClassifier( r'C:\Users\kartikeya singh\Desktop\my_cas\classifier\cascade.xml')

def eval_genomes(genomes, config):
    
    kar=10
    karr=40

    for genome_id, genome in genomes:
        
     #   x = [kar,-karr]
        x=[14]
        kar= kar+10
        karr= karr+12
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        output = net.activate(x)
      #  print(output)
        yo=0
        global point_a
        global point_b
        global point_c
        global point_d
        global point_e
        global point_f
        global point_g
        global point_h
       #for i in range(1, 3):
        while True:
            xx=int(output[0])
            xxx=int(output[1])
            client.moveToPositionAsync(-19-yo, 5+xx, -9-xxx,1).join()
          #  client.moveToPositionAsync(-5-yo, 5, -9,1).join()

            #client.moveToPositionAsync(-yo, xx, -xxx,1).join()
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
                    cv2.line(png, (x,y), (2,2), (255,0,255), 1)
                    cv2.line(png, (x+w,y+h), (316,177), (0,255,255), 1)
                    cv2.line(png,(x+w,y) ,(316,2), (0,255,0), 1)
                    cv2.line(png, (x,y+h), (2,177) , (0,0,255), 1)
                 
                    point_a = np.array((x,y))
                    point_b = np.array((2,2))
                    point_c = np.array((x+w,y+h))
                    point_d = np.array((316,177))
                    point_e = np.array((x+w,y))
                    point_f = np.array((316,2))
                    point_g = np.array((x,y+h))
                    point_h = np.array((2,177))         
            Left_top = np.linalg.norm(point_a - point_b)
            Left_bottom= np.linalg.norm(point_c - point_d)
            Right_top= np.linalg.norm(point_e - point_f)
            Right_bottom= np.linalg.norm(point_g - point_h)
                    
            k=abs(Left_top-Right_bottom)
            n=abs(Left_bottom-Right_top)
            #print("K:-")
            #print(k)
            #print(n)
            #print("/K:-")
            cv2.imshow("View", png)
            key = cv2.waitKey(1) & 0xFF;
            if (key == 27 or key == ord('q') or key == ord('x')):
                  break;
            if (k<4 and n<4):
                   print("PERFECTLY ALIGNED TOWARDS THE TARGET")
                   genome.fitness += 1 
                   
               #   client.moveToPositionAsync(-5-xx, 5+yo, -9-xxx, 1).join()
            else:
                   print("Not aligned")
                   break
            
            
            


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)