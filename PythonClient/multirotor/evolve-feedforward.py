

from __future__ import print_function
import os
import neat
import visualize

# 2-input XOR inputs and expected outputs.
# In settings.json first activate computer vision mode: 
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode
from __future__ import print_function
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
    x = 0
    y = 0

    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        output = net.activate(x)
        #put AirSim here
        for i in range(1, 100):
        client.moveToPositionAsync(-5-x, 5, -9, 1).join()
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
            cv2.imshow("View", png)
        if (Left_top==Right_bottom and Left_bottom==Right_top):
            print("PERFECTLY ALIGNED TOWARDS THE TARGET")
            genome.fitness += reward 
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
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)