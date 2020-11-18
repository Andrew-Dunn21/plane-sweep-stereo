'''
Ryan Parsons
Andrew Dunn

11-17-2020
CSCI 497/597

Program to compute various evaluation metrics on our plane_sweep_stereo output compared to the ground truth
'''

import sys
import imageio
import math
import numpy as np
from util import pyrdown
from dataset import load_dataset

imageio.plugins.freeimage.download()

# load in dataset from command line arg
data = load_dataset(sys.argv[1])

# load in data files
name = sys.argv[1]
depth_file = 'output/{0}_depth.npy'.format(name)
gt_file = 'data/{0}-perfect/disp0.pfm'.format(name)

our_depth = np.load(depth_file)
ground_truth = imageio.imread(gt_file, 'PFM-FI')
ground_truth = np.asarray(ground_truth)

print('files loaded')

# get calibration information for disparity/depth conversion
# modified from the dataset.py file provided
calib_values = {}
with open('data\Flowers-perfect\calib.txt', 'r') as file:
    for line in file:
        name, value = line.strip().split('=')
        calib_values[name] = value

    cam0 = calib_values['cam0']
    cam1 = calib_values['cam1']
    doffs = float(calib_values['doffs'])
    baseline = float(calib_values['baseline'])
    width = int(calib_values['width'])
    height = int(calib_values['height'])

    # since this is a String, work with it to obtain f value
    cam0 = cam0[1:-1]
    tokens = cam0.split(' ')
    f = float(tokens[0])

# reduce size of ground truth image using the pyrdown from util.py
# reduce size by same amount as the plane_sweep_stereo did
for i in range(data.stereo_downscale_factor):
    ground_truth = pyrdown(ground_truth)

# we have some infinite values so set them to the max value we find in the ground truth
ground_truth[ground_truth == float('inf')] = 0
ground_truth[ground_truth == 0] = np.max(ground_truth)

# convert our depths to disparity
our_depth = (baseline * f) / our_depth

print('done converting disparity to depth')

# compute root mean squared error
N = np.size(ground_truth)
h, w = ground_truth.shape
sum = 0
for y in range(h):
    for x in range(w):
        sum += (our_depth[y, x] - ground_truth[y, x])**2
sum /= N
R = math.sqrt(sum)

print("Root Mean Squared Error: " + str(R))

# compute percentage of bad matching pixels
delta = 1.0 # using what the provided paper used

sum = 0
for y in range(h):
    for x in range(w):
        diff = abs(our_depth[y, x] - ground_truth[y, x])
        if diff > delta:
            sum += 1
B = sum / N

print("Percentage of Bad Matching Pixels: " + str(B))
