'''
Ryan Parsons
Andrew Dunn

11-17-2020
CSCI 497/597

Program to compute various evaluation metrics on our plane_sweep_stereo output compared to the ground truth
'''

import imageio
import numpy as np

imageio.plugins.freeimage.download()

# load in data files
our_depth = np.load('output/Flowers_depth.npy')
ground_truth = imageio.imread('data/Flowers-perfect/disp0.pfm', 'PFM-FI')
ground_truth = np.asarray(ground_truth)

print('files loaded')


