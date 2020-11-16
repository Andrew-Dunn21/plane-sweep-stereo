from imageio import imwrite
import cv2
import numpy as np
import sys
import time

from util import preprocess_ncc, pyrdown, get_depths, \
    unproject_corners, compute_ncc, project

from dataset import load_dataset

from gifwriter import GifWriter

assert len(sys.argv) > 1
data = load_dataset(sys.argv[1])

K_right = data.K_right
K_left = data.K_left
Rt_right = data.Rt_right
Rt_left = data.Rt_left

ncc_size = data.ncc_size
width = data.width
height = data.height

ncc_gif_writer = GifWriter(data.ncc_temp, data.ncc_gif)
projected_gif_writer = GifWriter(data.projected_temp, data.projected_gif)


# Images are pretty large, so we're going to reduce their size
# in each dimension.
right = data.right[0]
left = data.left[0]
for i in range(data.stereo_downscale_factor):
    right = pyrdown(right)
    left = pyrdown(left)
    height, width, _ = right.shape
    K_left[:2, :] /= 2
    K_right[:2, :] /= 2

# We give you the depth labels for this problem.
depths = get_depths(data)

tic = time.time()

# The planes will be swept fronto-parallel to the right camera, so no
# reprojection needs to be done for this image.  Simply compute the normalized
# patches across the entire image.
right_normalized = preprocess_ncc(right[:, :, :3], ncc_size)


# We'll sweep a series of planes that are fronto-parallel to the right camera.
# The image from the left camera is to be projected onto each of these planes,
# normalized, and then compared to the normalized right image.
volume = []
for pos, depth in enumerate(depths):
    # Task 5: complete the plane sweep stereo loop body by filling in
    # the following TODO lines

    # (TODO) Unproject the pixel coordinates from the right camera onto the virtual plane.
    points = unproject_corners(K_right, width, height, depth, Rt_right)

    # (TODO) Project the 3D corners into the two cameras to generate correspondences.
    points_left = project(K_left, Rt_left, points).reshape((4, 2))
    points_right = project(K_right, Rt_right, points).reshape((4, 2))

    # Solve for a homography to map the left image to the right:
    # Note: points_left and points_right should have shape 4x2
    # where each row contains (x,y)
    H, _ = cv2.findHomography(points_left, points_right)

    # Warp left image onto right image
    projected_left = cv2.warpPerspective(left, H, (width, height))

    # (TODO) Normalize the left image in preparation for NCC
    left_normalized = preprocess_ncc(projected_left, ncc_size)

    # (TODO) Compute the NCC score between the right and left images.
    ncc = compute_ncc(right_normalized, left_normalized)

    # append computed ncc to volume - added this so that we didn't get error later when trying to stack values in volume
    volume.append(ncc)

    # generate outputs and report progress:
    # write this slice of the cost volume to a frame of a gif
    projected_gif_writer.append(np.uint8(projected_left))
    ncc_gif_writer.append(np.uint8(255 * np.clip(ncc / 2 + 0.5, 0, 1)))

    sys.stdout.write('Progress: {0}\r'.format(int(100 * pos / len(depths))))
    sys.stdout.flush()

toc = time.time()

print('Plane sweep took {0} seconds'.format(toc - tic))

ncc_gif_writer.close()
projected_gif_writer.close()

# Stack NCC layers get together into a volume.
volume = np.dstack(volume)

# Use the simplest algorithm to select a depth layer per pixel: i
# the argmax across depth labels.
solution = volume.argmax(axis=2)

print('Saving NCC to {0}'.format(data.ncc_png))
imwrite(data.ncc_png, (solution * 2).astype(np.uint8))


# Remap the label IDs back to their associated depth values.
solution = depths[solution]

print('Saving depth to {0}'.format(data.depth_npy))
np.save(data.depth_npy, solution)
