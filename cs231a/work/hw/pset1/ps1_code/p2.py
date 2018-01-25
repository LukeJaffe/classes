#!/usr/bin/env python3

# CS231A Homework 1, Problem 2
import numpy as np

'''
DATA FORMAT

In this problem, we provide and load the data for you. Recall that in the original
problem statement, there exists a grid of black squares on a white background. We
know how these black squares are setup, and thus can determine the locations of
specific points on the grid (namely the corners). We also have images taken of the
grid at a front image (where Z = 0) and a back image (where Z = 150). The data we
load for you consists of three parts: real_XY, front_image, and back_image. For a
corner (0,0), we may see it at the (137, 44) pixel in the front image and the
(148, 22) pixel in the back image. Thus, one row of real_XY will contain the numpy
array [0, 0], corresponding to the real XY location (0, 0). The matching row in
front_image will contain [137, 44] and the matching row in back_image will contain
[148, 22]
'''

'''
COMPUTE_CAMERA_MATRIX
Arguments:
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    camera_matrix - The calibrated camera matrix (3x4 matrix)
'''
def compute_camera_matrix(real_XY, front_image, back_image):
    # Create real point matrix
    front_z = np.zeros(front_image.shape[0])[:, np.newaxis]
    back_z = 150*np.ones(front_image.shape[0])[:, np.newaxis]
    both_z = np.concatenate([front_z, back_z], axis=0)
    twice_real_XY = np.concatenate([real_XY, real_XY], axis=0)
    real_ones = np.ones(twice_real_XY.shape[0])[:, np.newaxis]
    A = np.concatenate([twice_real_XY, both_z, real_ones], axis=1)

    # Create projected point matrix
    both_image = np.concatenate([front_image, back_image], axis=0)
    image_ones = np.ones(both_image.shape[0])[:, np.newaxis]
    b = np.concatenate([both_image, image_ones], axis=1)

    # Solve with least-squares
    xt, _, _, _ = np.linalg.lstsq(A, b)
    x = xt.T

    return x 

'''
RMS_ERROR
Arguments:
     camera_matrix - The camera matrix of the calibrated camera
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    rms_error - The root mean square error of reprojecting the points back
                into the images
'''
def rms_error(camera_matrix, real_XY, front_image, back_image):
    # Create real point matrix
    front_z = np.zeros(front_image.shape[0])[:, np.newaxis]
    back_z = 150*np.ones(front_image.shape[0])[:, np.newaxis]
    both_z = np.concatenate([front_z, back_z], axis=0)
    twice_real_XY = np.concatenate([real_XY, real_XY], axis=0)
    real_ones = np.ones(twice_real_XY.shape[0])[:, np.newaxis]
    A = np.concatenate([twice_real_XY, both_z, real_ones], axis=1)

    # Create projected point matrix
    real_proj = np.concatenate([front_image, back_image], axis=0)

    # Calculate corner locations
    calc_b = np.dot(A, camera_matrix.T)
    calc_proj = calc_b[:, :2]

    # Compute RMS error
    rms_error = np.sqrt(np.sum((calc_proj - real_proj)**2)/real_proj.shape[0])

    return rms_error

if __name__ == '__main__':
    # Loading the example coordinates setup
    real_XY = np.load('real_XY.npy')
    front_image = np.load('front_image.npy')
    back_image = np.load('back_image.npy')

    camera_matrix = compute_camera_matrix(real_XY, front_image, back_image)
    print("Camera Matrix:\n", camera_matrix)
    print()
    print("RMS Error: ", rms_error(camera_matrix, real_XY, front_image, back_image))
