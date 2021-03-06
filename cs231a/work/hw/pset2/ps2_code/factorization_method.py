#!/usr/bin/env python2

import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.gridspec as gridspec
from epipolar_utils import *

def factorization_method(points_im1, points_im2):
    '''
    FACTORIZATION_METHOD The Tomasi and Kanade Factorization Method to determine
    the 3D structure of the scene and the motion of the cameras.
    Arguments:
        points_im1 - N points in the first image that match with points_im2
        points_im2 - N points in the second image that match with points_im1

        Both points_im1 and points_im2 are from the get_data_from_txt_file() method
    Returns:
        structure - the structure matrix
        motion - the motion matrix
    '''
    # Dump last column of 1s
    xy1 = points_im1[:, :2]
    xy2 = points_im2[:, :2]

    # Compute centroids for points in each image
    u1 = np.mean(xy1, axis=0)
    u2 = np.mean(xy2, axis=0)

    # Subtract centroids from points in each image
    xy1c = xy1 - u1
    xy2c = xy2 - u2

    # Stack the observations from different cameras to produce D
    D = np.concatenate([xy1c.T, xy2c.T], axis=0)

    # Perform SVD on D
    U, b, V = np.linalg.svd(D, full_matrices=False)

    print 'Singular values:', b

    # Take columns and rows of U and V corresponding to 3 largest singular values
    U3 = U[:, :3]
    b3 = b[:3]
    V3 = V[:3, :]
    B3 = np.diag(b3)

    # Factorize D into structure and motion matrices
    structure = np.dot(np.sqrt(B3), V3)
    motion = np.dot(U3, np.sqrt(B3))

    return structure, motion
    

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set1_subset']:
        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points_im1 = get_data_from_txt_file(im_set + '/pt_2D_1.txt')
        points_im2 = get_data_from_txt_file(im_set + '/pt_2D_2.txt')
        points_3d = get_data_from_txt_file(im_set + '/pt_3D.txt')
        assert (points_im1.shape == points_im2.shape)

        # Run the Factorization Method
        structure, motion = factorization_method(points_im1, points_im2)

        # Plot the structure
        fig = plt.figure()
        ax = fig.add_subplot(121, projection = '3d')
        scatter_3D_axis_equal(structure[0,:], structure[1,:], structure[2,:], ax)
        ax.set_title('Factorization Method')
        ax = fig.add_subplot(122, projection = '3d')
        scatter_3D_axis_equal(points_3d[:,0], points_3d[:,1], points_3d[:,2], ax)
        ax.set_title('Ground Truth')

        plt.show()
