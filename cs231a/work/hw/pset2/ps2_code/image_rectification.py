#!/usr/bin/env python2

import numpy as np
import matplotlib.pyplot as plt
from fundamental_matrix_estimation import *


def compute_epipole(points1, points2, F):
    '''
    COMPUTE_EPIPOLE computes the epipole in homogenous coordinates
    given matching points in two images and the fundamental matrix
    Arguments:
        points1 - N points in the first image that match with points2
        points2 - N points in the second image that match with points1
        F - the Fundamental matrix such that (points1)^T * F * points2 = 0

        Both points1 and points2 are from the get_data_from_txt_file() method
    Returns:
        epipole - the homogenous coordinates [x y 1] of the epipole in the image
    '''
    # Compute epipole using epipolar lines, Le = 0
    el = np.dot(points2, F)
    U, s, V = np.linalg.svd(el)
    e = V.T[:, -1]

    # Normalize the epipole
    e /= e[2]

    return e
    
def compute_matching_homographies(e2, F, im2, points1, points2):
    '''
    COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
    rectify a pair of images
    Arguments:
        e2 - the second epipole
        F - the Fundamental matrix
        im2 - the second image
        points1 - N points in the first image that match with points2
        points2 - N points in the second image that match with points1
    Returns:
        H1 - the homography associated with the first image
        H2 - the homography associated with the second image
    '''
    # Get width, height of image
    h, w = im2.shape

    # Compute translation matrix
    T = np.array([
        [1, 0, -w/2.0],
        [0, 1, -h/2.0],
        [0, 0, 1]
    ])

    # Translate the epipole
    et = np.dot(T, e2)

    # Compute rotation matrix
    if et[0] >= 0:
        a = 1
    else:
        a = -1
    en = np.linalg.norm(et)
    R = np.array([
        [a*(et[0]/en), a*(et[1]/en), 0],
        [-a*(et[1]/en), a*(et[0]/en), 0],
        [0, 0, 1]
    ])

    # Compute point at infinity transform
    f = np.dot(R, et)[0]
    G = np.array([
        [1,    0, 0],
        [0,    1, 0],
        [-1/f, 0, 1]
    ])

    # Compute H2 = (T^-1)GRT
    H2 = np.linalg.pinv(T).dot(G).dot(R).dot(T)

    # Convert 3-vector to cross product form
    def _cross(x):
        return np.array([
            [0,     -x[2], x[1] ],
            [x[2],  0,     -x[0]],
            [-x[1], x[0],  0    ]
        ])

    # Compute M
    v = np.array([1, 1, 1])
    M = np.dot(_cross(e2), F) + np.outer(e2, v)

    # Compute transformed points
    p1h = H2.dot(M).dot(points1.T).T
    p2h = H2.dot(points2.T).T
    
    # Normalize transformed points
    p1h /= p1h[:, -1][:, np.newaxis]
    p2h /= p2h[:, -1][:, np.newaxis]

    # Compute Ha
    W = p1h 
    b = p2h[:, 0]
    a, r, _, _ = np.linalg.lstsq(W, b)
    Ha = np.eye(3)
    Ha[0, :] = a

    # Compute H1 = Ha*H2*M
    H1 = Ha.dot(H2).dot(M)

    return H1, H2

if __name__ == '__main__':
    # Read in the data
    im_set = 'data/set1'
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    e1 = compute_epipole(points1, points2, F)
    e2 = compute_epipole(points2, points1, F.transpose())
    print "e1", e1
    print "e2", e2

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print "H1:\n", H1
    print
    print "H2:\n", H2

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new, im_set=im_set, alg='Epipolar Lines on Rectified Images')
    plt.show()
