#!/usr/bin/env python2

import os
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import scipy.io as sio
from epipolar_utils import *

def lls_eight_point_alg(points1, points2, test=False):
    '''
    LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using 
    linear least squares eight point algorithm
    Arguments:
        points1 - N points in the first image that match with points2
        points2 - N points in the second image that match with points1

        Both points1 and points2 are from the get_data_from_txt_file() method
    Returns:
        Fp - the fundamental matrix such that (points2)^T * Fp * points1 = 0
    Please see lecture notes and slides to see how the linear least squares eight
    point algorithm works
    '''
    # Convert point pairs to outer product matrix form
    A = []
    for p1, p2 in zip(points1, points2):
        row = np.outer(p2, p1).reshape(9)
        A.append(row)
    A = np.array(A)
    # Take SVD of matrix
    _, _, V = np.linalg.svd(A)
    # Last column of V is solution
    f = V.T[:, -1]
    # Af should be close to 0
    if test:
        null = A.dot(f)
        print 'Af =', null
    # Convert solution vector f to fundamental matrix F
    F = f.reshape(3, 3)
    # Point, F products should be close to 0
    if test:
        for p1, p2 in zip(points1, points2):
            print(p2.T.dot(F).dot(p1))
    # SVD again to minimize Frobenius norm
    Up, dp, Vp = np.linalg.svd(F)
    # Zero out last element of Dp to meet rank 2 constraint
    dp[-1] = 0
    Dp = np.diag(dp)
    # Build Fp from this
    Fp = np.dot(Up, np.dot(Dp, Vp))
    # Point, F products should be close to 0
    if test:
        for p1, p2 in zip(points1, points2):
            print(p2.T.dot(Fp).dot(p1))

    return Fp

def normalized_eight_point_alg(points1, points2, test=False):
    '''
    NORMALIZED_EIGHT_POINT_ALG  computes the fundamental matrix from matching points
    using the normalized eight point algorithm
    Arguments:
        points1 - N points in the first image that match with points2
        points2 - N points in the second image that match with points1

        Both points1 and points2 are from the get_data_from_txt_file() method
    Returns:
        F - the fundamental matrix such that (points2)^T * F * points1 = 0
    Please see lecture notes and slides to see how the normalized eight
    point algorithm works
    '''
    ### Normalize points
    # Dump z values for now
    xy1 = points1[:, :2]
    xy2 = points2[:, :2]
    # Compute translation params
    t1 = np.mean(xy1, axis=0)
    t2 = np.mean(xy2, axis=0)
    # Compute scaling params
    msd1 = ((xy1 - t1)**2).sum(axis=1).mean()
    msd2 = ((xy2 - t2)**2).sum(axis=1).mean()
    s1 = np.sqrt(2/msd1)
    s2 = np.sqrt(2/msd2)
    # Scale translation params
    t1 *= s1
    t2 *= s2
    # Build transformation matrices
    T1 = np.array([
        [s1, 0,      -t1[0]],
        [0,     s1,  -t1[1]],
        [0,     0,      1]
    ])
    T2 = np.array([
        [s2, 0,      -t2[0]],
        [0,     s2,  -t2[1]],
        [0,     0,      1]
    ])
    # Normalize the points
    np1 = np.dot(points1, T1.T)
    np2 = np.dot(points2, T2.T)
    # Make sure means are close to zero, and MSD are close to 2
    if test:
        print 'mean1:', np1.mean(axis=0)
        print 'mean2:', np2.mean(axis=0)
        print 'msd1:', (np1[:, :2]**2).sum(axis=1).mean()
        print 'msd2:', (np2[:, :2]**2).sum(axis=1).mean()
    ### Call eight point alg on normalized point sets
    Fhp = lls_eight_point_alg(np1, np2)
    ### Denormalize results
    F = T2.T.dot(Fhp).dot(T1)

    return F

def plot_epipolar_lines_on_images(points1, points2, im1, im2, F, im_set=None, alg=None):
    '''
    PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
    draws the epipolar lines on the images
    Arguments:
        points1 - N points in the first image that match with points2
        points2 - N points in the second image that match with points1
        im1 - a HxW(xC) matrix that contains pixel values from the first image 
        im2 - a HxW(xC) matrix that contains pixel values from the second image 
        F - the fundamental matrix such that (points2)^T * F * points1 = 0

        Both points1 and points2 are from the get_data_from_txt_file() method
    Returns:
        Nothing; instead, plots the two images with the matching points and
        their corresponding epipolar lines. See Figure 1 within the problem set
        handout for an example
    '''
    # Large figure to fit plots without distortion
    plt.figure(figsize=(20, 10))
    plt.suptitle('{} {}'.format(im_set, alg), fontsize=20)
    # Subroutine to draw epipolar lines on one image
    def plot_epipolar_line_on_image(points1, points2, im, F, i):
        # Set up plot aesthetics
        plt.subplot(int('12{}'.format(i)))
        plt.imshow(im, cmap='gray')
        plt.axis('off')
        plt.xlim(0, 512)
        plt.ylim(512, 0)
        # Solve for epipolar points
        # l = [a, b, c] where: ax + by + c = 0
        el = np.dot(points2, F)
        # Plot each epipolar line
        for a, b, c in el:
            x = np.arange(0, 512)
            y = -(a/b)*x - c/b
            # Threshold points so lines are bounded by images
            idx1 = y >= 0
            idx2 = y <= 512
            idx = idx1 & idx2
            x = x[idx]
            y = y[idx]
            plt.plot(x, y, c='red')
        # Plot original points on images
        x, y, z = zip(*points1)
        plt.scatter(x, y, s=80, c='blue', marker=(5, 1))
        plt.title('Image {}'.format(i), fontsize=16)

    # Plot epipolar lines on first image
    plot_epipolar_line_on_image(points1, points2, im1, F, 1)
    # Plot epipolar lines on second image
    plot_epipolar_line_on_image(points2, points1, im2, F.T, 2)
    # Save plots
    path = os.path.join('plots', '{}_{}.png'.format(im_set.replace('/', '_'), alg))
    plt.savefig(path)
    

def compute_distance_to_epipolar_lines(points1, points2, F):
    '''
    COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set a 
    points to their corresponding epipolar lines
    Arguments:
        points1 - N points in the first image that match with points2
        points2 - N points in the second image that match with points1
        F - the fundamental matrix such that (points2)^T * F * points1 = 0

        Both points1 and points2 are from the get_data_from_txt_file() method
    Returns:
        average_distance - the average distance of each point to the epipolar line
    '''
    # Compute epipolar lines for points2 using F
    el = np.dot(points2, F)
    # Normalize lines, points already normalized
    ea, eb, ec = el.T
    la, lb, lc = ea/np.sqrt(ea**2 + eb**2), eb/np.sqrt(ea**2 + eb**2), ec/np.sqrt(ea**2 + eb**2)
    en = zip(la, lb, lc)
    dist_list = []
    # Measure distance between each epipolar line and correspondig point in points1
    for e, p in zip(en, points1):
        d = np.dot(e, p)
        dist_list.append(d)
    # Compute average distance
    avg_dist = np.mean(dist_list)
    return avg_dist

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set2']:
        print '-'*80
        print "Set:", im_set
        print '-'*80

        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
        points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        # Running the linear least squares eight point algorithm
        F_lls = lls_eight_point_alg(points1, points2)
        pFp = [points2[i].dot(F_lls.dot(points1[i])) 
            for i in xrange(points1.shape[0])]
        print "p'^T F p =", np.abs(pFp).max()
        print "Fundamental Matrix from LLS  8-point algorithm:\n", F_lls
        print "Distance to lines in image 1 for LLS:", \
            compute_distance_to_epipolar_lines(points1, points2, F_lls)
        print "Distance to lines in image 2 for LLS:", \
            compute_distance_to_epipolar_lines(points2, points1, F_lls.T)

        # Running the normalized eight point algorithm
        F_normalized = normalized_eight_point_alg(points1, points2)

        pFp = [points2[i].dot(F_normalized.dot(points1[i])) 
            for i in xrange(points1.shape[0])]
        print "p'^T F p =", np.abs(pFp).max()
        print "Fundamental Matrix from normalized 8-point algorithm:\n", \
            F_normalized
        print "Distance to lines in image 1 for normalized:", \
            compute_distance_to_epipolar_lines(points1, points2, F_normalized)
        print "Distance to lines in image 2 for normalized:", \
            compute_distance_to_epipolar_lines(points2, points1, F_normalized.T)

        # Plotting the epipolar lines
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls, im_set=im_set, alg='LLS 8-Point Algorithm')
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized, im_set=im_set, alg='Normalized 8-Point Algorithm')

        #plt.show()
