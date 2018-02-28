#!/usr/bin/env python2

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from utils import *
import math
import sys


def match_keypoints(descriptors1, descriptors2, threshold = 0.7):
    '''
    MATCH_KEYPOINTS: Given two sets of descriptors corresponding to SIFT keypoints, 
    find pairs of matching keypoints.

    Note: Read Lowe's Keypoint matching, finding the closest keypoint is not
    sufficient to find a match. thresh is the theshold for a valid match.

    Arguments:
        descriptors1 - Descriptors corresponding to the first image. Each row
            corresponds to a descriptor. This is a ndarray of size (M_1, 128).

        descriptors2 - Descriptors corresponding to the second image. Each row
            corresponds to a descriptor. This is a ndarray of size (M_2, 128).

        threshold - The threshold which to accept from Lowe's Keypoint Matching
            algorithm

    Returns:
        matches - An int ndarray of size (N, 2) of indices that for keypoints in 
            descriptors1 match which keypoints in descriptors2. For example, [7 5]
            would mean that the keypoint at index 7 of descriptors1 matches the
            keypoint at index 5 of descriptors2. Not every keypoint will necessarily
            have a match, so N is not the same as the number of rows in descriptors1
            or descriptors2. 
    '''
    # Initialize empty list of matches
    match_list = []
    for i, d in enumerate(descriptors1):
        # Get dist of this element of descriptors1 and all elements
        # of descriptors2
        dist_arr = np.linalg.norm(descriptors2 - d, axis=1)
        # Get the indeces sorted by distance
        sorted_idx = dist_arr.argsort()
        # Get the two minimum distances
        d1, d2 = dist_arr[sorted_idx[:2]]
        # Compute the distance ratio
        r = d1 / d2
        # If the ratio is less than the threshold, we accept this is a match
        if r < threshold:
            match_list.append([i, sorted_idx[0]])
    # Convert the match list to ndarray
    matches = np.array(match_list)

    return matches


def refine_match(keypoints1, keypoints2, matches, reprojection_threshold = 10,
        num_iterations = 1000):
    '''
    REFINE_MATCH: Filter out spurious matches between two images by using RANSAC
    to find a projection matrix. 

    Arguments:
        keypoints1 - Keypoints in the first image. Each row is a SIFT keypoint
            consisting of (u, v, scale, theta). Overall, this variable is a ndarray
            of size (M_1, 4).

        keypoints2 - Keypoints in the second image. Each row is a SIFT keypoint
            consisting of (u, v, scale, theta). Overall, this variable is a ndarray
            of size (M_2, 4).

        matches - An int ndarray of size (N, 2) of indices that indicate what
            keypoints from the first image (keypoints1)  match with the second 
            image (keypoints2). For example, [7 5] would mean that the keypoint at
            index 7 of keypoints1 matches the keypoint at index 5 of keypoints2). 
            Not every keypoint will necessarily have a  match, so N is not the same
            as the number of rows in keypoints1 or keypoints2. 

        reprojection_threshold - If the reprojection error is below this threshold,
            then we will count it as an inlier during the RANSAC process.

        num_iterations - The number of iterations we will run RANSAC for.

    Returns:
        inliers - A vector of integer indices that correspond to the inliers of the
            final model found by RANSAC.

        model - The projection matrix H found by RANSAC that has the most number of
            inliers.
    '''

    ### Normalize points
    # Dump scale, angle  values for now
    xy1 = keypoints1[:, :2]
    xy2 = keypoints2[:, :2]
    xyh1 = np.concatenate([xy1, np.ones((xy1.shape[0], 1))], axis=1)
    xyh2 = np.concatenate([xy2, np.ones((xy2.shape[0], 1))], axis=1)
    
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
    np1 = np.dot(xyh1, T1.T)
    np2 = np.dot(xyh2, T2.T)

    # Convert back to euclidean
    np1 = np1[:, :2]/np1[:, 2][:, np.newaxis]
    np2 = np2[:, :2]/np2[:, 2][:, np.newaxis]

    # Variables for loop
    best_count = 0
    best_results = (None, None)

    # Iterate num_iterations times
    for _ in range(num_iterations):
        # Sample 4 points from matches to compute homography
        sample_idx = np.random.choice(range(len(matches)), len(matches), replace=False)
        sample_matches = matches[sample_idx[:4]]
        rem1, rem2 = matches.T
        rem_points1 = xyh1[rem1]
        rem_points2 = xy2[rem2]
        
        # Compute a (normalized) homograpgy x' = Hx  using DLT
        row_list = []
        for match in sample_matches:
            u1, v1, = np1[match[0]] 
            u2, v2, = np2[match[1]] 
            row = np.array([
                [-u1, -v1, -1, 0, 0, 0, u1*u2, v1*u2, u2],
                [0, 0, 0, -u1, -v1, -1, u1*v2, v1*v2, v2],
            ])
            row_list.append(row)
        # Form rows into A, to solve Ah = 0
        A = np.concatenate(row_list, axis=0)
        # Apply SVD
        U, d, Vt = np.linalg.svd(A)
        # Solution to Ah = 0 is last column of V
        h = Vt.T[:, -1]
        # Reshape solution to get normalized homography Hn
        Hn = h.reshape(3, 3)
        # Denormalize Hn to get H
        H = np.linalg.pinv(T2).dot(Hn).dot(T1)

        ### Compute reprojection error for remaining points
        # Project the points from image 1 using H
        pred_points2 = np.dot(rem_points1, H.T)
        # Convert the projected points back to Euclidean
        pred_points2 = pred_points2[:, :2]/(pred_points2[:, 2][:, np.newaxis]+1e-8)
        # Measure l2norm between predicted and actual points in image 2
        dist = np.linalg.norm(pred_points2 - rem_points2, axis=1)
        # Compute indeces of good reprojections
        good_idx = dist < reprojection_threshold
        # Get number of good reprojections
        new_count = good_idx.sum()
        # Save result if this is best H yet
        if new_count > best_count:
            best_results = good_idx, H
            best_count = new_count

    return best_results


'''
GET_OBJECT_REGION: Get the parameters for each of the predicted object
bounding box in the image

Arguments:
    keypoints1 - Keypoints in the first image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_1, 4).

    keypoints2 - Keypoints in the second image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_2, 4).

    matches - An int ndarray of size (N, 2) of indices that indicate what
        keypoints from the first image (keypoints1)  match with the second 
        image (keypoints2). For example, [7 5] would mean that the keypoint at
        index 7 of keypoints1 matches the keypoint at index 5 of keypoints2). 
        Not every keypoint will necessarily have a  match, so N is not the same
        as the number of rows in keypoints1 or keypoints2.

    obj_bbox - An ndarray of size (4,) that contains [xmin, ymin, xmax, ymax]
        of the bounding box. Note that the point (xmin, ymin) is one corner of
        the box and (xmax, ymax) is the opposite corner of the box.

    thresh - The threshold we use in Hough voting to state that we have found
        a valid object region.

Returns:
    cx - A list of the x location of the center of the bounding boxes

    cy - A list of the y location of the center of the bounding boxes

    w - A list of the width of the bounding boxes

    h - A list of the height of the bounding boxes

    orient - A list f the orientation of the bounding box. Note that the 
        theta provided by the SIFT keypoint is inverted. You will need to
        re-invert it.
'''
def get_object_region(keypoints1, keypoints2, matches, obj_bbox, thresh = 5, 
        nbins = 4):
    # Compute bbox params
    xmin, ymin, xmax, ymax = obj_bbox
    w1, h1 = xmax - xmin, ymax - ymin
    x1 = xmin + w1/2.0
    y1 = ymin + h1/2.0
    # Compute aspect ratio of bbox
    ar = w1/h1

    # Intialize arrays to store parameter estimates
    param_arr = np.zeros((len(matches), 4))

    # Iterate through all matches
    for i, (midx1, midx2) in enumerate(matches):
        # Extract keypoint data for the match
        u1, v1, s1, t1 = keypoints1[midx1]
        u2, v2, s2, t2 = keypoints2[midx2]
        # Compute parameter estimates
        fs = s2 / s1
        o2 = t2 - t1
        w2, h2 = fs*w1, fs*h1
        x2 = u2 + fs*math.cos(o2)*(x1 - u1) - fs*math.sin(o2)*(y1 - v1)
        y2 = v2 + fs*math.sin(o2)*(x1 - u1) + fs*math.cos(o2)*(y1 - v1)
        # Store the parameter estimates
        param_arr[i, :] = x2, y2, w2, o2

    hist, bins = np.histogramdd(param_arr, bins=nbins)
    bin_arr = np.array(bins)

    bin_med = (bin_arr[:, 1:] - bin_arr[:, :-1])/2.0 + bin_arr[:, :-1]

    thresh_idx = np.where(hist >= thresh)
    med_arr = np.zeros((len(thresh_idx[0]), 4))
    for i, idx in enumerate(zip(*thresh_idx)):
        med_arr[i, :] = bin_med[np.arange(4), idx] 

    # Unpack each parameter array from the result
    try:
        cx, cy, w, orient = zip(*med_arr)
    except ValueError:
        return [], [], [], [], []
    else:
        # Compute heights using aspect ratio
        h = w/ar
        return cx, cy, w, h, orient

'''
MATCH_OBJECT: The pipeline for matching an object in one image with another

Arguments:
    im1 - The first image read in as a ndarray of size (H, W, C).

    descriptors1 - Descriptors corresponding to the first image. Each row
        corresponds to a descriptor. This is a ndarray of size (M_1, 128).

    keypoints1 - Keypoints in the first image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_1, 4).

    im2 - The second image read in as a ndarray of size (H, W, C).

    descriptors2 - Descriptors corresponding to the second image. Each row
        corresponds to a descriptor. This is a ndarray of size (M_2, 128).

    keypoints2 - Keypoints in the second image. Each row is a SIFT keypoint
        consisting of (u, v, scale, theta). Overall, this variable is a ndarray
        of size (M_2, 4).

    obj_bbox - An ndarray of size (4,) that contains [xmin, ymin, xmax, ymax]
        of the bounding box. Note that the point (xmin, ymin) is one corner of
        the box and (xmax, ymax) is the opposite corner of the box.

Returns:
    descriptors - The descriptors corresponding to the keypoints inside the
        bounding box.

    keypoints - The pixel locations of the keypoints that reside in the 
        bounding box
'''
def match_object(im1, descriptors1, keypoints1, im2, descriptors2, keypoints2,
        obj_bbox):
    # Part A
    descriptors1, keypoints1, = select_keypoints_in_bbox(descriptors1,
        keypoints1, obj_bbox)
    matches = match_keypoints(descriptors1, descriptors2)
    #plot_matches(im1, im2, keypoints1, keypoints2, matches)
    
    # Part B
    inliers, model = refine_match(keypoints1, keypoints2, matches)
    #plot_matches(im1, im2, keypoints1, keypoints2, matches[inliers,:])

    # Part C
    cx, cy, w, h, orient = get_object_region(keypoints1, keypoints2,
        matches[inliers,:], obj_bbox)

    plot_bbox(cx, cy, w, h, orient, im2)

if __name__ == '__main__':
    # Load the data
    data = sio.loadmat('SIFT_data.mat')
    images = data['stopim'][0]
    obj_bbox = data['obj_bbox'][0]
    keypoints = data['keypt'][0]
    descriptors = data['sift_desc'][0]
    
    np.random.seed(0)

    for i in [2, 1, 3, 4]:
        match_object(images[0], descriptors[0], keypoints[0], images[i],
            descriptors[i], keypoints[i], obj_bbox)
        break
