#!/usr/bin/env python3

# CS231A Homework 0, Problem 4
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


def main():
    # ===== Problem 4a =====
    # Read in image1 as a grayscale image. Take the singular value
    # decomposition of the image.

    img1 = None

    # BEGIN YOUR CODE HERE
    img1 = misc.imread('image1.jpg', flatten=True)
    u, s, v = np.linalg.svd(img1, full_matrices=False)
    # END YOUR CODE HERE

    # ===== Problem 4b =====
    # Save and display the best rank 1 approximation 
    # of the (grayscale) image1.

    rank1approx = None

    # BEGIN YOUR CODE HERE
    rank = 1
    rank1approx = np.zeros((u.shape[0], v.shape[0]))
    for i in range(rank):
        rank1approx += np.outer(u.T[i]*s[i], v[i])
    plt.axis('off')
    plt.imshow(rank1approx, cmap='gray')
    plt.title('Rank 1 Approximation of Image 1')
    plt.savefig('p4b.png')
    # END YOUR CODE HERE

    # ===== Problem 4c =====
    # Save and display the best rank 20 approximation
    # of the (grayscale) image1.

    rank20approx = None

    # BEGIN YOUR CODE HERE
    rank = 20
    rank20approx = np.zeros((u.shape[0], v.shape[0]))
    for i in range(rank):
        rank20approx += np.outer(u.T[i]*s[i], v[i])
    plt.axis('off')
    plt.imshow(rank20approx, cmap='gray')
    plt.title('Rank 20 Approximation of Image 1')
    plt.savefig('p4c.png')
    # END YOUR CODE HERE


if __name__ == '__main__':
    main()
