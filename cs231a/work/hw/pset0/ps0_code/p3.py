#!/usr/bin/env python3

# CS231A Homework 0, Problem 3
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


def main():
    # ===== Problem 3a =====
    # Read in the images, image1.jpg and image2.jpg, as color images.

    img1, img2 = None, None

    # BEGIN YOUR CODE HERE
    path1 = 'image1.jpg' 
    path2 = 'image2.jpg' 
    img1 = misc.imread(path1)
    img2 = misc.imread(path2)
    # END YOUR CODE HERE

    # ===== Problem 3b =====
    # Convert the images to double precision and rescale them
    # to stretch from minimum value 0 to maximum value 1.

    # BEGIN YOUR CODE HERE
    img1 = img1.astype(np.double)
    img1 -= img1.min()
    img1 /= img1.max()
    img2 = img2.astype(np.double)
    img2 -= img2.min()
    img2 /= img2.max()
    # END YOUR CODE HERE

    # ===== Problem 3c =====
    # Add the images together and re-normalize them 
    # to have minimum value 0 and maximum value 1. 
    # Display this image.

    # BEGIN YOUR CODE HERE
    img3 = img1 + img2
    img3 -= img3.min()
    img3 /= img3.max()
    misc.imsave('p3c.png', img3)
    # END YOUR CODE HERE

    # ===== Problem 3d =====
    # Create a new image such that the left half of 
    # the image is the left half of image1 and the 
    # right half of the image is the right half of image2.

    newImage1 = None

    # BEGIN YOUR CODE HERE
    newImage1 = np.concatenate([img1[:, :img1.shape[1]//2, :], img2[:, img2.shape[1]//2:, :]], axis=1)    
    misc.imsave('p3d.png', newImage1)
    # END YOUR CODE HERE

    # ===== Problem 3e =====
    # Using a for loop, create a new image such that every odd 
    # numbered row is the corresponding row from image1 and the 
    # every even row is the corresponding row from image2. 
    # Hint: Remember that indices start at 0 and not 1 in Python.

    newImage2 = None

    # BEGIN YOUR CODE HERE
    row_list = []
    for i in range(img1.shape[0]): 
        if i%2 == 0:
            row_list.append(img2[i, :, :][np.newaxis, :, :])
        else:
            row_list.append(img1[i, :, :][np.newaxis, :, :])
    newImage2 = np.concatenate(row_list, axis=0)
    misc.imsave('p3e.png', newImage2)
    # END YOUR CODE HERE

    # ===== Problem 3f =====
    # Accomplish the same task as part e without using a for-loop.
    # The functions reshape and repmat may be helpful here.

    newImage3 = None

    # BEGIN YOUR CODE HERE
    odd_mask_base = np.concatenate([
            np.zeros((1, img1.shape[1], img1.shape[2])),
            np.ones((1, img1.shape[1], img1.shape[2]))
        ], axis=0)
    odd_mask = np.tile(odd_mask_base, (img1.shape[0]//2, 1, 1))
    even_mask_base = np.concatenate([
            np.ones((1, img2.shape[1], img2.shape[2])),
            np.zeros((1, img2.shape[1], img2.shape[2]))
        ], axis=0)
    even_mask = np.tile(even_mask_base, (img2.shape[0]//2, 1, 1))
    newImage3 = odd_mask*img1 + even_mask*img2
    misc.imsave('p3f.png', newImage3)
    # END YOUR CODE HERE

    # ===== Problem 3g =====
    # Convert the result from part f to a grayscale image. 
    # Display the grayscale image with a title.

    # BEGIN YOUR CODE HERE
    grayImage3 = misc.imread('p3f.png', flatten=True)
    plt.axis('off')
    plt.imshow(grayImage3, cmap='gray')
    plt.title('Images Interleaved by Pixel Row (grayscale)')
    plt.savefig('p3g.png')
    # END YOUR CODE HERE


if __name__ == '__main__':
    main()
