#!/usr/bin/env python

# Gibbs sampling algorithm to denoise an image
# Author : Gunaa AV, Isaac Caswell
# Edits : Bo Wang, Kratarth Goel, Aditya Grover
# Date : 2/17/2017

import sys
import cProfile
import itertools
#import progressbar

import math
import copy
import random
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

import scipy.special

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def markov_blanket(i,j,Y,X):
    '''
    return:
        the a list of Y values that are markov blanket of Y[i][j]
        e.g. if i = j = 1,
            the function should return [Y[0][1], Y[1][0], Y[1][2], Y[2][1], X[1][1]]
    '''
    # Get x at same position, get all neighbors of yij in Y
    return (X[i][j], Y[i-1][j], Y[i+1][j], Y[i][j-1], Y[i][j+1])

def sampling_prob(markov_blanket):
    '''
    markov_blanket: a list of the values of a variable's Markov blanket
        The order doesn't matter (see part (a)). e.g. [1,1,-1,1]
    return:
         a real value which is the probability of a variable being 1 given its Markov blanket
    '''
    # Equation for P reduces to sum of values in ym*2
    return sigmoid(2.0*sum(markov_blanket))

def sample(i, j, Y, X, DUMB_SAMPLE = 0):
    '''
    return a new sampled value of Y[i][j]
    It should be sampled by
        (i) the probability condition on all the other variables if DUMB_SAMPLE = 0
        (ii) the consensus of Markov blanket if DUMB_SAMPLE = 1
    '''
    #blanket = markov_blanket(i,j,Y,X)

    if not DUMB_SAMPLE:
        #prob = sampling_prob(blanket)
        # Condensed significantly for speedup
        prob = 1.0 / (1.0 + math.exp(-(2.0*sum((X[i][j], Y[i-1][j], Y[i+1][j], Y[i][j-1], Y[i][j+1])))))
        if random.random() < prob:
            return 1
        else:
            return -1
    else:
        c_b = Counter((X[i][j], Y[i-1][j], Y[i+1][j], Y[i][j-1], Y[i][j+1]))
        if c_b[1] >= c_b[-1]:
            return 1
        else:
            return -1

def compute_energy(Y, X):
    # X term is reduced to dot product of X and Y
    x_term = np.dot(X.ravel(),Y.ravel())
    # Sum over all product to the right of and below current
    # to avoid double counting edges
    y_term = 0.0
    for i in xrange(1, Y.shape[0]-1):
        for j in xrange(1, Y.shape[1]-1):
            y_term += Y[i][j]*Y[i+1][j]
            y_term += Y[i][j]*Y[i][j+1]
    # Energy is negative sum of these terms
    return -(x_term + y_term)

def get_posterior_by_sampling(filename, initialization = 'same', 
    logfile = None, DUMB_SAMPLE = 0, 
    MAX_BURNS=100, MAX_SAMPLES=1000, MAX_DUMB=30):
    '''
    Do Gibbs sampling and compute the energy of each assignment for the image specified in filename.
    If not dumb_sample, it should run MAX_BURNS iterations of burn in and then
    MAX_SAMPLES iterations for collecting samples.
    If dumb_sample, run MAX_SAMPLES iterations and returns the final image.

    filename: file name of image in txt
    initialization: 'same' or 'neg' or 'rand'
    logfile: the file name that stores the energy log (will use for plotting later)
        look at the explanation of plot_energy to see detail
    DUMB_SAMPLE: equals 1 if we want to use the trivial reconstruction in part (d)

    return value: posterior, Y, frequencyZ
        posterior: an 2d-array with the same size of Y, the value of each entry should
            be the probability of that being 1 (estimated by the Gibbs sampler)
        Y: The final image (for DUMB_SAMPLE = 1, in part (d))
        frequencyZ: a dictionary with key: count the number of 1's in the Z region
                                      value: frequency of such count
    '''
    # Read the image file in and store to X
    X = np.array(read_txt_file(filename))

    # Get dimenions of X
    r, c = X.shape

    # Initialize Y
    Y = None
    if initialization == 'same':
        Y = np.array(X)
    elif initialization == 'neg': 
        Y = np.array(-X)
    elif initialization == 'rand': 
        Y = np.random.choice([-1.0, 1.0], size=(r-2, c-2), 
            replace=True, p=[0.5, 0.5])
        Y = np.pad(Y, ((1,1), (1,1)), 'constant', constant_values=0)
    else:
        raise Exception(
            "Invalid initialization mode for get_posterior_by_sampling")

    # Initialize posterior as None
    P = None

    # Initialize frequency of +1 in Z as empty dict
    f_Z = {}

    if DUMB_SAMPLE == 0:
        # List of energy for each iteration
        E_dict = {"burn": [], "good": []}

        # Sample burn in period
        for _ in xrange(MAX_BURNS):
            # Iterate through all valid i,j
            for i in xrange(1, r-1):
                for j in xrange(1, c-1):  
                    # Sample and update Y
                    Y[i][j] = sample(i, j, Y, X)
            # Compute energy of sample
            E_dict["burn"].append(compute_energy(Y, X))

        # Initialize count matrix for estimating posterior
        C = np.zeros_like(Y)

        # Sample from equilibrated distribution
        for _ in xrange(MAX_SAMPLES):
            # Iterate through all valid i,j
            for i in xrange(1, r-1):
                for j in xrange(1, c-1):  
                    # Sample and update Y
                    Y[i][j] = sample(i, j, Y, X)
            # Compute energy of sample
            E_dict["good"].append(compute_energy(Y, X))
            # Increment counts in posterior matrix
            C[Y==1] += 1
            # Estimate Z square
            Z = Y[125:162, 143:174]
            # Get number of +1 in Z square  
            c_Z = len(Z[Z==1])
            # Add 1 to value of that count in frequency dict
            if c_Z not in f_Z:
                f_Z[c_Z] = 0
            f_Z[c_Z] += 1

        # Estimate posterior by average count matrix
        P = C/MAX_SAMPLES

        # Write energy values to log file in specified format
        with open(logfile, 'wb') as fp:
            # Write burn energies
            for i, E in enumerate(E_dict["burn"], 1):
                fp.write("{}\t{}\t{}\n".format(i, E, 'B'))
            # Write good energies
            for i, E in enumerate(E_dict["good"], MAX_BURNS+1):
                fp.write("{}\t{}\t{}\n".format(i, E, 'S'))
    elif DUMB_SAMPLE == 1:
        for _ in range(MAX_DUMB):
            # Iterate through all valid i,j
            for i in xrange(1, r-1):
                for j in xrange(1, c-1):  
                    # Sample and update Y
                    Y[i][j] = sample(i, j, Y, X, DUMB_SAMPLE=1)

    # Return Y after sampling is done
    return P, Y, f_Z

def denoise_image(filename, initialization = 'rand', logfile=None, DUMB_SAMPLE = 0):
    '''
    Do Gibbs sampling on the image and return the denoised one and frequencyZ
    '''
    posterior, Y, frequencyZ = get_posterior_by_sampling(
        filename, initialization, logfile=logfile, 
        DUMB_SAMPLE = DUMB_SAMPLE
    )

    if DUMB_SAMPLE:
        for i in xrange(len(Y)):
            for j in xrange(len(Y[0])):
                Y[i][j] = .5*(1.0-Y[i][j]) # 1, -1 --> 1, 0
        return Y, frequencyZ
    else:
        denoised = np.zeros(posterior.shape)
        denoised[np.where(posterior<.5)] = 1
        return denoised, frequencyZ


# ===========================================
# Helper functions for plotting etc
# ===========================================

def plot_energy(filename):
    '''
    filename: a file with energy log, each row should have three terms separated by a \t:
        iteration: iteration number
        energy: the energy at this iteration
        S or B: indicates whether it's burning in or a sample
    e.g.
        1   -202086.0   B
        2   -210446.0   S
        ...
    '''
    its_burn, energies_burn = [], []
    its_sample, energies_sample = [], []
    with open(filename, 'r') as f:
        for line in f:
            it, en, phase = line.strip().split()
            if phase == 'B':
                its_burn.append(it)
                energies_burn.append(en)
            elif phase == 'S':
                its_sample.append(it)
                energies_sample.append(en)
            else:
                print "bad phase: -%s-"%phase

    p1, = plt.plot(its_burn, energies_burn, 'r')
    p2, = plt.plot(its_sample, energies_sample, 'b')
    plt.title(filename)
    plt.xlabel("Iterations")
    plt.ylabel("Energy")
    plt.legend([p1, p2], ["burn in", "sampling"])
    plt.savefig(filename)
    plt.close()


def read_txt_file(filename):
    '''
    filename: image filename in txt
    return:   2-d array image
    '''
    f = open(filename, "r")
    lines = f.readlines()
    height = int(lines[0].split()[1].split("=")[1])
    width = int(lines[0].split()[2].split("=")[1])
    Y = [[0]*(width+2) for i in range(height+2)]
    for line in lines[2:]:
        i,j,val = [int(entry) for entry in line.split()]
        Y[i+1][j+1] = val
    return Y


def convert_to_png(denoised_image, title):
    '''
    save array as a png figure with given title.
    '''
    plt.imshow(denoised_image, cmap=plt.cm.gray)
    plt.title(title)
    plt.savefig(title + '.png')


def get_error(img_a, img_b):
    '''
    compute the fraction of all pixels that differ between the two input images.
    '''
    N = len(img_b[0])*len(img_b)*1.0
    return sum([sum([1 if img_a[row][col] != img_b[row][col] else 0 for col in           range(len(img_a[0]))])
	 for row in range(len(img_a))]
	 ) /N


fig_num = 0

def plot_part_d(orig, noisy, denoised, title):
    global fig_num
    fig_num += 1
    plt.figure(fig_num, figsize=(9,3), dpi=200)
    plt.suptitle("{}% plots".format(title), fontsize=14)

    plt.subplot(131)
    plt.imshow(noisy, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('Noisy Image', fontsize=12)

    plt.subplot(132)
    plt.imshow(denoised, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('Denoised Image', fontsize=12)

    plt.subplot(133)
    plt.imshow(orig, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title('Original Image', fontsize=12)

    plt.savefig(title + '.png')

#==================================
# doing part (c), (d), (e), (f)
#==================================

def perform_part_c():
    '''
    Run denoise_image function with different initialization and plot out the energy functions.
    '''

    image_file = "noisy_20.txt"
    log_same_file = "log_same"
    log_neg_file = "log_neg"
    log_rand_file = "log_rand"

    #cProfile.run('denoise_image("noisy_20.txt", initialization = "same", logfile="log_c.txt", DUMB_SAMPLE = 0)')
    denoise_image(image_file, initialization = "same", 
        logfile=log_same_file, DUMB_SAMPLE = 0)
    denoise_image(image_file, initialization = "neg", 
        logfile=log_neg_file, DUMB_SAMPLE = 0)
    denoise_image(image_file, initialization = "rand", 
        logfile=log_rand_file, DUMB_SAMPLE = 0)

    #### plot out the energy functions
    plot_energy(log_rand_file)
    plot_energy(log_neg_file)
    plot_energy(log_same_file)

def perform_part_d():
    '''
    Run denoise_image function with different noise levels of 10% and 20%, and report the errors between denoised images and original image
    '''

    # Load the original image and convert to {0, 1}
    orig_img = np.array(read_txt_file("orig.txt"))
    orig_img[orig_img==1] = 0
    orig_img[orig_img==-1] = 1

    # Load noisy image (10)
    noisy_10 = np.array(read_txt_file("noisy_10.txt"))
    noisy_10[noisy_10==1] = 0
    noisy_10[noisy_10==-1] = 1

    # Load noisy image (20)
    noisy_20 = np.array(read_txt_file("noisy_20.txt"))
    noisy_20[noisy_20==1] = 0
    noisy_20[noisy_20==-1] = 1

    # Denoise noisy image (10)
    denoised_10, frequencyZ = denoise_image("noisy_10.txt", 
        initialization = "rand", 
        logfile="tmp", DUMB_SAMPLE = 0)

    # Denoise noisy image (20)
    denoised_20, frequencyZ = denoise_image("noisy_20.txt", 
        initialization = "rand", 
        logfile="tmp", DUMB_SAMPLE = 0)

    # Remove padding from all images
    orig_img = orig_img[1:-1, 1:-1]
    noisy_10 = noisy_10[1:-1, 1:-1]
    noisy_20 = noisy_20[1:-1, 1:-1]
    denoised_10 = denoised_10[1:-1, 1:-1]
    denoised_20 = denoised_20[1:-1, 1:-1]

    # Compute error between original and denoised images
    d10_err = get_error(orig_img, denoised_10)
    print "Denoised 10 error: {}".format(d10_err) 

    # Compute error between original and denoised images
    d20_err = get_error(orig_img, denoised_20)
    print "Denoised 20 error: {}".format(d20_err) 

    # PLot orig, noisy, denoised for 10%
    plot_part_d(orig_img, noisy_10, denoised_10, "denoised_10")

    # PLot orig, noisy, denoised for 20%
    plot_part_d(orig_img, noisy_20, denoised_20, "denoised_20")


def perform_part_e():
    '''
    Run denoise_image function using dumb sampling with different noise levels of 10% and 20%.
    '''
    # Load the original image and convert to {0, 1}
    orig_img = np.array(read_txt_file("orig.txt"))
    orig_img[orig_img==1] = 0
    orig_img[orig_img==-1] = 1

    # Load noisy image (10)
    noisy_10 = np.array(read_txt_file("noisy_10.txt"))
    noisy_10[noisy_10==1] = 0
    noisy_10[noisy_10==-1] = 1

    # Load noisy image (20)
    noisy_20 = np.array(read_txt_file("noisy_20.txt"))
    noisy_20[noisy_20==1] = 0
    noisy_20[noisy_20==-1] = 1

    # Denoise noisy image (10)
    denoised_10, frequencyZ = denoise_image("noisy_10.txt", 
        initialization = "rand", 
        logfile="tmp", DUMB_SAMPLE = 1)

    # Denoise noisy image (20)
    denoised_20, frequencyZ = denoise_image("noisy_20.txt", 
        initialization = "rand", 
        logfile="tmp", DUMB_SAMPLE = 1)

    # Remove padding from all images
    orig_img = orig_img[1:-1, 1:-1]
    noisy_10 = noisy_10[1:-1, 1:-1]
    noisy_20 = noisy_20[1:-1, 1:-1]
    denoised_10 = denoised_10[1:-1, 1:-1]
    denoised_20 = denoised_20[1:-1, 1:-1]

    # Compute error between original and denoised images
    d10_err = get_error(orig_img, denoised_10)
    print "Denoised dumb 10 error: {}".format(d10_err) 

    # Compute error between original and denoised images
    d20_err = get_error(orig_img, denoised_20)
    print "Denoised dumb 20 error: {}".format(d20_err) 

    # PLot orig, noisy, denoised for 10%
    plot_part_d(orig_img, noisy_10, denoised_10, "denoised_dumb_10")

    # PLot orig, noisy, denoised for 20%
    plot_part_d(orig_img, noisy_20, denoised_20, "denoised_dumb_20")

def perform_part_f():
    '''
    Run Z square analysis
    '''
    global fig_num

    width = 1.0

    d, f = denoise_image('noisy_10.txt', initialization = 'same', 
        logfile = 'tmp')
    fig_num += 1
    plt.figure(fig_num)
    plt.clf()
    plt.bar(f.keys(), f.values(), width, color = 'b')
    plt.title("Frequency Counts of Z Region (10% noise)")
    plt.xlabel("Frequency of +1")
    plt.ylabel("Count of Frequency")
    plt.savefig("noisy_10_zfreq")

    d, f = denoise_image('noisy_20.txt', initialization = 'same', 
        logfile = 'tmp')
    fig_num += 1
    plt.figure(fig_num)
    plt.clf()
    plt.bar(f.keys(), f.values(), width, color = 'b')
    plt.title("Frequency Counts of Z Region (20% noise)")
    plt.xlabel("Frequency of +1")
    plt.ylabel("Count of Frequency")
    plt.savefig("noisy_20_zfreq")

if __name__ == "__main__":
    perform_part_c()
    perform_part_d()
    perform_part_e()
    perform_part_f()
