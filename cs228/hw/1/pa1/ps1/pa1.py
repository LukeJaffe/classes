#!/usr/bin/env python

"""
CS 228: Probabilistic Graphical Models
Winter 2017
Programming Assignment 1: Bayesian Networks

Author: Aditya Grover
"""
import sys
from pprint import pprint

import numpy as np 
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.io import loadmat

def plot_histogram(data, title='histogram', xlabel='value', ylabel='frequency', savefile='hist'):
    '''
    Plots a histogram.
    '''

    plt.figure()
    plt.hist(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(savefile, bbox_inches='tight')
    plt.show()
    plt.close()

    return

def get_p_z1(z1_val):
    '''
    Computes the prior probability for variable z1 to take value z1_val.
    '''

    return bayes_net['prior_z1'][z1_val]

def get_p_z2(z2_val):
    '''
    Computes the prior probability for variable z2 to take value z2_val.
    '''

    return bayes_net['prior_z2'][z2_val]

def get_p_cl(z1_val, z2_val):
    '''
    Get cond likelihood probabilities for z1_val, z2_val
    '''

    return bayes_net["cond_likelihood"][(z1_val, z2_val)].squeeze()

def get_p_xk_cond_z1_z2(z1_val, z2_val, k):
    '''
    Computes the conditional likelihood of variable xk assumes value 1 
    given z1 assumes value z1_val and z2 assumes value z2_val. 
    '''

    return get_p_cl(z1_val, z2_val)[k]

def get_sample_z1():
    # Get z1 prior from net
    prior_z1 = bayes_net['prior_z1']
    # Take random sample using probability table
    z1_val = np.random.choice(prior_z1.keys(), p=prior_z1.values())
    return z1_val

def get_sample_z2():
    # Get z2 prior from net
    prior_z2 = bayes_net['prior_z2']
    # Take random sample using probability table
    z2_val = np.random.choice(prior_z2.keys(), p=prior_z2.values())
    return z2_val

def get_empty_sample():
    # Initialize empty sample
    sample = np.zeros_like(bayes_net["cond_likelihood"].values()[0]).squeeze()
    return sample

def get_p_x_cond_z1_z2(z1_val, z2_val):
    # Initialize empty sample
    sample = get_empty_sample()

    # Populate array with [0,1] randomly sampled using cond likelihood probs
    for i in range(sample.shape[0]):
        cl = get_p_xk_cond_z1_z2(z1_val, z2_val, i)
        sample[i] = np.random.choice([1,0], p=[cl, 1.0-cl])

    return sample

def get_pixels_sampled_from_p_x_joint_z1_z2():
    '''
    This function returns the sampled values of pixel variables.
    '''
    # Randomly sample z1, z2 from given probability dicts
    z1_val = get_sample_z1()
    z2_val = get_sample_z2()

    # Randomly sample from cond_likelihood using these vals
    return get_p_x_cond_z1_z2(z1_val, z2_val)

def get_conditional_expectation(data):
    def calc_means(X): 
        # Convert input image to binary mask
        mask1 = X.astype(bool)
        # Make a mask for X=0 for readability
        mask0 = ~mask1
        # Initialize pair to store z_sum
        z_sum = np.zeros(2)
        # Initialize variable to store P(X)
        norm_sum = 0.0
        # Find all values of joint prob before the sum and log, for log-sum-exp trick
        for i, z1_val in enumerate(disc_z1):
            for j, z2_val in enumerate(disc_z2):
                # Combine z vals in array
                z_vals = np.array([z1_val, z2_val])
                # Get terms from net
                cl = get_p_cl(z1_val, z2_val)
                z1 = get_p_z1(z1_val)
                z2 = get_p_z2(z2_val)
                ### Calculate array of probabilities for x1:784, P(X=1) or P(X=0)
                # Set P(X) from cond_likelihood
                px = np.zeros_like(X)
                px[mask1] = cl[mask1]
                px[mask0] = 1.0-cl[mask0]
                # Calculate and store joint probability
                jp = z1*z2*px.prod()
                # Add z terms to sum
                z_sum += z_vals*jp
                # Add jp to norm term sum
                norm_sum += jp
        # Normalize z_vals
        z_mean = z_sum/norm_sum
        # Return log jp result
        return z_mean

    # Initialize arrays to store z means
    mean_z1 = np.zeros(data.shape[0])
    mean_z2 = np.zeros(data.shape[0])

    # Calculate z means for each image
    for k,v in enumerate(data):
        print "{}/{}".format(k+1, data.shape[0])
        mean = calc_means(v)
        mean_z1[k], mean_z2[k] = mean

    return mean_z1, mean_z2

def q4():
    '''
    Plots the pixel variables sampled from the joint distribution as 28 x 28 images.
    '''
    
    plt.figure()
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(get_pixels_sampled_from_p_x_joint_z1_z2().reshape(28, 28), cmap='gray')
        plt.title('Sample: ' + str(i+1))
    plt.tight_layout()
    plt.savefig('a4', bbox_inches='tight')
    plt.show()
    plt.close()

    return

def q5():
    '''
    Plots the expected images for each latent configuration on a 2D grid.
    '''
    canvas = np.empty((28*len(disc_z1), 28*len(disc_z2)))
    for i, z1_val in enumerate(disc_z1):
        for j, z2_val in enumerate(disc_z2):
            canvas[(len(disc_z1)-i-1)*28:(len(disc_z2)-i)*28, j*28:(j+1)*28] = \
            get_p_x_cond_z1_z2(z1_val, z2_val).reshape(28, 28)

    plt.figure()        
    plt.imshow(canvas, cmap='gray')
    plt.tight_layout()
    plt.savefig('a5', bbox_inches='tight')
    plt.show()
    plt.close()

    return

def q6():
    '''
    Loads the data and plots the histograms. Rest is TODO.
    '''

    # Load data
    mat = loadmat('q6.mat')
    val_data = mat['val_x']
    test_data = mat['test_x']

    # Calculate the max log-likelihood using the log-sum-exp trick
    def calc_mll_lse(X): 
        # Convert input image to binary mask
        mask1 = X.astype(bool)
        # Make a mask for X=0 for readability
        mask0 = ~mask1
        # Initialize array to store result of prob for each manifest var combo
        log_joint_arr = np.zeros((len(disc_z1), len(disc_z2)))
        # Find all values of joint prob before the sum and log, for log-sum-exp trick
        for i, z1_val in enumerate(disc_z1):
            for j, z2_val in enumerate(disc_z2):
                cl = get_p_cl(z1_val, z2_val)
                z1 = get_p_z1(z1_val)
                z2 = get_p_z2(z2_val)
                ### Calculate array of probabilities for x1:784, P(X=1) or P(X=0)
                # Set P(X) from cond_likelihood
                px = np.zeros_like(X)
                px[mask1] = cl[mask1]
                px[mask0] = 1.0-cl[mask0]
                # Calculate and store log joint prob
                log_jp = np.log(z1)+np.log(z2)+np.sum(np.log(px))
                log_joint_arr[i][j] = log_jp
        ### Log-sum-exp trick
        # log(sum(p)) = log(sum(e^ln(p))) = a + log(sum(e^(ln(p)-a)))
        # = a + log(sum(e^(ln(p0)+ln(p1)+...+ln(pn)-a)))
        # Set a to max of log joint probabilities
        a = np.max(log_joint_arr)
        # Calculate log-sum-exp
        lse = a + np.log(np.sum(np.e**(log_joint_arr-a)))
        # Return log-sum-exp result
        return lse

    # Compute marginal log-likelihood on validation set
    val_mll_arr = np.zeros(val_data.shape[0])
    for k,v in enumerate(val_data):
        print "{}/{}".format(k+1, val_data.shape[0])
        # Calculate mll for one validation image
        mll = calc_mll_lse(v)
        # Store log of likelihood
        val_mll_arr[k] = mll

    # Compute mean,std of marginal log-likelihood on validation set
    mll_avg = val_mll_arr.mean()
    mll_std = val_mll_arr.std()
    mll_lower_bound = mll_avg - 3*mll_std
    mll_upper_bound = mll_avg + 3*mll_std

    # Initialize binary mask for classification labels
    Y = np.zeros(test_data.shape[0]).astype(bool)
    # Initialize array to store mll in for histogram
    test_mll_arr = np.zeros(test_data.shape[0])
    # Classify images in test set using avg, std of mll
    for k,v in enumerate(test_data):
        print "{}/{}".format(k+1, len(test_data))
        # Calculate mll of test image
        mll = calc_mll_lse(v)
        test_mll_arr[k] = mll
        # Classify using the rule
        if mll_lower_bound <= mll <= mll_upper_bound:
            Y[k] = True
        else:
            Y[k] = False

    # Group the test results using the binary mask
    real_marginal_log_likelihood = test_mll_arr[Y]
    corrupt_marginal_log_likelihood = test_mll_arr[~Y]
            
    # Plot histograms for mll values of images classified as real and corrupted	
    plot_histogram(real_marginal_log_likelihood, 
        title='Histogram of marginal log-likelihood for real data',
        xlabel='marginal log-likelihood', savefile='a6_hist_real')

    plot_histogram(corrupt_marginal_log_likelihood, 
        title='Histogram of marginal log-likelihood for corrupted data',
        xlabel='marginal log-likelihood', savefile='a6_hist_corrupt')

    return

def q7():
    '''
    Loads the data and plots a color coded clustering of the conditional expectations.
    '''

    mat = loadmat('q7.mat')
    data = mat['x']
    labels = mat['y']

    print "Calculating conditional expectations:"
    mean_z1, mean_z2 = get_conditional_expectation(data)
    print "Finished calculating conditional expectations."

    plt.figure() 
    plt.scatter(mean_z1, mean_z2, c=labels)
    plt.colorbar()
    plt.grid()
    plt.savefig('a7', bbox_inches='tight')
    plt.show()
    plt.close()

    return

def load_model(model_file):
    '''
    Loads a default Bayesian network with latent variables (in this case, a variational autoencoder)
    '''

    with open(model_file, 'rb') as infile:
        cpts = pkl.load(infile)

    model = {}
    model['prior_z1'] = cpts[0]
    model['prior_z2'] = cpts[1]
    model['cond_likelihood'] = cpts[2]

    return model

def main():

    # Set up global variables
    global disc_z1, disc_z2
    n_disc_z = 25
    disc_z1 = np.linspace(-3, 3, n_disc_z)
    disc_z2 = np.linspace(-3, 3, n_disc_z)

    # Load the trained model into the bayes_net object for global use
    global bayes_net
    bayes_net = load_model('trained_mnist_model')

    # Run function for each assignment question
    q4()
    q5()
    q6()
    q7()

    return

if __name__== '__main__':

    main()
