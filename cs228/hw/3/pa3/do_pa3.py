#!/usr/bin/env python

###############################################################################
# Finishes PA 3
# author: Billy Jun, Xiaocheng Li
# date: Jan 31, 2016
###############################################################################

## Utility code for PA3
import sys
import itertools

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from cluster_graph import *
from factors import *

from pprint import pprint

# Set the random seed for reproducible experiment
np.random.seed(1)

def loadLDPC(name):
    """
    :param - name: the name of the file containing LDPC matrices
  
    return values:
    G: generator matrix
    H: parity check matrix
    """
    A = sio.loadmat(name)
    G = A['G']
    H = A['H']
    return G, H

def loadImage(fname, iname):
    '''
    :param - fname: the file name containing the image
    :param - iname: the name of the image
    (We will provide the code using this function, so you don't need to worry too much about it)  
  
    return: image data in matrix form
    '''
    img = sio.loadmat(fname)
    return img[iname]


def applyChannelNoise(y, p):
    '''
    :param y - codeword with 2N entries
    :param p channel noise probability
  
    return corrupt message yhat  
    yhat_i is obtained by flipping y_i with probability p 
    '''
    ###############################################################################
    # TODO: Your code here!
    yhat = np.zeros_like(y)
    for i,b in enumerate(y):
        flip = np.random.choice([1, 0], p=[p, 1.0-p])
        if flip:
            yhat[i] = 1-b
        else:
            yhat[i] = b
    
    ###############################################################################
    return yhat


def encodeMessage(x, G):
    '''
    :param - x orginal message
    :param[in] G generator matrix
    :return codeword y=Gx mod 2
    '''
    return np.mod(np.dot(G, x), 2)


def constructClusterGraph(yhat, H, p):
    '''
    :param - yhat: observed codeword
    :param - H parity check matrix
    :param - p channel noise probability

    return G clusterGraph
   
    You should consider two kinds of factors:
    - M unary factors 
    - N each parity check factors
    '''
    N = H.shape[0]
    M = H.shape[1]
    G = ClusterGraph(M)
    domain = [0, 1]
    G.nbr = [[] for _ in range(M+N)]
    G.sepset = [[None for _ in range(M+N)] for _ in range(M+N)]
    ##############################################################
    
    # Set the variables of G
    G.var = range(M)

    # Set unary factors
    for i in range(M):
        val = None
        if yhat[i] == 1:
            val = np.array([p, 1-p])
        else:
            val = np.array([1-p, p])

        f = Factor(scope=[G.var[i]], card=[len(domain)], 
                val=val, 
                name="unary_{}".format(i))
        G.factor.append(f)

    # Set parity check factors
    for i in range(N):
        scope = (np.arange(M)[H[i]==1])
        card = [len(domain) for _ in range(len(scope))]
        poss = list(itertools.product(domain, repeat=len(scope)))
        val = np.zeros(card)
        for p in poss:
            val[p] = 1-np.count_nonzero(p)%2    
        f = Factor(scope=scope, card=card, 
            val=val, name="parity_{}".format(i))
        G.factor.append(f)

    # Initialize sepset of factors
    for i in range(M+N):
        si = set(G.factor[i].scope)
        for j in range(M+N):
            sj = set(G.factor[j].scope)
            # Parity factors do not share variables between them
            if (len(si) == 1 and len(sj) > 1) or (len(si) > 1 and len(sj) == 1):
                shared = list(si.intersection(sj))
                # If list of shared vars is empty, don't set
                if len(shared) > 0:
                    G.sepset[i][j] = shared


    # Initalize neighbors of factors using sepset
    num_nbr = 0
    for i in range(M+N):
        for j in range(M+N):
            if G.sepset[i][j] is not None:
                G.nbr[i].append(j)
                num_nbr += 1

    # Iterate over the sepset to form messages       
    for i in range(M+N):
        for j in range(M+N):
            sepset = G.sepset[i][j]
            if sepset is not None:
                card = [len(domain) for _ in range(len(sepset))]
                message = Factor(scope=sepset, card=card,
                    val=np.ones(card), name="msg_{}_{}".format(i, j))
                G.messages[(i, j)] = message.normalize()

    ##############################################################
    return G

def do_part_a():
    yhat = np.array([[1, 1, 1, 1, 1]]).reshape(5,1)
    H = np.array([ \
        [0, 1, 1, 0, 1], \
        [0, 1, 0, 1, 1], \
        [1, 1, 0, 1, 0], \
        [1, 0, 1, 1, 0], \
        [1, 0, 1, 0, 1]])
    p = 0.95
    G = constructClusterGraph(yhat, H, p)
    ##############################################################
    # To do: your code starts here 
    # Design two invalid codewords ytest1, ytest2 and one valid codewords ytest3.
    # Report their weights respectively.
    ytest1 = [1, 0, 0, 0, 0]
    ytest2 = [0, 1, 0, 0, 0]
    ytest3 = [0, 0, 0, 0, 0]

    ##############################################################
    print(
        G.evaluateWeight(ytest1), \
        G.evaluateWeight(ytest2), \
        G.evaluateWeight(ytest3))

def run_loopy_bp(y, H, p, iterations, checkpoint=None):
    # Encode noise into message
    yhat = applyChannelNoise(y, p)

    # Show number of mistakes in original
    h_start = len(y) - np.count_nonzero(y.ravel()==yhat.ravel())
    print "Iter {}: Hamming={}".format(0, h_start)

    # Create the cluster graph (clique tree)
    CG = constructClusterGraph(yhat, H, p)

    # Perform loopy BP on the cluster graph
    pp = CG.runParallelLoopyBP(y, iterations, checkpoint=checkpoint)

    # Return posterior probability array
    return pp, h_start

def prep_loopy_bp():
    G, H = loadLDPC('ldpc36-128.mat')
    N = G.shape[1]
    x = np.zeros((N, 1), dtype='int32')
    #x = np.random.binomial(1, 0.5, size=N).astype('int32')
    y = encodeMessage(x, G)

    return y, H

def do_part_c(iterations=50, p=0.05):
    '''
    In part b, we provide you an all-zero initialization of message x, you should
    apply noise on y to get yhat, znd then do loopy BP to obatin the
    marginal probabilities of the unobserved y_i's.
    '''
    # Setup and run loopy bp, get prob arr
    y, H = prep_loopy_bp()
    pp, _ = run_loopy_bp(y, H, p, iterations)

    # Plot estimated posterior probability that each codeword == 1
    p1 = zip(*pp)[1]

    # Make bar plot
    plt.figure(figsize=(7,5), dpi=150)
    plt.bar(range(len(p1)), p1, color="red") 
    plt.title("Estimated P(bit=1) for Codeword Bits")
    plt.xlabel("bit #")
    plt.ylabel("P(bit=1)")
    plt.axis([0, len(p1), 0.0, np.max(p1)*2])
    plt.savefig('p5c')

    ##############################################################

def do_part_de(numTrials, error, iterations=0, part=None):
    '''
    param - numTrials: how many trials we repreat the experiments
    param - error: the transmission error probability
    param - iterations: number of Loopy BP iterations we run for each trial
    '''
    ##############################################################
    # To do: your code starts here

    # Set up plot
    plt.figure(figsize=(7,5), dpi=150)

    # Store all h_list
    h_mat = []

    # Setup and run loopy bp, get prob arr
    y, H = prep_loopy_bp()
    for i in range(numTrials):
        # Set up checkpoint array
        checkpoint = range(1, iterations+1)

        # Run loopy bp for some random noise channel
        pp_dict, h_start = run_loopy_bp(y, H, error, iterations, 
            checkpoint=checkpoint)

        # Set up list of hamming distances
        h_list = [h_start]
        checkpoint.insert(0, 0)

        # Calculate hamming distance for each iteration of trial
        for pp in pp_dict.itervalues():
            estimate = pp.argmax(axis=1)
            h = len(y) - np.count_nonzero(y.ravel()==estimate)
            h_list.append(h)

        # Store h_list in h_mat
        h_mat.append(h_list)
        
        # Plot hamming distance vs. iterations
        plt.plot(checkpoint, h_list)


    # Make bar plot
    plt.axis([0, iterations, 0, 
        max(40, max([max(h_list) for h_list in h_mat])+5)])
    plt.title("Hamming Distance v. Iteration: Error={:2.2f}".format(error))
    plt.xlabel("Iteration")
    plt.ylabel("Hamming Distance")
    plt.savefig('p5{}'.format(part))

    ##############################################################

def do_part_fg(p, part=None):
    '''
    param - error: the transmission error probability
    '''
    G, H = loadLDPC('ldpc36-1600.mat')
    img = loadImage('images.mat', 'cs242')

    # Set up sublot figure
    plt.figure(1, figsize=(4,8), dpi=150)
    plt.suptitle("Decoded Output: Error={:2.2f}".format(p), fontsize=18)

    # Plot the original image
    #plt.subplot(421)
    #plt.axis('off')
    #plt.imshow(img)

    img_shape = img.shape
    x = img.reshape(np.prod(img_shape))

    N = G.shape[1]
    y = encodeMessage(x, G)

    # Apply noise to the message
    yhat = applyChannelNoise(y, p)

    # Plot the image with noise
    recovered = yhat[:x.shape[0]].reshape(img_shape)
    plt.subplot(421)
    plt.axis('off')
    plt.title("Iter 0")
    plt.imshow(recovered)

    # Show number of mistakes in original
    print "Iter {}: Hamming={}".format(0,
        len(y) - np.count_nonzero(y.ravel()==yhat.ravel())
    )

    # Create the cluster graph (clique tree)
    CG = constructClusterGraph(yhat, H, p)

    # Perform loopy BP on the cluster graph
    checkpoint = [1, 2, 3, 5, 10, 20, 30]
    pp_dict = CG.runParallelLoopyBP(y, max(checkpoint), checkpoint=checkpoint)

    # Iterate through all checkpointed prob arrays
    for i,(iter,pp) in enumerate(pp_dict.iteritems()):
        # Take the most likely value for each pixel
        argmax = np.argmax(pp, axis=1)
        # Reform the original image
        recovered = argmax[:x.shape[0]].reshape(img_shape)
        # Plot the image
        plt.subplot(int("42{}".format((i+2))))
        plt.axis('off')
        plt.title("Iter {}".format(iter))
        plt.imshow(recovered)

    # Save and show the plot
    plt.savefig('p5{}'.format(part))

print('Doing part (a): Should see 0.0, 0.0, >0.0')
do_part_a()
print('Doing part (c)')
#do_part_c()
print('Doing part (d)')
#do_part_de(10, 0.06, iterations=50, part="de1")
print('Doing part (e)')
#do_part_de(10, 0.08, iterations=50, part="de2")
#do_part_de(10, 0.10, iterations=50, part="de3")
print('Doing part (f)')
#do_part_fg(0.06, part='f')
print('Doing part (g)')
#do_part_fg(0.10, part='g')
