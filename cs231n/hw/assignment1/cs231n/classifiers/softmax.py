import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    dW = np.zeros_like(W)

    #############################################################################
    # Store the loss in loss and the gradient in dW. If you are not careful         #
    # here, it is easy to run into numeric instability. Don't forget the                #
    # regularization!                                                                                                                       #
    #############################################################################
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        # Trick to help with numerical instability
        scores -= np.max(scores)
        # Calculate exp of scores, to be reused
        exp_scores = np.exp(scores)
        # Calculate sum of exp scores using explicit loop
        sum_term = 0.0
        for j in xrange(num_classes):
            sum_term += exp_scores[j]
        # Calculate loss and dW using explicity loop
        for j in xrange(num_classes):
            p = exp_scores[j]/sum_term
            dW[:,j] += p*X[i]
            if y[i] == j: 
                loss += -np.log(p)
                dW[:,j] -= X[i]

    # Normalize data loss by num_train
    loss /= num_train

    # Normalize dW by number of training elements
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    # Add regularization component to gradient
    dW += 2.0*reg*W

    #############################################################################
    #                                                    END OF YOUR CODE                                                                   #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful         #
    # here, it is easy to run into numeric instability. Don't forget the                #
    # regularization!                                                                                                                       #
    #############################################################################
    # Dot product Wx to get scores, take exp now
    all_scores = X.dot(W)
    # Subtract max to avoid numeric instability
    all_scores = (all_scores.T - np.max(all_scores, axis=1)).T
    # Take exponent of all scores
    all_scores = np.exp(all_scores)
    # Index correct class scores
    correct_class_scores = all_scores[np.arange(0, num_train), y]
    # Sum scores
    score_sum = np.sum(all_scores, axis=1)
    # Calculate contribution to loss
    p = correct_class_scores/score_sum
    L = -np.log(p)
    loss = np.sum(L)/num_train
    # Calculate contribution to dW
    p_j = all_scores.T/score_sum
    dW = np.dot(X.T, p_j.T)
    mask = np.zeros((num_train, num_classes))
    mask[np.arange(0, num_train), y] = -1
    dW += np.dot(X.T, mask)
    
    # Normalize dW by number of training elements
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    # Add regularization component to gradient
    dW += 2.0*reg*W
    #############################################################################
    #                                                    END OF YOUR CODE                                                                   #
    #############################################################################

    return loss, dW

