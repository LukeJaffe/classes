import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                # Add margin to loss term
                loss += margin
                # Add margin derivative to gradient term
                dW[:,j] += X[i]
                dW[:,y[i]] -= X[i]

    # Normalize dW by number of training elements
    dW /= num_train

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    # Add regularization component to gradient
    dW += 2.0*reg*W

    #############################################################################
    # TODO:                                                                                                                                         #
    # Compute the gradient of the loss function and store it dW.                                #
    # Rather that first computing the loss and then computing the derivative,       #
    # it may be simpler to compute the derivative at the same time that the         #
    # loss is being computed. As a result you may need to modify some of the        #
    # code above to compute the gradient.                                                                               #
    #############################################################################


    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_train = X.shape[0]

    #############################################################################
    # Implement a vectorized version of the structured SVM loss, storing the        #
    # result in loss.                                                                                                                       #
    #############################################################################
    # Dot product Wx to get scores
    all_scores = X.dot(W)
    # Index correct class scores
    correct_class_scores = all_scores[np.arange(0, num_train), y]
    # Compute margin 
    margin = all_scores.T - correct_class_scores + 1
    # Get indeces where margin is <= 0, > 0
    margin_idx = margin <= 0
    # Binary margin mask
    margin_mask = np.ones_like(margin)
    margin_mask[margin_idx] = 0
    # Compute how many times margin > 0 for each data point
    margin_sum = np.sum(margin_mask, axis=0)
    # Subtract margin_sum from margin mask
    margin_mask.T[np.arange(0, num_train), y] -= margin_sum
    # Take dot product of margin mask and X to get dW, normalize by # data
    dW += X.T.dot(margin_mask.T)/num_train
    # Apply max(0, x) function to margin
    margin[margin_idx] = 0
    # Zero out margin where j == y_i
    margin.T[np.arange(0, num_train), y] = 0
    # Sum margin to get loss and normalize by # data
    loss = np.sum(margin)/num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    # Add regularization component to gradient
    dW += 2.0*reg*W
    #############################################################################
    #                                                           END OF YOUR CODE                                                            #
    #############################################################################


    #############################################################################
    # TODO:                                                                                                                                         #
    # Implement a vectorized version of the gradient for the structured SVM         #
    # loss, storing the result in dW.                                                                                       #
    #                                                                                                                                                       #
    # Hint: Instead of computing the gradient from scratch, it may be easier        #
    # to reuse some of the intermediate values that you used to compute the         #
    # loss.                                                                                                                                         #
    #############################################################################
    pass
    #############################################################################
    #                                                           END OF YOUR CODE                                                            #
    #############################################################################

    return loss, dW
