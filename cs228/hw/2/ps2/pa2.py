#!/usr/bin/env python

import os
import sys
import numpy as np
from scipy.misc import logsumexp
from collections import Counter
import random

# helpers to load data
from data_helper import load_vote_data, load_incomplete_entry
# helpers to learn and traverse the tree over attributes
from tree import get_mst, get_tree_root, get_tree_edges

# pseudocounts for uniform dirichlet prior
alpha = 0.1

def renormalize(cnt):
    '''
    renormalize a Counter()
    '''
    tot = 1. * sum(cnt.values())
    for a_i in cnt:
        cnt[a_i] /= tot
    return cnt

def gen_missing(entries):
    new_entries = []
    try:
        for entry in entries:
            i = entry.index(-1)
            entry_a, entry_b = list(entry), list(entry)
            entry_a[i], entry_b[i] = 0, 1
            new_entries.extend([entry_a, entry_b])
        return gen_missing(new_entries)
    except:
        return entries

#--------------------------------------------------------------------------
# Naive bayes CPT and classifier
#--------------------------------------------------------------------------
class NBCPT(object):
    '''
    NB Conditional Probability Table (CPT) for a child attribute.  Each child
    has only the class variable as a parent
    '''

    def __init__(self, A_i):
        '''
        TODO create any persistent instance variables you need that hold the
        state of the learned parameters for this CPT
            - A_i: the index of the child variable
        '''
        self.A_i = A_i
        self.num_classes = 2
        self.num_vals = 2
        self.p = np.zeros((self.num_classes, self.num_vals))

    def learn(self, A, C, L=alpha):
        '''
        TODO populate any instance variables specified in __init__ to learn
        the parameters for this CPT
            - A: a 2-d numpy array where each row is a sample of assignments 
            - C: a 1-d n-element numpy where the elements correspond to the
              class labels of the rows in A
        '''
        for c in range(self.num_classes):
            for v in range(self.num_vals):
                Ac = A[C==c]
                self.p[c][v] = (np.count_nonzero(Ac.T[self.A_i]==v)+L)/float(len(Ac)+2*L)


    def get_cond_prob(self, entry, c):
        '''
        TODO return the conditional probability P(X|Pa(X)) for the values
        specified in the example entry and class label c
            - entry: full assignment of variables 
                e.g. entry = np.array([0,1,1]) means A_0 = 0, A_1 = 1, A_2 = 1
            - c: the class 
        '''
        return self.p[c][entry[self.A_i]]

class NBClassifier(object):
  '''
  NB classifier class specification
  '''

  def __init__(self, A_train, C_train):
    '''
    TODO create any persistent instance variables you need that hold the
    state of the trained classifier and populate them with a call to
    Suggestions for the attributes in the classifier:
        - P_c: the probabilities for the class variable C
        - cpts: a list of NBCPT objects
    '''
    # Initialize dimension parameters
    self.num_classes = 2
    self.num_data, self.num_features = A_train.shape

    # Initialize probabilities for each class
    num_C = float(C_train.shape[0])
    self.P_c = np.zeros(2)
    self.P_c[0] = np.count_nonzero(C_train==0)/num_C
    self.P_c[1] = np.count_nonzero(C_train==1)/num_C

    # Initialize conditional probability tables
    self.cpts = [NBCPT(i) for i in range(self.num_features)]

    # Train the classifier
    self._train(A_train, C_train)

  def _train(self, A_train, C_train):
    '''
    TODO train your NB classifier with the specified data and class labels
    hint: learn the parameters for the required CPTs
        - A_train: a 2-d numpy array where each row is a sample of assignments 
        - C_train: a 1-d n-element numpy where the elements correspond to
          the class labels of the rows in A
    '''
    for cpt in self.cpts:
        cpt.learn(A_train, C_train)
     

  def classify(self, entry):
    '''
    TODO return the log probabilites for class == 0 and class == 1 as a
    tuple for the given entry
    - entry: full assignment of variables 
    e.g. entry = np.array([0,1,1]) means variable A_0 = 0, A_1 = 1, A_2 = 1

    NOTE this must return both the predicated label {0,1} for the class
    variable and also the log of the conditional probability of this
    assignment in a tuple, e.g. return (c_pred, logP_c_pred)

    '''
    # Check if entry has missing field
    entries = None
    #flag = False
    if -1 in entry:
        entries = gen_missing([entry.tolist()])
        #flag = True
    else:
        entries = [entry]

    # Marginalize over all empty fields
    mpAC0, mpAC1 = 0.0, 0.0
    for entry in entries:
        pAC0, pAC1 = 1.0, 1.0
        for cpt in self.cpts:
            pAC0 *= cpt.get_cond_prob(entry, 0)
            pAC1 *= cpt.get_cond_prob(entry, 1)
        mpAC0 += pAC0
        mpAC1 += pAC1
    #if flag:
    #    print "Marginalized:", mpAC0, mpAC1
    mpAC0 *= self.P_c[0]
    mpAC1 *= self.P_c[1]

    pC0A = mpAC0/(mpAC0+mpAC1)
    pC1A = mpAC1/(mpAC0+mpAC1)

    if pC0A > pC1A:
        return (0, np.log(pC0A))
    else:
        return (1, np.log(pC1A))

  def predict_missing(self, entry, index):
    # Check if entry has missing field
    entries = None
    if -1 in entry:
        entries = gen_missing([entry.tolist()])
    else:
        entries = [entry]

    # Split into entries where a12 is 0,1
    Ai_entries = [[], []]
    for entry in entries:
        if entry[index] == 0:
            Ai_entries[0].append(entry)
        else:
            Ai_entries[1].append(entry)

    # Marginalize over all empty fields
    mpAiC = np.array([0.0, 0.0])
    for a in [0, 1]:
        for entry in Ai_entries[a]:
            for c in range(self.num_classes):
                pAC = 1.0
                for cpt in self.cpts:
                    pAC *= self.P_c[c]*cpt.get_cond_prob(entry, c)
                mpAiC[a] += pAC

    mpAiC /= mpAiC.sum()
    return mpAiC.argmax(), np.log(mpAiC.max())

    #pC0A = mpAC0/(mpAC0+mpAC1)
    #pC1A = mpAC1/(mpAC0+mpAC1)

    #if pC0A > pC1A:
    #    return (0, np.log(pC0A))
    #else:
    #    return (1, np.log(pC1A))

#--------------------------------------------------------------------------
# TANB CPT and classifier
#--------------------------------------------------------------------------
class TANBCPT(object):
  '''
  TANB CPT for a child attribute.  Each child can have one other attribute
  parent (or none in the case of the root), and the class variable as a
  parent
  '''

  def __init__(self, A_i, A_p):
    '''
    TODO create any persistent instance variables you need that hold the
    state of the learned parameters for this CPT
     - A_i: the index of the child variable
     - A_p: the index of its parent variable (in the Chow-Liu algorithm,
       the learned structure will have a single parent for each child)
    '''
    self.A_i = A_i
    self.A_p = A_p
    self.num_classes = 2
    self.num_vals = 2

  def learn(self, A, C, L=alpha):
    '''
    TODO populate any instance variables specified in __init__ to learn
    the parameters for this CPT
     - A: a 2-d numpy array where each row is a sample of assignments 
     - C: a 1-d n-element numpy where the elements correspond to the class
       labels of the rows in A
    '''
    if self.A_p == -1:
        self.p = np.zeros((self.num_classes, self.num_vals))
        for c in range(self.num_classes):
            Ac = A[C==c]
            for v in range(self.num_vals):
                self.p[c][v] = (np.count_nonzero(Ac.T[self.A_i]==v)+L)/float(len(Ac)+2*L)
    else:
        self.p = np.zeros((self.num_classes, self.num_vals, self.num_vals))
        for c in range(self.num_classes):
            Ac = A[C==c]
            for vp in range(self.num_vals):
                Avp = Ac.T[self.A_i][Ac.T[self.A_p]==vp]
                for vc in range(self.num_vals):
                    self.p[c][vp][vc] = (np.count_nonzero(Avp==vc)+L)/float(len(Avp)+2*L)

  def get_cond_prob(self, entry, c):
    '''
    TODO return the conditional probability P(X|Pa(X)) for the values
    specified in the example entry and class label c  
        - entry: full assignment of variables 
                e.g. entry = np.array([0,1,1]) means A_0 = 0, A_1 = 1, A_2 = 1
        - c: the class               
    '''
    if self.A_p == -1:
        return self.p[c][entry[self.A_i]]
    else:
        return self.p[c][entry[self.A_p]][entry[self.A_i]]


class TANBClassifier(NBClassifier):
  '''
  TANB classifier class specification
  '''

  def __init__(self, A_train, C_train):
    '''
    TODO create any persistent instance variables you need that hold the
    state of the trained classifier and populate them with a call to
    _train()
        - A_train: a 2-d numpy array where each row is a sample of
          assignments 
        - C_train: a 1-d n-element numpy where the elements correspond to
          the class labels of the rows in A

    '''
    # Initialize dimension parameters
    self.num_classes = 2
    self.num_data, self.num_features = A_train.shape

    # Initialize probabilities for each class
    num_C = float(C_train.shape[0])
    self.P_c = np.zeros(2)
    self.P_c[0] = np.count_nonzero(C_train==0)/num_C
    self.P_c[1] = np.count_nonzero(C_train==1)/num_C

    # Initialize empty list of CPTs
    self.cpts = []

    # Get dependencies from tree
    mst = get_mst(A_train, C_train)
    root = get_tree_root(mst)
    edges = list(get_tree_edges(mst, root))
    for edge in edges:
        self.cpts.append(TANBCPT(edge[1], edge[0]))

    # Add any children that were missed
    covered_children = zip(*edges)[1]
    for i in range(self.num_features):
        if i not in covered_children:
            self.cpts.append(TANBCPT(i, -1))

    # Train the classifier
    self._train(A_train, C_train)


# load all data
A_base, C_base = load_vote_data()

def evaluate(classifier_cls, train_subset=False):
    '''
    evaluate the classifier specified by classifier_cls using 10-fold cross
    validation
    - classifier_cls: either NBClassifier or TANBClassifier
    - train_subset: train the classifier on a smaller subset of the training
    data
    NOTE you do *not* need to modify this function
    '''
    global A_base, C_base

    A, C = A_base, C_base

    # score classifier on specified attributes, A, against provided labels,
    # C
    def get_classification_results(classifier, A, C):
        results = []
        pp = []
        for entry, c in zip(A, C):
            c_pred, _ = classifier.classify(entry)
            results.append((c_pred == c))
            pp.append(_)
        #print 'logprobs', np.array(pp)
        return results

    # partition train and test set for 10 rounds
    M, N = A.shape
    tot_correct = 0
    tot_test = 0
    step = M / 10
    for holdout_round, i in enumerate(xrange(0, M, step)):
        A_train = np.vstack([A[0:i,:], A[i+step:,:]])
        C_train = np.hstack([C[0:i], C[i+step:]])
        A_test = A[i:i+step,:]
        C_test = C[i:i+step]
        if train_subset:
            A_train = A_train[:16,:]
            C_train = C_train[:16]

        # train the classifiers
        classifier = classifier_cls(A_train, C_train)
  
        train_results = get_classification_results(classifier, A_train, C_train)
        #print '  train correct {}/{}'.format(np.sum(nb_results), A_train.shape[0])
        test_results = get_classification_results(classifier, A_test, C_test)
        tot_correct += sum(test_results)
        tot_test += len(test_results)

    return 1.*tot_correct/tot_test, tot_test

def evaluate_incomplete_entry(classifier_cls):

  global A_base, C_base

  # train a TANB classifier on the full dataset
  classifier = classifier_cls(A_base, C_base)

  # load incomplete entry 1
  entry = load_incomplete_entry()

  #entry = np.array([-1 for _ in entry])

  c_pred, logP_c_pred = classifier.classify(entry)

  print '  P(C={}|A_observed) = {:2.4f}'.format(c_pred, np.exp(logP_c_pred))

  return

def predict_vote(classifier_cls):

  global A_base, C_base

  # train a TANB classifier on the full dataset
  classifier = classifier_cls(A_base, C_base)

  # load incomplete entry 1
  entry = load_incomplete_entry()

  c_pred, logP_c_pred = classifier.predict_missing(entry, 11)

  print '  P(C={}|A_observed) = {:2.4f}'.format(c_pred, np.exp(logP_c_pred))

  return

def main():
    '''
    TODO modify or add calls to evaluate() to evaluate your implemented
    classifiers
    '''
    print 'Naive Bayes'
    accuracy, num_examples = evaluate(NBClassifier, train_subset=False)
    print '  10-fold cross validation total test accuracy {:2.4f} on {} examples'.format(
    accuracy, num_examples)

    print 'TANB Classifier'
    accuracy, num_examples = evaluate(TANBClassifier, train_subset=False)
    print '  10-fold cross validation total test accuracy {:2.4f} on {} examples'.format(
    accuracy, num_examples)

    print 'Naive Bayes Classifier on missing data'
    evaluate_incomplete_entry(NBClassifier)

    print 'TANB Classifier on missing data'
    evaluate_incomplete_entry(TANBClassifier)

    print 'Naive Bayes A12 for missing congressman'
    predict_vote(NBClassifier)

    print 'TANB A12 for missing congressman'
    predict_vote(TANBClassifier)
    
    print 'Naive Bayes'
    accuracy, num_examples = evaluate(NBClassifier, train_subset=True)
    print '  10-fold cross validation total test accuracy {:2.4f} on subset'.format(
    accuracy, num_examples)

    print 'TANB Classifier'
    accuracy, num_examples = evaluate(TANBClassifier, train_subset=True)
    print '  10-fold cross validation total test accuracy {:2.4f} on subset'.format(
    accuracy, num_examples)



if __name__ == '__main__':
    main()
