###############################################################################
# cluster graph data structure implementation (similar as the CliqueTree
# implementation in PA2)
# author: Billy Jun, Xiaocheng Li
# date: Jan 31st, 2016
###############################################################################

from factors import *
import numpy as np

from pprint import pprint

class ClusterGraph:
    def __init__(self, numVar=0):
        '''
        var - list: index/names of variables
        domain - list: the i-th element represents the domain of the i-th variable; 
            for this programming assignments, all the domains are [0,1]
        varToCliques - list of lists: the i-th element is a list with the indices 
            of cliques/factors that contain the i-th variable
        nbr - list of lists: it has the same length with the number of 
            cliques/factors, if factor[i] and factor[j] shares variable(s), 
            then j is in nbr[i] and i is in nbr[j]
        factor: a list of Factors
        sepset: two dimensional array, 
            sepset[i][j] is a list of variables shared by factor[i] and factor[j]
        messages: a dictionary to store the messages, 
            keys are (src, dst) pairs, values are the Factors of sepset[src][dst]. 
            Here src and dst are the indices for factors.
        '''
        self.var = [None for _ in range(numVar)]
        self.domain = [None for _ in range(numVar)]
        self.varToCliques = [[] for _ in range(numVar)]
        self.nbr = []
        self.factor = []
        self.sepset = []
        self.messages = {}
    
    def evaluateWeight(self, assignment):
        '''
        param - assignment: the full assignment of all the variables
        return: the multiplication of all the factors' values for this assigments
        '''
        a = np.array(assignment, copy=False)
        output = 1.0
        for f in self.factor:
            output *= f.val.flat[assignment_to_indices([a[f.scope]], f.card)]
            #print "ass: {}".format([a[f.scope]])
            #print "card: {}".format(f.card)
            #print "val: {}".format(f.val)
            #print "a2i: {}".format(assignment_to_indices([a[f.scope]], f.card))
            #print "output: {}".format(output)
        return output[0]
    
    def getInMessage(self, src, dst):
        '''
        param - src: the source factor/clique index
        param - dst: the destination factor/clique index
        return: Factor with var set as sepset[src][dst]
        
        In this function, the message will be initialized as an all-one vector if 
        it is not computed and used before. 
        '''
        """
        if (src, dst) not in self.messages:
            inMsg = Factor()
            inMsg.scope = self.sepset[src][dst]
            inMsg.card = [len(self.domain[s]) for s in inMsg.scope]
            inMsg.val = np.ones(np.prod(inMsg.card))
            self.messages[(src, dst)] = inMsg
        """
        return self.messages[(src, dst)]


    def runParallelLoopyBP(self, y, iterations, eps=1e-10, checkpoint=None): 
        '''
        param - iterations: the number of iterations you do loopy BP
          
        In this method, you need to implement the loopy BP algorithm. 
        The only values you should update in this function is self.messages. 
        
        Warning: Don't forget to normalize the message at each time. 
        You may find the normalize method in Factor useful.
        '''
        if checkpoint:
            checkpoint_dict = {}

        prob_arr, new_prob_arr = None, None
        for iter in range(iterations):
        ########################################################################
            # Update all messages
            for i,fi in enumerate(self.factor):
                #print "{}/{}".format(i, len(self.factor))
                for j in self.nbr[i]:
                    # Get the target factor
                    fj = self.factor[j]      
                    # Initialize the new message
                    new_message = None
                    # Take product of all incoming messages to fj
                    prod = Factor(f=fi)
                    for k in self.nbr[i]:
                        if k != j:
                            # Multiply fi by all incoming messages
                            m = self.messages[(k,i)]
                            prod = prod.multiply(m)
                    # Get all variables in intersect(i, j)
                    scope = set(prod.scope).intersection(set(fj.scope))
                    # Marginalize out all variables in i, not in j
                    new_message = prod.marginalize_all_but(scope)
                    # Store the new message
                    self.messages[(i, j)] = new_message.normalize()

            # Compute MAP for all variables
            new_prob_arr, estimate = self.getMarginalMAP()

            # Compute distance between iteration prob arr
            diff = "?"
            if prob_arr is not None:
                diff = np.linalg.norm(new_prob_arr-prob_arr)

            # Check the number of errors
            print "Iter {}: hamming={}, diff={}".format(iter+1,
                len(y) - np.count_nonzero(y.ravel()==estimate),
                diff
            )

            if checkpoint is not None and (iter+1) in checkpoint:
                checkpoint_dict[(iter+1)] = new_prob_arr

            prob_arr = new_prob_arr

        if checkpoint:
            return checkpoint_dict
        else:
            return new_prob_arr
 
        ############################################################################
        

    def estimateMarginalProbability(self, var):
        '''
        param - var: a single variable index
        return: the marginal probability of the var
        
        example: 
        >>> cluster_graph.estimateMarginalProbability(0)
        >>> [0.2, 0.8]
    
        Since in this assignment, we only care about the marginal 
        probability of a single variable, you only need to implement the marginal 
        query of a single variable.     
        '''
        ########################################################################
        # Marginalize all but var in all messages, take product
        s = self.factor[var]
        for i in self.nbr[var]:#self.messages.itervalues():
            f = self.messages[(i, var)]
            s = s.multiply(f)

        # Normalize to prob dist
        return s.normalize().val
        
        ###############################################################################
    

    def getMarginalMAP(self):
        '''
        In this method, the return value output should be the marginal MAP 
        assignments for the variables. You may utilize the method
        estimateMarginalProbability.
        
        example: (N=2, 2*N=4)
        >>> cluster_graph.getMarginalMAP()
        >>> [0, 1, 0, 0]
        '''
        ###############################################################################
        # To do: your code here  
        prob_arr = np.array(
            [self.estimateMarginalProbability(i) for i in self.var]
        ) 
        
        output = prob_arr.argmax(axis=1)
        ###############################################################################  
        return prob_arr, output
