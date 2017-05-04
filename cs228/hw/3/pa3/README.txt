Name: Luke Jaffe
SUNet ID: jaffe5
Explanation: All code is implemented in the provided skeleton files.

The ClusterGraph skeleton was used to construct a clique tree implementation of loopy belief propagation, as described in the lecture slides.

In each iteration of the algorithm, all messages were passed between factors in both directions. As the messages were computed, they were stored back to the same dictionary, which seemed to increase the speed of convergence considerably.

Marginals for each variable were computed by normalizing the product of the unary variable factor and all incoming messages. 

The argmax of these marginals was taken to get the predicted value of each resulting codeword bit.
