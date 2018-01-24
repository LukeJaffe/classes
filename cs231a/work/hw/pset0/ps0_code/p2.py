#!/usr/bin/env python3

# CS231A Homework 0, Problem 2
import numpy as np
import matplotlib.pyplot as plt


def main():
    # ===== Problem 2a =====
    # Define Matrix M and Vectors a,b,c in Python with NumPy

    M, a, b, c = None, None, None, None

    # BEGIN YOUR CODE HERE
    M = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [0, 2, 2]
        ])

    a = np.array([1, 1, 0])

    b = np.array([-1, 2, 5])

    c = np.array([0, 2, 3, 2])
    # END YOUR CODE HERE

    # ===== Problem 2b =====
    # Find the dot product of vectors a and b, save the value to aDotb

    aDotb = None

    # BEGIN YOUR CODE HERE
    aDotb = np.dot(a, b)
    print('Value of aDotb:\n', aDotb)
    # END YOUR CODE HERE

    # ===== Problem 2c =====
    # Find the element-wise product of a and b

    # BEGIN YOUR CODE HERE
    p2_c = a*b
    print('Element-wise product value:\n', p2_c)
    # END YOUR CODE HERE

    # ===== Problem 2d =====
    # Find (a^T b)Ma

    # BEGIN YOUR CODE HERE
    p2_d = np.dot(a, b)*np.dot(M, a)
    print('Dot product value:\n', p2_d)
    # END YOUR CODE HERE

    # ===== Problem 2e =====
    # Without using a loop, multiply each row of M element-wise by a.
    # Hint: The function repmat() may come in handy.

    newM = None

    # BEGIN YOUR CODE HERE
    ar = np.tile(a[np.newaxis, :], (M.shape[0], 1))
    newM = M*ar 
    print('Element-wise product value:\n', newM)
    # END YOUR CODE HERE

    # ===== Problem 2f =====
    # Without using a loop, sort all of the values 
    # of M in increasing order and plot them.
    # Note we want you to use newM from e.

    # BEGIN YOUR CODE HERE
    s = np.sort(newM.reshape(-1))
    print('Sorted values:\n', s)
    plt.plot(range(len(s)), s)
    plt.title('Line Plot of Sorted Matrix Values')
    plt.xlabel('Index')
    plt.ylabel('Magnitude')
    plt.savefig('p2f.png')
    # END YOUR CODE HERE


if __name__ == '__main__':
    main()
