#!/usr/bin/env python

import numpy as np

aGc = np.array([[0.0, 1.0], [1.0, 0.0]])
bGa = np.array([[0.0, 1.0], [1.0, 0.0]])
cGb = np.array([[0.0, 1.0], [1.0, 0.0]])

s = 0.0
for a in [0,1]:
    for b in [0,1]:
        for c in [0,1]:
            s += aGc.T[a][c]*bGa.T[b][a]*cGb.T[c][b]

print s

aGc = np.array([[0.0, 1.0], [1.0, 0.0]])
bGa = np.array([[0.0, 1.0], [1.0, 0.0]])
C = np.array([0.0, 1.0])

s = 0.0
for a in [0,1]:
    for b in [0,1]:
        for c in [0,1]:
            s += aGc.T[a][c]*bGa.T[b][a]*C[c]

print s
