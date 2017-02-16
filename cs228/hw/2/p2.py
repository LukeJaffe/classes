#!/usr/bin/env python

import itertools

enum = {
    0 : 'A',
    1 : 'B',
    2 : 'C',
    3 : 'D'
}

table = [
    ([0, 0, 0, 0], 1./8.),
    ([1, 1, 0, 0], 1./8.),
    ([1, 1, 1, 0], 1./4.),
    ([0, 1, 0, 1], 1./4.),
    ([1, 0, 1, 1], 1./4.)
]

vals = [0, 1]

def marg1(i, i_v):
    mp = 0.0
    for entry in table:
        if entry[0][i] == i_v:
           mp += entry[1] 
    return mp

def marg2(i, i_v, j, j_v):
    mp = 0.0
    for entry in table:
        if entry[0][i] == i_v and entry[0][j] == j_v:
           mp += entry[1] 
    return mp

def cond_marg1(i, i_v, k, k_v):
    cp = 0.0
    mp = 0.0
    for entry in table:
        if entry[0][k] == k_v:
            mp += entry[1]
            if entry[0][i] == i_v:
               cp += entry[1] 
    if mp == 0.0:
        return 0.0
    else:
        return cp/mp

def cond_marg2(i, i_v, k, k_v, l, l_v):
    cp = 0.0
    mp = 0.0
    for entry in table:
        if entry[0][k] == k_v and entry[0][l] == l_v:
            mp += entry[1]
            if entry[0][i] == i_v:
               cp += entry[1] 
    if mp == 0.0:
        return 0.0
    else:
        return cp/mp

def cond2_marg1(i, i_v, j, j_v, k, k_v):
    cp = 0.0
    mp = 0.0
    for entry in table:
        if entry[0][k] == k_v:
            mp += entry[1]
            if entry[0][i] == i_v and entry[0][j] == j_v:
               cp += entry[1] 
    if mp == 0.0:
        return 0.0
    else:
        return cp/mp

def cond2_marg2(i, i_v, j, j_v, k, k_v, l, l_v):
    cp = 0.0
    mp = 0.0
    for entry in table:
        if entry[0][k] == k_v and entry[0][l] == l_v:
            mp += entry[1]
            if entry[0][i] == i_v and entry[0][j] == j_v:
               cp += entry[1] 
    if mp == 0.0:
        return 0.0
    else:
        return cp/mp

def check_ind(i, j):
    for i_v in vals:
        mi = marg1(i, i_v)
        for j_v in vals:
            mj = marg1(j, j_v)
            mij = marg2(i, i_v, j, j_v)
            if mi*mj != mij:
                return False
    return True

def check_cond_ind1(i, j, k):
    for k_v in vals:
        for i_v in vals:
            mi = cond_marg1(i, i_v, k, k_v)
            for j_v in vals:
                mj = cond_marg1(j, j_v, k, k_v)
                mij = cond2_marg1(i, i_v, j, j_v, k, k_v)
                if mi*mj != mij:
                    return False
    return True
        
def check_cond_ind2(i, j, k, l):
    for l_v in vals:
        for k_v in vals:
            for i_v in vals:
                mi = cond_marg2(i, i_v, k, k_v, l, l_v)
                for j_v in vals:
                    mj = cond_marg2(j, j_v, k, k_v, l, l_v)
                    mij = cond2_marg2(i, i_v, j, j_v, k, k_v, l, l_v)
                    if mi*mj != mij:
                        return False
    return True
        
for v in vals:
    print "P(C={}) = {}".format(v, marg1(2, v))
print

for v in vals:
    print "P(D={}) = {}".format(v, marg1(3, v))
print

for v_i in vals:
    for v_j in vals:
        for v_k in vals:
            print "P(A={}|C={},D={}) = {}".format(v_i, v_j, v_k, cond_marg2(0, v_i, 2, v_j, 3, v_k))
print

for v_i in vals:
    for v_j in vals:
        for v_k in vals:
            print "P(B={}|A={},D={}) = {}".format(v_i, v_j, v_k, cond_marg2(1, v_i, 0, v_j, 3, v_k))
print

import sys; sys.exit()
for i,j in itertools.combinations(range(4), 2):
    ind_ij = check_ind(i, j)
    print "{} i {}: {}".format(enum[i], enum[j], ind_ij)
    k,l = [e for e in range(4) if e != i and e != j]
    cond_ind_ij_k = check_cond_ind1(i, j, k)
    print "{} i {} | {}: {}".format(enum[i], enum[j], enum[k], cond_ind_ij_k)
    cond_ind_ij_l = check_cond_ind1(i, j, l)
    print "{} i {} | {}: {}".format(enum[i], enum[j], enum[l], cond_ind_ij_l)
    cond_ind_ij_kl = check_cond_ind2(i, j, k, l)
    print "{} i {} | {}, {}: {}".format(enum[i], enum[j], enum[k], enum[l], cond_ind_ij_kl)


