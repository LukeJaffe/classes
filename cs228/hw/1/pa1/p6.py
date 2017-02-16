#!/usr/bin/env python

import numpy as np

n = 100000000

c = np.random.rand(n)
i = np.random.rand(n)

# Calculate E(C)
print "1. E(C) = {}".format(c.mean())

# Calculate P(A=1)
s = c+i
print "P(A=1) = {}".format(len((s>1.5).nonzero()[0])/float(n))

# Calculate E(C|A=1)
aidx = s>1.5
print "2. E(C|A=1) = {}".format(c[aidx].mean())
print "P(C|A=1) = {}".format(len(c[aidx])/float(n))

# Calculate P(C|I=0.95)
lb,ub = 0.95-0.00005, 0.95+0.00005
lidx = (i>lb).nonzero()
uidx = (i<ub).nonzero()
iidx = lidx[0][np.in1d(lidx, uidx)]
print "3. E(C|I=0.95) = {}".format(c[iidx].mean())

# Calculate P(C|A=1, I=0.95)
aidx = aidx.nonzero()[0]
idx = iidx[np.in1d(iidx, aidx)]
print "4. E(C|A=1,I=0.95) = {}".format(c[idx].mean())
