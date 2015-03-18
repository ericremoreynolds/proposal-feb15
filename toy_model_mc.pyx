from random import random

def mc(s1, psum, minus_prob, exps, m, int N):
    cdef float fv
    cdef int i, j, k
    cdef int n_exps = len(exps)
    cdef int n_px = m.shape[0]
    print "Monte Carlo starting"
    for k in range(N):
        if k % 1000 == 0:
            print k
        s1[0] = 6.0 + 0.5 * random()
        s1[1] = -2.0 + 3.0 * random()
        p = -minus_prob(s1)
        psum += p
        for i in range(n_exps):
            fv = s1[0] + s1[1] * exps[i]
            j = int((fv - 1.5) / (6.0 - 1.5) * n_px)
            if 0 <= j and j < n_px:
                m[j, i] += p