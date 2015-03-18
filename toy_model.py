import math
import numpy as np
from scipy.stats import lognorm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from random import random

import toy_model_mc

import clr
import csharp
import System

from numba import jit

import logging as log
log.basicConfig(level=log.INFO)

log.info("Setting up lognormal distribution")
ln = lognorm(1, 0, 0.5/math.exp(0.5))
log.info("ln.mean = %s", ln.mean())

data = [
	# expiry, bid, ask
	[ 1.0, 5.0, 5.5 ],
	[ 2.0, 2.0, 4.5 ],
	[ 3.0, 3.0, 3.5 ],
]


# plot data
log.info("Plotting data")
expiries, bids, asks = zip(*data)
plt.clf()
plt.axis([0, 4, 1.5, 6.0])
plt.xlabel("Expiry")
plt.ylabel("Price")
handle_bid, = plt.plot(expiries, bids, "ro", label="Bid")
handle_ask, = plt.plot(expiries, asks, "go", label="Ask")
plt.legend(handles=[handle_bid, handle_ask], numpoints=1)
plt.savefig("out/toy-model-bid-ask.pdf")

# plot data with mids + ls trendline
log.info("Plotting data + mids + ls fit")
mids = [ 0.5 * (bid + ask) for bid, ask in zip(bids, asks) ]
plt.clf()
plt.axis([0, 4, 1.5, 6.0])
plt.xlabel("Expiry")
plt.ylabel("Price")
handle_bid, = plt.plot(expiries, bids, "ro", label="Bid")
handle_ask, = plt.plot(expiries, asks, "go", label="Ask")
handle_mid, = plt.plot(expiries, mids, "bo", label="Mid")
trendline = np.poly1d(np.polyfit(expiries, mids, 1))
plt.plot([0, 4], trendline([0, 4]), "b-")
plt.legend(handles=[handle_bid, handle_ask, handle_mid], numpoints=1)
plt.savefig("out/toy-model-mid-ls.pdf")

# fit probabilistically
def minus_prob(s):
	p = -1.0
	for T, bid, ask in data:
		fv = s[0] + s[1] * T
		p *= ln.pdf(ask - fv)
		p *= ln.pdf(fv - bid)
	return p

log.info("Optimization")
res = minimize(minus_prob, [ 6.25, -1.0 ])
print res
s = list(res.x)

# plot data with mids + ls trendline
log.info("Plotting data + ML fit")
fvs = [ s[0] + s[1] * T for T in expiries ]
plt.clf()
plt.axis([0, 4, 1.5, 6.0])
plt.xlabel("Expiry")
plt.ylabel("Price")
handle_bid, = plt.plot(expiries, bids, "ro", label="Bid")
handle_ask, = plt.plot(expiries, asks, "go", label="Ask")
handle_mid, = plt.plot(expiries, fvs, "bo", label="MLFV")
trendline = np.poly1d(np.polyfit(expiries, fvs, 1))
plt.plot([0, 4], trendline([0, 4]), "b-")
plt.legend(handles=[handle_bid, handle_ask, handle_mid], numpoints=1)
plt.savefig("out/toy-model-max-lik.pdf")

# generate mc
log.info("Monte carlo importance sampling")
N = 100000
psum = 0.0
s1 = np.zeros((2,), dtype="f")
# m = np.zeros((30, 30))
m = System.Array.CreateInstance(System.Double, 30, 30)
m_exps = np.linspace(0.0, 4.0, 31) # m.shape[1]+1)
m_exps = (m_exps+0.5*m_exps[1])[:-1]
#m_fvs = np.linspace(1.5, 6.0, m.shape[0])
#j = np.zeros(m.shape[1], dtype=np.int32)
#toy_model_mc.mc(s1, psum, minus_prob, m_exps, m, N)
psum = csharp.ToyModel.MC(System.Func[System.Array[float], float](minus_prob), m_exps.tolist(), m, N)
# for _ in range(N):
	# if _ % 1000 == 0:
		# print _
	# s1[0] = 6.0 + 0.5 * random()
	# s1[1] = -2.0 + 3.0 * random()
	# p = -minus_prob(s1)
	# psum += p
	# fv = s1[0] + s1[1] * m_exps
	# j = np.floor((fv - 1.5) / (6.0 - 1.5) * m.shape[0], j)
	# cond = (j >= 0) * (j < m.shape[0])
	# i = np.arange(m.shape[1])
	# m[j[cond], i[cond]] += p

fvs = [ s[0] + s[1] * T for T in expiries ]
plt.clf()
plt.axis([0, 4, 1.5, 6.0])
plt.xlabel("Expiry")
plt.ylabel("Price")
plt.imshow(m, extent=(0.0, 4.0, 1.5, 6.0), interpolation='bicubic', origin='lower', cmap=cm.Blues)
handle_bid, = plt.plot(expiries, bids, "ro", label="Bid")
handle_ask, = plt.plot(expiries, asks, "go", label="Ask")
handle_mid, = plt.plot(expiries, fvs, "bo", label="MLFV")
#trendline = np.poly1d(np.polyfit(expiries, fvs, 1))
#plt.plot([0, 4], trendline([0, 4]), "b-")
plt.legend(handles=[handle_bid, handle_ask, handle_mid], numpoints=1)
plt.savefig("out/toy-model-max-lik-heat.pdf")
plt.show()