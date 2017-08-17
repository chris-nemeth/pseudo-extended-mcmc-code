#This file produces the first illustration from the paper with N=2 pseudo-samples and a Gaussian proposal

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

#log-Gaussian density
def log_gaussian(x, mu, v):
    return -0.5 * np.log(2*np.pi) - 0.5*np.log(v) - 0.5*np.square(mu - x)/v

#target
def f(x):
    mu1, mu2 = -1, 1
    v1, v2 = 0.1, 0.02
    return np.exp(log_gaussian(x, mu1, v1)) + np.exp(log_gaussian(x, mu2, v2))

#proposal
def q(x):
    return np.exp(log_gaussian(x, 0, 2))


#Plot

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
#fig, axes = plt.subplots(1, 4)
xx = np.linspace(-3, 3, 300)
ax1.plot(xx, f(xx))
#ax1.plot(xx, q(xx))
ax1.set_title(r"$\pi(x)$")

xx, yy = np.meshgrid(xx, xx)
ax2.contour(xx, yy, 0.5*(q(xx)*f(yy) + f(xx)*q(yy)), colors='b')
ax2.set_title(r"$\pi(x_{1,2})$")

ax1.set_xlabel(r"$x$")
ax2.set_xlabel(r"$x_1$")
ax2.set_ylabel(r"$x_2$")

fig.tight_layout()
plt.show()
