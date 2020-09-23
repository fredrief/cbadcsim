import numpy as np
import matplotlib.pyplot as plt
from utils import *
from frontend import adc, lna, frontend
from siggen import sine_generator

#------------ DEFINE PARAMS ---------- #

n = 5
beta = 0.48
rho = 0
T = 1/21.5
gamma = T*beta
OSR = 32
b = 1
kappa = 1.05
kappa2 = beta/n*(n-1)
eta = (gamma/np.pi * OSR)**n

#------------ GENERATE INPUT ---------- #
A = 0.7
fin = 0.1 + np.random.rand()*0.1
k = np.arange(0, 1e4, dtype=int)
t = k*T
u = A*np.sin(2*np.pi*fin*t)
u[1] = 0
# u += np.random.normal(0, 0.1, len(t))

# plot(u)

#------------ CYCLE ---------- #
# Declear varables
x = np.zeros((n, len(k)))
s = np.zeros((n, len(k)))

# x[:,0] = 2*np.random.rand(5)-1
for i in range(1, len(k)):
    x[0,i] = u[i]
    # x[0,i] += kappa2*(s[0,i-1] + s[1,i-1] + s[2,i-1] + s[3,i-1] + s[4,i-1])å
    for l in range(1,n):
        # print('x[l-1,i]: {}\ns[l,i-1]: {}\nx[l,i-1]: {}'.format(x[l-1,i],s[l,i-1],x[l,i-1]))
        x[l,i] = beta*(x[l-1,i] - kappa*s[l,i-1])+x[l,i-1]
        s[l,i] = 1 if x[l,i]>=0 else -1

        # print('x[l,i]: {}\ns[l,i]: {}\n'.format(x[l,i],s[l,i]))

plot(x[-1][600:700], title="A={}".format(A))
