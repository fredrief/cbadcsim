from scipy import integrate
import numpy as np
from utils import mtrap, msimp
x2 = lambda x: x**2
x3 = lambda x: x**3
sin = lambda x: np.sin(x)

x  = lambda x: np.array([x**2, x**3, np.sin(x)])
t1 = 0
t2 = 100
nstep = 10000


print('QUAD: {}'.format(integrate.quad(x2, t1, t2)[0]))
print('QUAD: {}'.format(integrate.quad(x3, t1, t2)[0]))
print('QUAD: {}'.format(integrate.quad(sin, t1, t2)[0]))

print('MTRAP: {}'.format(mtrap(x, t1, t2, nstep=nstep)))
print('MSIMP: {}'.format(msimp(x, t1, t2, nstep=nstep)))
