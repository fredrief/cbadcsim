import csv
import numpy as np
from scipy import linalg as la
from scipy.integrate import quad
from sympy import Matrix
from utils import plot, estimate_signal_and_noise, get_state_space_mat
import matplotlib.pyplot as plt

# -------------- SCRIPT OPTIONS -------------- #
USE_CONV_AIDS = False  # Use convergence aids on matrices A and B

# -------------- Import digital control signal -------------- #
s = np.array(())
with open('ctrl_sig/digital_control_1in.npy', 'rb') as f:
    s = np.load(f)

# -------------- Parameters of analog system -------------- #
rho = 0
beta = 6250  # Hz
kappa = 1.25
T = 54e-6
OSR = 32
n = 1
eta = ((T*beta/np.pi)*OSR)**n # (Eta -> fcrit -> OSR)

# Reduce dim of s
s = s[0:n,:]

# State-space matrices
A,B,C,GAMMA = get_state_space_mat(n,beta,eta,kappa,rho)

# print(A)
# print(B)
# print(C)
# print(GAMMA)

# -------------- Offline computations -------------- #
# Calculate covariance matrix
Vf = la.solve_continuous_are(A.T,1/eta*C,B.dot(B.T),1)
Vb = la.solve_continuous_are(A.T,1/eta*C,-B.dot(B.T),-1)

# To correct for sign on diagonal (for some reason)
Vb = Vb*((np.identity(n)*-2) + np.ones((n,n)))

# Calculate precision matrix W
W = np.linalg.solve((Vf+Vb), B)

# Calculate matrices Af and Ab
Af = np.exp((A - Vf / (eta**2) )*T)
Ab = np.exp(-(A + Vb / (eta**2) )*T)

# Calculate matrices Bf and Bb
Bf = np.zeros_like(Af)
Bb = np.zeros_like(Ab)
for i in range(0,n):
    Bf[i,i] = quad(lambda t: np.exp((A[i,i] - Vf[i,i] / (eta**2) ) * (T-t) ) * GAMMA[i,i], 0, T)[0]
    Bb[i,i] = quad(lambda t: -1*np.exp(-(A[i,i] - Vf[i,i] / (eta**2) ) * (T-t) ) * GAMMA[i,i], 0, T)[0]



# Convergence aids
if USE_CONV_AIDS:
    Af = Af/la.norm(Af)
    Ab = Ab/la.norm(Ab)
    Bf = Bf/la.norm(Bf)
    Bb = Bb/la.norm(Bb)
    # # Write matrix A on Jordan form
    # Af_symb = Matrix(Af)
    # Ab_symb = Matrix(Ab)
    # P, Af_J = Af_symb.jordan_form()
    # P, Ab_J = Ab_symb.jordan_form()
    # Af_J = np.array(Af_J).astype(np.float64)
    # Ab_J = np.array(Ab_J).astype(np.float64)
    # Af = Af_J
    # Ab = Ab_J


# ---------------------- Estimate signal --------------------------

# print('Vf: \n{}'.format(Vf))
# print('Vb: \n{}'.format(Vb))

# print('Af: \n{}'.format(Af))
# print('Bf: \n{}'.format(Bf))
# print('Ab: \n{}'.format(Ab))
# print('Bb: \n{}'.format(Bb))

N = len(s[0,:]) # Window len
u = np.zeros(len(s[0,:]))
mf = np.zeros((n,N))
mb = np.zeros((n,N))
# Backward and forward recursion
for k in range(0,N-1):
    mf[:,k+1] = Af.dot(mf[:,k]) + Bf.dot(s[:,k])
    mb[:,N-k-2] = Ab.dot(mb[:,N-k-1]) + Bb.dot(s[:,N-k-2])

for i in range(0,N):
    u[i] = (W.T).dot(mb[:,i]-mf[:,i])

# -------------- Plot spectrum ---------------
# Remove offset
buf = 10000
# u = u[buf:-buf]-np.mean(u[buf:-buf])
u = u-np.mean(u)
plot(u)

L = len(u)
t = np.arange(L)
U = np.fft.fft(u)
U = U[:int((L-1)/2)+1]
freq = (np.fft.fftfreq(L))[:int((L-1)/2)+1]
plt.semilogx(freq, 20*np.log10(np.abs(U)), label='N={}'.format(n))
# plt.axvline(x=72.4, color='grey', linestyle='dashed', label='fi')
plt.legend()
plt.show()

print('Script end')
