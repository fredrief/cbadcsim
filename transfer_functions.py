import numpy as np
import matplotlib.pyplot as plt
from utils import *
from frontend import adc, lna, frontend
from siggen import sine_generator

# Define parameters
beta = 6250  # Hz
rho = 0
kappa = 1.25
T = 54e-6
OSR = 32
n = 5
eta = ((T*beta/np.pi)*OSR)**n # (Eta -> fcrit -> OSR)
eta2 = eta**2

# Frequency
f = np.logspace(-1,4,1000)
w = 2*np.pi*f

print(eta2)

# Calculate Transfer functions
G2 = (beta**2/(w**2+rho**2))**n
H2 = G2/((G2 + eta2)**2)
G2H2 =  G2/((G2 + eta2))

H = np.sqrt(H2)
G = np.sqrt(G2)
STF = np.sqrt(G2H2)
G_dB = 20*np.log10(G)
H_dB = 20*np.log10(H)
STF_dB = 20*np.log10(STF)

# Results
wc = np.abs(beta)/(eta**(1/n))
fc = wc/(2*np.pi)

# plt.semilogx(f/(2*beta), G_dB, label="ATF")

# plt.semilogx(f, H_dB, label="NTF")
# plt.semilogx(f, STF_dB, label="STF")
# # Mark fc
# plt.axvline(x=fc, color='grey', linestyle='dashed', label='Cut-off freq')
# plt.legend()
# # plt.xlabel(r'$f/2\beta$')
# plt.xlabel(r'$f$[Hz]')
# plt.ylabel('[dB]')
# plt.title('Amplitude respons of different transfer functions')
# plt.show()
