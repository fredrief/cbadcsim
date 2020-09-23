import numpy as np
import matplotlib.pyplot as plt
from utils import *
from frontend import adc, lna, frontend
from siggen import sine_generator

# Define parameters
n = 5
beta = 0.48
rho = 0.03*beta
#eta2 = 104.3
eta2 = 200
eta = np.sqrt(eta2)

# Frequency
f = np.logspace(0,9,10000)
w = 2*np.pi*f
fs = 50e7

# Calculate Transfer functions
G2 = (beta**2/(np.abs(1+rho-np.exp(-1j*w/fs))**2))**n
H2 = G2/((G2 + eta2)**2)
G2H2 =  G2/((G2 + eta2))

H = np.sqrt(H2)
G = np.sqrt(G2)
STF = np.sqrt(G2H2)
G_dB = 20*np.log10(G)
H_dB = 20*np.log10(H)
STF_dB = 20*np.log10(STF)

# Results
wc = fs*np.arccos((-beta**2 * eta**(-2/n) + rho**2 +2*rho +2)/(2*(rho+1)))
fc = wc/(2*np.pi)
print(fc)

plt.semilogx(f, G_dB, label="ATF")
plt.semilogx(f, H_dB, label="NTF")
plt.semilogx(f, STF_dB, label="STF")
# Mark fc and fs
plt.axvline(x=fc, color='grey', linestyle='dashed', label='Cut-off freq')
plt.axvline(x=fs, color='red', linestyle='dashed', label='fs')

plt.legend()
plt.xlabel(r'$f$[Hz]')
plt.ylabel('[dB]')
plt.title('Amplitude respons of different transfer functions')
plt.show()
