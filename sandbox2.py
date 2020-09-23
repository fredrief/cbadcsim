import csv
import numpy as np
from scipy import linalg as la
from scipy.integrate import quad
from scipy import signal
from sympy import Matrix
from utils import *
import matplotlib.pyplot as plt

# -------------- SCRIPT OPTIONS -------------- #
COMPUTE_NEW_ESTIMATE = True
COMPUTE_NEW_MATRICES = True

PLOT_TD = True
PLOT_FD = True


# ---------------------------------------------------------------------------- #
print('# Script Start')
# ---------------------------------------------------------------------------- #


# Import digital control signal
ctr_sig = np.array(())
with open('ctrl_sig/digital_control_1v0in.npy', 'rb') as f:
    ctr_sig = np.load(f)

# ctr_sig = np.flip(ctr_sig,0)
# Length of signal
L = len(ctr_sig[0,:]) # Window len

# Repeat evaluation for different n
nmax = 5
sigestmat = np.zeros((nmax, L))
for n in range(1,nmax+1):

    #  Parameters of analog system
    rho = 0
    beta = 6250  # Hz
    kappa = 1.25
    T = 54e-6
    OSR = 32
    # n = 5
    eta = ((T*beta/np.pi)*OSR)**n  # (Eta -> fcrit -> OSR)

    # Reduce dim of s
    s = ctr_sig[0:n,:]

    # State-space matrices
    A,B,C,GAMMA = get_state_space_mat(n,beta,eta,kappa,rho)

    # ------------------------------------------------------------------------ #
    print('# Offline Computations (n={})'.format(n))
    # ------------------------------------------------------------------------ #

    if COMPUTE_NEW_MATRICES:
        # Calculate covariance matrix
        Vf = la.solve_continuous_are(A.T,1/eta*C,B.dot(B.T),1)
        Vb = la.solve_continuous_are(-1*A.T,1/eta*C,B.dot(B.T),1)

        # Calculate precision matrix W
        W = np.linalg.solve((Vf+Vb), B)

        # Calculate matrices Af and Ab
        Af = la.expm((A - Vf / (eta**2) )*T)
        Ab = la.expm(-(A + Vb / (eta**2) )*T)

        # Calculate matrices Bf and Bb
        If = lambda t: la.expm( (A - Vf / (eta**2) )*(T-t) ).dot(GAMMA)
        Ib = lambda t: -la.expm( -(A + Vb / (eta**2) )*(T-t) ).dot(GAMMA)

        Bf = mtrap(If, 0, T, 10000)
        Bb = mtrap(Ib, 0, T, 10000)

        # Write to cache
        with open('sim_cache/Af.npy', 'wb') as f:
            np.save(f, Af)
        with open('sim_cache/Ab.npy', 'wb') as f:
            np.save(f, Ab)
        with open('sim_cache/Bf.npy', 'wb') as f:
            np.save(f, Bf)
        with open('sim_cache/Bb.npy', 'wb') as f:
            np.save(f, Bb)
    else:
        # Load from cache
        Af = np.array(())
        Ab = np.array(())
        Bf = np.array(())
        Bb = np.array(())
        with open('sim_cache/Af.npy', 'rb') as f:
            Af = np.load(f)
        with open('sim_cache/Ab.npy', 'rb') as f:
            Ab = np.load(f)
        with open('sim_cache/Bf.npy', 'rb') as f:
            Bf = np.load(f)
        with open('sim_cache/Bb.npy', 'rb') as f:
            Bb = np.load(f)

    # ------------------------------------------------------------------------ #
    print('# Estimate Signal (n={})'.format(n))
    # ------------------------------------------------------------------------ #

    u = np.zeros(L)
    if COMPUTE_NEW_ESTIMATE:
        mf = np.zeros((n,L))
        mb = np.zeros((n,L))

        # Backward and forward recursion
        for k in range(0,L-1):
            mf[:,k+1] = Af.dot(mf[:,k]) + Bf.dot(s[:,k])
            mb[:,L-k-2] = Ab.dot(mb[:,L-k-1]) + Bb.dot(s[:,L-k-2])
        for i in range(0,L):
            u[i] = (W.T).dot(mb[:,i]-mf[:,i])

        # write signal to cache
        with open('sim_cache/u.npy', 'wb') as f:
            np.save(f, u)
    else:
        # Load from cache
        u = np.array(())
        with open('sim_cache/u.npy', 'rb') as f:
            u = np.load(f)

    sigestmat[n-1,:] = u


# --------------------------------------------------------------------------
print('# Plot spectrum')
# ------------------------------------------------------------------------ #


if PLOT_TD:
    plt.plot(u, ',')
    plt.xlabel('n')
    plt.ylabel(r'Estimated input signal $\hat{u}$')
    plt.title('Estimated input signal - Time domain')
    plt.show()

if PLOT_FD:
    # Calculate critical freq
    wc = np.abs(beta)/(eta**(1/n))
    fc = wc/(2*np.pi)

    # Compute and plot power spectrum
    t = np.arange(L)

    fs_ref = 1.25*np.sin(2*np.pi*100*T*t)
    _, fs_spec = signal.welch(fs_ref, 1./T, axis=0, nperseg = 1<<15 )
    fs_ref_max = np.max(fs_spec)

    # Mark fc
    plt.axvline(x=fc, color='grey', linestyle='dashed', label=r'$f_c$')
    for i in range(0,nmax):
        sigestmat[i,:] = sigestmat[i,:] - np.mean(sigestmat[i,:])
        freq, u_spec = signal.welch(sigestmat[i,:], 1./T, axis=0, nperseg = 1<<15 )
        plt.semilogx(freq, 20*np.log10(np.abs(u_spec/fs_ref_max)), label='n={}'.format(i+1))

        sigpwr_est, npwr_est = estimate_signal_and_noise(sigestmat[i,:], 1/T, 0, 1/(2*T), nperseg=1<<15,  PLOT_PSD=False)
        sndr = 20 * np.log10(sigpwr_est / npwr_est)
        enob = (sndr - 1.76) / 6.02

        print('n={}'.format(i+1))
        print('SNDR: {} dB'.format(sndr))
        print('ENOB: {} bits'.format(enob))
    plt.xlim(1,1e4)
    plt.ylim(-200,0)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [dBFS]')
    plt.title('Estimated input signal - PSD')
    plt.legend()
    plt.show()

# ---------------------------------------------------------------------------- #
print('# -------------- Script End ')
# ---------------------------------------------------------------------------- #
