import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftshift
import numpy as np
import numpy.linalg as nla


def plot(y, x=None, xlabel='x', ylabel='y', title='Plot Title'):
    if x is None:
        x = np.linspace(1, len(y), len(y))
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axis('tight')
    plt.show()


def multiplot(y1, y2, x1=None, x2=None, legend1='', legend2='', xlabel='x', ylabel='y', title='Plot Title'):
    if x1 is None:
        x1 = np.linspace(1, len(y1), len(y1))
    if x2 is None:
        x2 = np.linspace(1, len(y2), len(y2))
    plt.plot(x1, y1, label=legend1)
    plt.plot(x2, y2, label=legend2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axis('tight')
    plt.legend()
    plt.show()


def bin2dec(bin):
    """Converts binary number to decimal. A binary number is a
    numpy array of length N, with binary elements"""
    dec_out = 0
    for i in range(1, len(bin) + 1):
        dec_out += bin[-i] * np.power(2, i - 1)
    return dec_out


def estimate_signal_and_noise(x, fs, f1, f2, nperseg=None, PLOT_PSD=False):
    """ Calculate SNDR of signal "x" for noise frequencies between
        f1 and f2.
        Value returned in dB

        Using welch configuration from scipy.signal.welch documentation
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html

        Pxx is calculated with different settings for noise and signal
        to obtain more accurate results. The conversion between density
        and spectrum is not understood by the author."""

    # Calculate PSD. Use density for noise and spectrum for signal
    if not nperseg:
        nperseg = len(x)

    f, Pxx_den = signal.welch(x, fs, nperseg=nperseg)
    f, Pxx_spec = signal.welch(x, fs, 'flattop', nperseg=nperseg, scaling='spectrum')

    # indices of frequency bands
    fbin = f[1] - f[0]
    in1 = int(np.floor(f1 / fbin))
    in2 = int(np.floor(f2 / fbin))
    isig = np.argmax(Pxx_den)

    sig_width = 10               # Width of signal in number of points
    sig_noise_width = 20        # Region used to estimae noise under signal

    if PLOT_PSD:
        plt.figure()
        # Mark different parts of spectrum with different colors
        plt.loglog(f[:(in1 + 1)], Pxx_den[:(in1 + 1)], color='orange')
        plt.loglog(f[in2:], Pxx_den[in2:], color='orange')
        plt.loglog(f[in1:(in2 + 1)], Pxx_den[in1:(in2 + 1)], color='red')
        plt.loglog(f[isig - sig_width:isig + sig_width + 1],
                     Pxx_den[isig - sig_width:isig + sig_width + 1], color='green')
        plt.loglog(f[(isig + sig_width):(isig + sig_width + sig_noise_width + 1)],
                     Pxx_den[(isig + sig_width):(isig + sig_width + sig_noise_width + 1)], color='blue')
        plt.ylabel('PSD [V^2/Hz]')
        plt.title('Power Spectral Density')
        plt.show()

        plt.figure()

    # Obtain signal and noise measurements. Noise is calculated by
    # integrating noise density over freq band of interest
    # to obtain a more accurate result: addume mean noise under
    # signal band.
    npwr_est = np.mean(Pxx_den[(isig + 10):]) * fs / 2
    sigpwr_est = Pxx_spec.max()
    npwr_est_inbw = 0
    for i in range(in1, in2 + 1):
        # Do not integrate noise over signal band
        if i < isig - sig_width or i > isig + sig_width:
            npwr_est_inbw += Pxx_den[i] * fbin
    npwr_est_inbw += np.mean(Pxx_den[(isig + sig_width):(
        isig + sig_width + sig_noise_width + 1)]) * sig_width * 2 * fbin

    return sigpwr_est, npwr_est_inbw


def flicker_noise(x, fs, C):
    length = len(x)
    freqs = np.logspace(0, 7,
                        num=int(length / 2))
    noise_coeffs = np.zeros_like(freqs, dtype='complex')
    for k in range(0, int(len(freqs))):
        # Create random phase
        phi = np.random.rand() * 2 * np.pi
        noise_coeffs[k] = (C / np.sqrt(freqs[k])) * np.exp(phi * 1j)

    noise_coeffs_flip = np.flip(noise_coeffs)
    noise_coeffs = np.append(noise_coeffs_flip, noise_coeffs)

    return np.abs(np.fft.ifft(noise_coeffs[0:len(x)]))

def norm2(A):
    """ Return the 2-norm of matrix A along first dimension"""
    return nla.norm(A, ord=2, axis=0)

def get_transfer_functions(n, beta, kappa, rho, w, T, osr):
    """
        Get ideal transfer functions
        w: array
            angular frequencies to calculate transfer function for
    """

    # Declare arrays
    G2s = np.zeros((n, len(w)))
    H2s = np.zeros((n, len(w)))
    stf2s = np.zeros((n, len(w)))

    # Fix DC problem
    if 0.0 in w:
        w[np.where(w==0.0)] = 1e-10
    for l in range(1, n+1):
        # eta must be calculated for each order
        eta = ((T*beta/np.pi)*osr)**l
        eta2 = eta**2

        G2s[l-1, :] = (beta**2/(w**2 + rho**2))**l
        G2 = G2s[l-1, :]
        H2s[l-1, :] = G2/((G2 + eta2)**2)
        stf2s[l-1, :] = G2/(G2 + eta2)

    return G2s, H2s, stf2s

def get_state_space_mat(n, beta, eta, kappa, rho):
    """
        Generate state space matrices A, B, C and GAMMA.
        Cs and Cm correspond to a single or multiple output AS
    """

    A = np.zeros((n,n))
    gamma = np.zeros((n,n))
    B = np.zeros((n,1))
    C = np.zeros((n,1))
    for k in range(0,n):
        A[k,k] = -rho
        gamma[k,k] = kappa*beta # Sign of kappa only affect phase of estimate
        if k>0:
            A[k,k-1] = beta

    B[0,0] = beta
    C[-1,0] = 1

    return A,B,C,gamma

def mtrap(func, t1, t2, nstep=10000):
    """
    Integrate function 'func' using the trapezoidal method from t1, to t2 with nstep equally spaced steps.
    """
    res = np.zeros_like(func(0))
    time_steps = np.linspace(t1, t2, nstep)
    res += func(t1) + func(t2)
    for i in range(1, nstep-1):
        t = time_steps[i]
        res = res + 2*func(t)
    res *= (t2-t1)/(2*(nstep-1))
    return res

def msimp(func, t1, t2, nstep=10000):
    """
    Integrate function 'func' using Simpson's method from t1, to t2 with nstep equally spaced steps.
    """
    res = np.zeros_like(func(0))
    time_steps = np.linspace(t1, t2, nstep)
    res += func(t1) + func(t2)
    for i in range(1, nstep-1):
        t = time_steps[i]
        if i % 2:
            res = res + 4*func(t)
        else:
            res = res + 2*func(t)
    res *= (t2-t1)/(3*(nstep-1))
    return res

