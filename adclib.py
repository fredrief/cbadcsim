import numpy as np
from scipy import linalg as la
from scipy.integrate import quad
from scipy import signal
from sympy import Matrix
import matplotlib.pyplot as plt
from utils import *
from datetime import datetime
import os

class CIADC(object):
    """
        This class holds all information about a chain-of-integrators ADC
    """
    def __init__(self, order, beta, rho, kappa, T, osr, nperseg=1<<15):
        """
            Initialize object with system parameters, state space matrices and transfer functions
            Initalize also with simulated/measured control signals if applicable

            Parameters
            T: float
                Sampling frequency of DC
        """
        # Parameters
        self.order = order
        self.beta = beta
        self.rho = rho
        self.kappa = kappa
        self.T = T
        self.osr = osr
        self.eta = ((T*beta/np.pi)*osr)**order
        self.nperseg = nperseg

        # State-space matrices
        A,B,C,gamma = get_state_space_mat(self.order,self.beta,self.eta,self.kappa,self.rho)
        self.A = A
        self.B = B
        self.C = C
        self.gamma = gamma

        logfmax = np.ceil(np.log10(1/(2*self.T)))
        logfmin = logfmax - 5
        res = 10000
        f = np.logspace(logfmin,logfmax,res)
        w = 2*np.pi*f
        # Calculate critical/cut-off frequency
        wc = np.abs(self.beta)/(self.eta**(1/order))
        fc = wc/(2*np.pi)

        self.fc = fc

        # Calculate transfer functions
        self.calculate_transfer_functions(w)

    def calculate_transfer_functions(self, w):
        """ Calculate and set transfer functions for given angular frequencies """
        # Calculate transfer functions
        G2, H2, stf2 = get_transfer_functions(self.order, self.beta, self.kappa, self.rho, w, self.T, self.osr)

        # Save set parameters
        self.freqs = w/(2*np.pi)
        self.G2 = G2
        self.H2 = H2
        self.stf2 = stf2

    def run_digital_estimator(self, ctrl_signal_filename, recalculate=False):
        """
            Run the digital estimation filter, using given control_signals

            recalculate: bool
                Force recalculation, even if previous estimate exist in cache
        """

        if not os.path.exists('cache/{}/input_signal_estimates.npy'.format(ctrl_signal_filename)) or recalculate:

            # ---------------------------------------------------------------- #
            # Import digital control signal
            ctrl_signals = np.array(())
            with open('ctrl_sig/{}.npy'.format(ctrl_signal_filename), 'rb') as f:
                ctrl_signals = np.load(f)
            # Use the dim of ctrl_signals as filter order
            N = len(ctrl_signals[:,0])

            print('----------------------------------------------------------')
            print('# STARTING DIGITAL ESTIMATOR')
            print('# Control signal filename: {}.npy'.format(ctrl_signal_filename))
            print('# Number of states to estimate: {}'.format(N))
            print('----------------------------------------------------------')

            # Length of input signal
            L = len(ctrl_signals[0,:])

            states = np.arange(1,N+1)
            input_signal_estimates = np.zeros((np.max(states), L))

            #  Parameters of analog system
            T = self.T
            OSR = self.osr
            beta = self.beta

            for n in states:

                # State-space matrices (must be recalculated for all n)
                A,B,C,gamma = get_state_space_mat(n,beta,self.eta,self.kappa,self.rho)

                eta = ((T*beta/np.pi)*OSR)**n
                # Reduce dim of s
                s = ctrl_signals[0:n,:]

                # ------------------------------------------------------------ #
                print('# Offline Computations (n={})'.format(n))
                # ------------------------------------------------------------ #

                # Calculate covariance matrix
                Vf = la.solve_continuous_are(A.T,1/eta*C,B.dot(B.T),1)
                Vb = la.solve_continuous_are(-1*A.T,1/eta*C,B.dot(B.T),1)

                # Calculate precision matrix W
                W = np.linalg.solve((Vf+Vb), B)

                # Calculate matrices Af and Ab
                Af = la.expm((A - Vf / (eta**2) )*T)
                Ab = la.expm(-(A + Vb / (eta**2) )*T)

                # Calculate matrices Bf and Bb
                If = lambda t: la.expm( (A - Vf / (eta**2) )*(T-t) ).dot(gamma)
                Ib = lambda t: -la.expm( -(A + Vb / (eta**2) )*(T-t) ).dot(gamma)

                Bf = mtrap(If, 0, T, 10000)
                Bb = mtrap(Ib, 0, T, 10000)

                # ------------------------------------------------------------ #
                print('# Estimate Signal (n={})'.format(n))
                # ------------------------------------------------------------ #

                u = np.zeros(L)
                mf = np.zeros((n,L))
                mb = np.zeros((n,L))

                # Backward and forward recursion
                for k in range(0,L-1):
                    mf[:,k+1] = Af.dot(mf[:,k]) + Bf.dot(s[:,k])
                    mb[:,L-k-2] = Ab.dot(mb[:,L-k-1]) + Bb.dot(s[:,L-k-2])
                for i in range(0,L):
                    u[i] = (W.T).dot(mb[:,i]-mf[:,i])
                input_signal_estimates[n-1, :] = u

            # write signal to cache
            if not os.path.exists('cache/{}'.format(ctrl_signal_filename)):
                os.mkdir('cache/{}'.format(ctrl_signal_filename))
            with open('cache/{}/input_signal_estimates.npy'.format(ctrl_signal_filename), 'wb') as f:
                np.save(f, input_signal_estimates)
        else:
            # Load from cache
            print('----------------------------------------------------------')
            print('# FOUND PREVIUOS CALCULATION IN CACHE')
            print('# Control signal filename: {}.npy'.format(ctrl_signal_filename))
            print('----------------------------------------------------------')
            input_signal_estimates = np.array(())
            with open('cache/{}/input_signal_estimates.npy'.format(ctrl_signal_filename), 'rb') as f:
                input_signal_estimates = np.load(f)
        self.input_signal_estimates = input_signal_estimates

    def plot_ntf_vs_stf(self, states = None, show_eta=False, save_fig=False):
        """
            Plot ideal noise and signal transfer function

            Parameters
            freq_range: tuple
                frequency range of interest in Hz. If none, 5 decades below fs/2 is used

            res: int
                number of frequency points

            save_fig: boolean
                save plot figures as png files under path plot/
        """
        f = self.freqs
        fc = self.fc
        N = self.order

        states = np.arange(1,N+1) if states is None else states

        for n in states:
            i = n-1
            plt.semilogx(f, 10*np.log10(self.H2[i,:]), label=r'$|H_{}(\omega)|$'.format(i+1))
            plt.semilogx(f, 10*np.log10(self.stf2[i,:]), label=r'$|STF_{}(\omega)|$'.format(i+1))
            if show_eta:
                # Calculate eta for each n and indicate
                eta = ((self.T*self.beta/np.pi)*self.osr)**n
                plt.vlines(x=fc-20, ymin=-(3+eta), ymax=-3)
        # Mark fc
        plt.axvline(x=fc, color='grey', linestyle='dashed', label='Cut-off freq')
        plt.legend()
        # plt.xlabel(r'$f/2\beta$')
        plt.xlabel(r'$f$[Hz]')
        plt.ylabel('[dB]')
        plt.title('Amplitude respons of different transfer functions')
        if save_fig:
           plt.savefig('plots/NTF_VS_STF_{}'.format(datetime.now().strftime('%m%d%Y_%H:%M:%S')))
        plt.show()

    def plot_atf(self, show_fc=False, save_fig=False):
        """
            Plot ideal analog transfer function

            Parameters
            freq_range: tuple
                frequency range of interest in Hz. If none, 5 decades below fs/2 is used

            res: int
                number of frequency points

            save_fig: boolean
                save plot figures as png files under path plot/
        """
        N = self.order
        f = self.freqs
        fc = self.fc
        for i in range(0,N):
            plt.semilogx(f, 10*np.log10(self.G2[i,:]), label=r'$|G_{}(\omega)|$'.format(i+1))

        # Mark fc
        if show_fc:
            plt.axvline(x=fc, color='grey', linestyle='dashed', label='Cut-off freq')
        plt.legend()
        # plt.xlabel(r'$f/2\beta$')-
        plt.xlabel(r'$f$[Hz]')
        plt.ylabel('[dB]')
        plt.title('Amplitude respons of different transfer functions')
        if save_fig:
           plt.savefig('plots/ATF_{}'.format(datetime.now().strftime('%m%d%Y_%H:%M:%S')))
        plt.show()

    def plot_estimate_in_time(self, state=None, save_fig=False):
        """ Show time domain plot of signal estimate

            state: int
                Which estimate state to show. Default to last state
        """
        if not hasattr(self, 'states_estimated'):
            print('No signal estimate to plot')
        else:
            n = state if not state is None else -1
            u = self.input_signal_estimates[n]
            plt.plot(u, ',')
            plt.xlabel('n')
            plt.ylabel(r'Estimated input signal $\hat{u}$')
            plt.title('Estimated input signal - Time domain')
            if save_fig:
                plt.savefig('plots/u(t)_{}'.format(datetime.now().strftime('%m%d%Y_%H:%M:%S')))
            plt.show()

    def plot_estimate_psd(self, states=None, print_snr=False, save_fig=False):
        """ Plot PSD of estimate

            states: array
                States to show
        """
        if not (hasattr(self, 'input_signal_estimates')):
            print('Cannot plot the selected states')
        else:
            input_signal_estimates = self.input_signal_estimates
            L = len(input_signal_estimates[0,:])
            N = len(input_signal_estimates[:,0])

            # States to plot
            states = np.arange(1,N+1) if states is None else states

            # Critical freq
            fc = self.fc
            # Sampling time
            T = self.T

            t = np.arange(L)

            # Reference FS signal
            fs_ref = 1.25*np.sin(2*np.pi*100*T*t)
            freq, fs_spec = signal.welch(fs_ref, 1./T, axis=0, nperseg = self.nperseg )
            fs_ref_max = np.max(fs_spec)

            # Mark fc
            plt.axvline(x=fc, color='grey', linestyle='dashed', label=r'$f_c$')
            for n in states:
                i = n-1
                input_signal_estimates[i,:] = input_signal_estimates[i,:] - np.mean(input_signal_estimates[i,:])
                _, u_spec = signal.welch(input_signal_estimates[i,:], 1./T, axis=0, nperseg = self.nperseg )
                plt.semilogx(freq, 10*np.log10(np.abs(u_spec/fs_ref_max)), label='n={}'.format(i+1))

                sigpwr_est, npwr_est = estimate_signal_and_noise(input_signal_estimates[i,:], 1/T, 0, 1/(2*T), nperseg=self.nperseg,  PLOT_PSD=False)
                sndr = 10 * np.log10(sigpwr_est / npwr_est)
                enob = (sndr - 1.76) / 6.02

                if print_snr:
                    print('SNDR(n={}): {} dB'.format(n,sndr))
                    print('ENOB(n={}): {} bits'.format(n,enob))
            plt.xlim(1,1e4)
            plt.ylim(-200,1)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('PSD [dBFS]')
            plt.title('Estimated input signal - PSD')
            plt.legend()
            if save_fig:
                plt.savefig('plots/PSD_{}'.format(datetime.now().strftime('%m%d%Y_%H:%M:%S')))
            plt.show()

    def plot_psd_vs_ntf(self, state, show_stf=False, save_fig=False):
        """ Plot PSD of estimate

            states: array
                States to show
        """
        if not (hasattr(self, 'input_signal_estimates')):
            print('Cannot plot the selected state')
        else:
            input_signal_estimates = self.input_signal_estimates
            L = len(input_signal_estimates[0,:])
            N = len(input_signal_estimates[:,0])

            # Critical freq
            fc = self.fc
            # Sampling time
            T = self.T

            t = np.arange(L)

            # Reference FS signal
            fs_ref = 1.25*np.sin(2*np.pi*100*T*t)
            freq, fs_spec = signal.welch(fs_ref, 1./T, axis=0, nperseg = self.nperseg )
            fs_ref_max = np.max(fs_spec)
            w = 2*np.pi*freq

            # Recalculate transfer functions
            self.calculate_transfer_functions(w)

            # Mark fc
            plt.axvline(x=fc, color='grey', linestyle='dashed', label=r'$f_c$')

            n = state
            i = n-1
            input_signal_estimates[i,:] = input_signal_estimates[i,:] - np.mean(input_signal_estimates[i,:])
            _, u_spec = signal.welch(input_signal_estimates[i,:], 1./T, axis=0, nperseg = self.nperseg )
            plt.semilogx(freq, 10*np.log10(np.abs(u_spec/1)), label='PSD')

            plt.semilogx(freq, 10*np.log10(self.H2[i,:]), label='NTF')
            if show_stf:
                plt.semilogx(freq, 10*np.log10(self.stf2[i,:]), label='STF')

            plt.xlim(1,1e4)
            plt.ylim(-200,0)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('PSD [dBFS]')
            plt.title('Estimated vs. theoretical PSD for n={}'.format(n))
            plt.legend()
            if save_fig:
                plt.savefig('plots/PSD_VS_NTF_{}'.format(datetime.now().strftime('%m%d%Y_%H:%M:%S')))
            plt.show()
