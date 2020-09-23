import csv
import numpy as np
from scipy import linalg as la
from scipy.integrate import quad, odeint
from scipy import signal
from sympy import Matrix
from utils import plot, estimate_signal_and_noise, get_state_space_mat
import matplotlib.pyplot as plt
from adclib import CIADC


ctrl_sig_filename = 'digital_control_1v0in'

# Define parameters
beta = 6250  # Hz
rho = -0*beta
kappa = 1.25
T = 54e-6
osr = 32
n = 5

CIADC = CIADC(n,beta,rho,kappa,T,osr, nperseg=1<<15)

# CIADC.plot_ntf_vs_stf(states=[1, 3, 5], show_eta=True, save_fig=False)

# CIADC.plot_atf()

CIADC.run_digital_estimator(ctrl_sig_filename, recalculate=False)

CIADC.plot_estimate_psd(states=[1, 3, 5], print_snr=True, save_fig=True)

CIADC.plot_psd_vs_ntf(state=1, show_stf=True, save_fig=True)
# CIADC.plot_psd_vs_ntf(state=3, show_stf=True, save_fig=False)
CIADC.plot_psd_vs_ntf(state=5, show_stf=True, save_fig=True)

