import numpy as np
from utils import *
from frontend import adc, lna, frontend
from siggen import sine_generator

#------------------------------------------------------------#
#-----------------------  SETTINGS  -------------------------#
#------------------------------------------------------------#
# Plot PSD of the digital signal
PLOT_PSD = False
# Plot PSD of the analog signal
PLOT_PSD_IN = False
# Plot digital vs. analog signal in time domain
PLOT_TIME_DOMAIN = False
# Print SNDR
PRINT_SNDR = False
# Print ENOB
PRINT_ENOB = False

#------------------------------------------------------------#
#---------------------  CONFIGURATION  ----------------------#
#------------------------------------------------------------#
# ADC Config
fs = 20e6   # Sampling freq
res = 10    # resolution bits
vref = 0.8  # Volt
adc = adc.ctrlbnd_adc(res, vref, fs)

# LNA Config
lna_gain = 1
lna = lna.lna(lna_gain)

# Frontend Config
frontend = frontend.frontend(lna, adc)

# Signal Generator Config
thermal_noise_density = 10e-6  # uV/rt(Hz)
flicker_noise_coef = 0
siggen = sine_generator(length=1e4,
                        thermal_noise_density=thermal_noise_density,
                        fs=fs,
                        flicker_noise_coef=flicker_noise_coef)

#------------------------------------------------------------#
#------------------------  STIMULI  -------------------------#
#------------------------------------------------------------#
# Create sine wave stimuli
amp = vref / 4
x = siggen.generate(freq=4.127e6, amp=amp, offset=vref / 2)
if PLOT_PSD_IN:
    sigpwr_est, npwr_est = estimate_signal_and_noise(
        x, fs=fs, f1=2.5e6, f2=7.5e6, PLOT_PSD=True)

#------------------------------------------------------------#
#--------------------------  DUT  ---------------------------#
#------------------------------------------------------------#
# Convert values
adc_out = np.zeros(res * len(x))
# for i in range(0, len(x)):
#     adc_out[res * i:res * i + res] = frontend.convert(x[i])
print(adc)


#------------------------------------------------------------#
#---------------------  POST PROCESS  -----------------------#
#------------------------------------------------------------#
# Convert back to analog
dac_out = np.zeros_like(x)
for i in range(0, len(x)):
    dac_out[i] = bin2dec(adc_out[res * i:res * i + res]
                         ) / np.power(2, res) * vref

# Do plotting
if PLOT_TIME_DOMAIN:
    multiplot(x * lna_gain, dac_out,
              legend1='LNA Out', legend2='DAC Out')

sigpwr_est, npwr_est = estimate_signal_and_noise(
    dac_out, fs=fs, f1=0, f2=fs / 2, PLOT_PSD=PLOT_PSD)
sndr = 20 * np.log10(sigpwr_est / npwr_est)
enob = (sndr - 1.76) / 6.02


if PRINT_SNDR:
    print('True thermal noise power:        {}'.format(
        np.power(thermal_noise_density, 2) * fs / 2))
    print('Estimated noise power:           {}'.format(npwr_est))
    print('True Signal Power:               {}'.format(
        np.power((amp / np.sqrt(2)), 2)))
    print('Estimated Signal Power:          {}\n'.format(sigpwr_est))

    print('SNDR:                            {} dB'.format(sndr))
    print('ENOB:                            {} bits'.format(enob))
