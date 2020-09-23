from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from utils import plot
np.random.seed(1234)


fs = 20e6
N = 1e5
amp = 0.4
freq = 4.127e6
noise_power = np.power(10e-6, 2) * fs / 2

time = np.arange(N) / fs
x = amp * np.sin(2 * np.pi * freq * time) + 0.4
x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)

f, Pxx_den = signal.welch(x, fs, nperseg=1024)
f, Pxx_spec = signal.welch(x, fs, 'flattop', 1024, scaling='spectrum')

f1 = 2.5e6
f2 = 7.5e6

fsig = freq
fbin = f[1] - f[0]
isig = int(np.floor(fsig / fbin)) + 1
in1 = int(np.floor(f1 / fbin))
in2 = int(np.floor(f2 / fbin))

plt.figure()
# Mark different parts of spectrum with different colors
plt.semilogy(f[:in1], Pxx_den[:in1], color='orange')
plt.semilogy(f[in2:], Pxx_den[in2:], color='orange')
plt.semilogy(f[in1:in2], Pxx_den[in1:in2], color='red')
plt.semilogy(f[isig - 6:isig + 6],
             Pxx_den[isig - 6:isig + 6], color='green')
plt.xlabel('frequency [Hz]')
plt.ylabel('Linear spectrum [V RMS]')
plt.show()
plt.figure()

npwr_est = np.mean(Pxx_den[(np.argmax(Pxx_den) + 5):]) * fs / 2
npwr_est_inbw = npwr_est * (f2 - f1) / (f[-1] - f[0])
sigpwr_est = Pxx_spec.max()

print('True noise power:\t\t\t{}\nEstimated noise power:\t\t{}\n'.format(
    noise_power, npwr_est))

print('True Signal Power:\t\t\t{}\nEstimated Signal Power:\t\t{}\n'.format(
    np.power((amp / np.sqrt(2)), 2), sigpwr_est))

sndr = 20 * np.log10(sigpwr_est / npwr_est)
print('SNDR:{} dB'.format(sndr))
enob = (sndr - 1.76) / 6.02
print('ENOB: {} bits'.format(enob))
