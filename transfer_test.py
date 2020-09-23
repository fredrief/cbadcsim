from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

num = [10, 0]
den = [(1+1.05*10), -1]

H = signal.TransferFunction(num, den)

w, h = signal.freqz(num, den, fs=1/21.5)

fig, ax1 = plt.subplots()
ax1.set_title('Digital filter frequency response')

ax1.plot(w, 20 * np.log10(abs(h)), 'b')
ax1.set_ylabel('Amplitude [dB]', color='b')
ax1.set_xlabel('Frequency [rad/sample]')

ax2 = ax1.twinx()
angles = np.unwrap(np.angle(h))
ax2.plot(w, angles, 'g')
ax2.set_ylabel('Angle (radians)', color='g')
ax2.grid()
ax2.axis('tight')
plt.show()
