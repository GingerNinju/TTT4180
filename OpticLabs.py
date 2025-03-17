import numpy as np
from scipy.fft import fft, fftshift
import scipy as sci
import sys
import matplotlib.pyplot as plt

# Fetch array from .txt
R_curve = np.genfromtxt("./OptikklabData/250306/202503131841_finger-johs-Puls-RED.txt", delimiter=",")
G_curve = np.genfromtxt("./OptikklabData/250306/202503131841_finger-johs-Puls-GREEN.txt", delimiter=",")
B_curve = np.genfromtxt("./OptikklabData/250306/202503131841_finger-johs-Puls-BLUE.txt", delimiter=",")
'''
#Plot curves
plt.plot(G_curve)
plt.title('Green curve')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.show()
'''


fig, axs = plt.subplots(3)

axs[0].plot(R_curve)
axs[0].set_title('Red curve')

axs[1].plot(G_curve)
axs[1].set_title('Green curve')

axs[2].plot(B_curve)
axs[2].set_title('Blue curve')

plt.show()

#Cut off the start
Cutoff = 0
R_curve = R_curve[Cutoff:]
G_curve = G_curve[Cutoff:]
B_curve = B_curve[Cutoff:]

# Remove DC offset
R_curve = R_curve - np.mean(R_curve)
G_curve = G_curve - np.mean(G_curve)
B_curve = B_curve - np.mean(B_curve)

#Detrend
R_curve = sci.signal.detrend(R_curve)
G_curve = sci.signal.detrend(G_curve)
B_curve = sci.signal.detrend(B_curve)



# Zero-padding to increase FFT resolution
n = len(R_curve)
n_padded = 1 * n  # You can adjust this factor to increase the resolution further

R_curve_padded = np.pad(R_curve, (0, n_padded - n), 'constant')
G_curve_padded = np.pad(G_curve, (0, n_padded - n), 'constant')
B_curve_padded = np.pad(B_curve, (0, n_padded - n), 'constant')

# FFT of the curve
R_curve_fft = fftshift(fft(R_curve))
G_curve_fft = fftshift(fft(G_curve))
B_curve_fft = fftshift(fft(B_curve))

# Calculate the sampling rate depening on Cutoff
duration = 30 - Cutoff /len(R_curve) * 30
sampling_rate = 28  # frames per second, hva er den egentlig

# x-axis in Hz
x = np.linspace(-sampling_rate * 60 / 2, sampling_rate * 60 / 2, len(R_curve_fft))

fig, axs = plt.subplots(3)

axs[0].plot(x,np.abs(R_curve_fft))
axs[0].set_title('Red curve')

axs[1].plot(x,np.abs(G_curve_fft))
axs[1].set_title('Green curve')

axs[2].plot(x,np.abs(B_curve_fft))
axs[2].set_title('Blue curve')

plt.show()

'''
# Plot the R_FFT
plt.plot(x, np.abs(R_curve_fft))
plt.title('FFT of the R_curve')
plt.ylabel('Amplitude')
plt.xlabel('Frequency (Hz)')
plt.show()

# Find the frequency of the peak
peak = np.argmax(np.abs(R_curve_fft))
print("Peak frequency: ", x[peak])

# Plot the G_FFT
plt.plot(x, np.abs(G_curve_fft))
plt.title('FFT of the G_curve')
plt.ylabel('Amplitude')
plt.xlabel('Frequency (Hz)')
plt.show()

# Plot the B_FFT
plt.plot(x, np.abs(B_curve_fft))
plt.title('FFT of the B_curve')
plt.ylabel('Amplitude')
plt.xlabel('Frequency (Hz)')
plt.show()
'''