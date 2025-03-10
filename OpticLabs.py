import numpy as np
from scipy.fft import fft, fftshift
import scipy as sci
import sys
import matplotlib.pyplot as plt

# Fetch array from .txt
R_curve = np.genfromtxt("../Lab3/Data/Finger_1cmR.txt", delimiter=",")
G_curve = np.genfromtxt("../Lab3/Data/Finger_1cmG.txt", delimiter=",")
B_curve = np.genfromtxt("../Lab3/Data/Finger_1cmB.txt", delimiter=",")

# Remove DC offset
R_curve = R_curve - np.mean(R_curve)
G_curve = G_curve - np.mean(G_curve)
B_curve = B_curve - np.mean(B_curve)

# Zero-padding to increase FFT resolution
n = len(R_curve)
n_padded = 24 * n  # You can adjust this factor to increase the resolution further

R_curve_padded = np.pad(R_curve, (0, n_padded - n), 'constant')
G_curve_padded = np.pad(G_curve, (0, n_padded - n), 'constant')
B_curve_padded = np.pad(B_curve, (0, n_padded - n), 'constant')

# FFT of the curve
R_curve_fft = fftshift(fft(R_curve_padded))
G_curve_fft = fftshift(fft(G_curve_padded))
B_curve_fft = fftshift(fft(B_curve_padded))

# x-axis from -146 to 146
x = np.linspace(-146, 146, len(R_curve_fft))

# Plot the R_FFT
plt.plot(x, np.abs(R_curve_fft))
plt.title('FFT of the R_curve')
plt.ylabel('Amplitude')
plt.xlabel('Frequency')
plt.show()

# Find the frequency of the peak
peak = np.argmax(np.abs(R_curve_fft))
print("Peak frequency: ", peak)

# Plot the G_FFT
plt.plot(x, np.abs(G_curve_fft))
plt.title('FFT of the G_curve')
plt.ylabel('Amplitude')
plt.xlabel('Frequency')
plt.show()

# Plot the B_FFT
plt.plot(x, np.abs(B_curve_fft))
plt.title('FFT of the B_curve')
plt.ylabel('Amplitude')
plt.xlabel('Frequency')
plt.show()
