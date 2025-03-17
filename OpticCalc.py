import numpy as np
from scipy.fft import fft, fftshift
import scipy as sci
import sys
import matplotlib.pyplot as plt

# Fetch array from .txt
R_curve = np.genfromtxt("./OptikklabData/250306/202503131854_finger-tommelAndreas-RED.txt", delimiter=",")
G_curve = np.genfromtxt("./OptikklabData/250306/202503131854_finger-tommelAndreas-GREEN.txt", delimiter=",")
B_curve = np.genfromtxt("./OptikklabData/250306/202503131854_finger-tommelAndreas-BLUE.txt", delimiter=",")

# Fetch array from .txt
R_curve_2 = np.genfromtxt("./OptikklabData/250306/202503131908_finger-tommel2-RED.txt", delimiter=",")
G_curve_2 = np.genfromtxt("./OptikklabData/250306/202503131908_finger-tommel2-GREEN.txt", delimiter=",")
B_curve_2 = np.genfromtxt("./OptikklabData/250306/202503131908_finger-tommel2-BLUE.txt", delimiter=",")

# Fetch array from .txt
R_curve_3 = np.genfromtxt("./OptikklabData/250306/202503131911_finger-tommel3-RED.txt", delimiter=",")
G_curve_3 = np.genfromtxt("./OptikklabData/250306/202503131911_finger-tommel3-GREEN.txt", delimiter=",")
B_curve_3 = np.genfromtxt("./OptikklabData/250306/202503131911_finger-tommel3-BLUE.txt", delimiter=",")

# Fetch array from .txt
R_curve_4 = np.genfromtxt("./OptikklabData/250306/202503131911_finger-tommel4-RED.txt", delimiter=",")
G_curve_4 = np.genfromtxt("./OptikklabData/250306/202503131911_finger-tommel4-GREEN.txt", delimiter=",")
B_curve_4 = np.genfromtxt("./OptikklabData/250306/202503131911_finger-tommel4-BLUE.txt", delimiter=",")

# Fetch array from .txt
R_curve_5 = np.genfromtxt("./OptikklabData/250306/202503131913_finger-tommel5-RED.txt", delimiter=",")
G_curve_5 = np.genfromtxt("./OptikklabData/250306/202503131913_finger-tommel5-GREEN.txt", delimiter=",")
B_curve_5 = np.genfromtxt("./OptikklabData/250306/202503131913_finger-tommel5-BLUE.txt", delimiter=",")

R_curves = [R_curve, R_curve_2, R_curve_3, R_curve_4, R_curve_5]
G_curves = [G_curve, G_curve_2, G_curve_3, G_curve_4, G_curve_5]
B_curves = [B_curve, B_curve_2, B_curve_3, B_curve_4, B_curve_5]

sampling_rate = 30

#peaks
peaks = np.zeros((5))
for i, (R, G, B) in enumerate(zip(R_curves, G_curves, B_curves), start=1):

    #Cut off the start
    Cutoff = 100
    R = R[Cutoff:]
    G = G[Cutoff:]
    B = B[Cutoff:]

   # Remove DC offset
    R = R - np.mean(R)
    G = G - np.mean(G)
    B = B - np.mean(B)
    
    # Detrend
    R = sci.signal.detrend(R)
    G = sci.signal.detrend(G)
    B = sci.signal.detrend(B)
    '''
    # Plot curves
    fig, axs = plt.subplots(3)
    axs[0].plot(R)
    axs[0].set_title(f'Red curve {i}')
    
    axs[1].plot(G)
    axs[1].set_title(f'Green curve {i}')
    
    axs[2].plot(B)
    axs[2].set_title(f'Blue curve {i}')
    
    plt.show()
    '''
    # FFT of the curve
    R_fft = fftshift(fft(R))
    G_fft = fftshift(fft(G))
    B_fft = fftshift(fft(B))
    '''
    # x-axis in Hz
    x = np.linspace(-sampling_rate * 60 / 2, sampling_rate * 60 / 2, len(R_fft))

    fig, axs = plt.subplots(3)

    axs[0].plot(x,np.abs(R_fft))
    axs[0].set_title('Red curve')

    axs[1].plot(x,np.abs(G_fft))
    axs[1].set_title('Green curve')

    axs[2].plot(x,np.abs(B_fft))
    axs[2].set_title('Blue curve')

    plt.show()
    '''
    
    peaks[i-1]= np.argmax(np.abs(R_fft))
    
    # Exclude the zero lag peak
    peaks[i-1] = peaks[peaks > len(R_fft) // 2]

print(peaks)

#Mean and variance of peaks
mean = np.mean(peaks)
variance = np.var(peaks)
print(mean)
print(variance)




