import numpy as np
from scipy.fft import fft, fftshift
import scipy as sci
import sys
import matplotlib.pyplot as plt

# Fetch array from .txt
R_curve = np.genfromtxt("./OptikklabData/250306/202503131841_finger-johs-Puls-RED.txt", delimiter=",")
G_curve = np.genfromtxt("./OptikklabData/250306/202503131841_finger-johs-Puls-GREEN.txt", delimiter=",")
B_curve = np.genfromtxt("./OptikklabData/250306/202503131841_finger-johs-Puls-BLUE.txt", delimiter=",")

#Plot curves
plt.plot(R_curve)
plt.title('Red curve')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.show()

plt.plot(G_curve)
plt.title('Green curve')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.show()

plt.plot(B_curve)
plt.title('Blue curve')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.show()


#Cut off the start
Cutoff = 100
R_curve = R_curve[Cutoff:]
G_curve = G_curve[Cutoff:]
B_curve = B_curve[Cutoff:]

# Remove DC offset
R_curve = R_curve - np.mean(R_curve)
G_curve = G_curve - np.mean(G_curve)
B_curve = B_curve - np.mean(B_curve)

#detrend
R_curve = sci.signal.detrend(R_curve)
G_curve = sci.signal.detrend(G_curve)
B_curve = sci.signal.detrend(B_curve)

#Plot curves
plt.plot(R_curve)
plt.title('Red curve')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.show()

def calculate_periodicity(autocorrelation):
    # Find the peaks in the autocorrelation
    peaks, _ = sci.signal.find_peaks(autocorrelation)
    
    # Exclude the zero lag peak
    peaks = peaks[peaks > len(autocorrelation) // 2]
    
    if len(peaks) > 0:
        # Find the largest peak
        largest_peak_index = np.argmax(autocorrelation[peaks])
        period = peaks[largest_peak_index] - len(autocorrelation) // 2
        return period
    else:
        return None



# Correlation of Red
RAutoCorrelate = sci.signal.correlate(R_curve, R_curve, mode = 'full')



R_period = calculate_periodicity(RAutoCorrelate)

print("Periodicity of R_curve:", R_period)


x = np.arange(-len(RAutoCorrelate)/2, len(RAutoCorrelate)/2, 1)

#plot the Correlation
plt.plot(x,RAutoCorrelate)
plt.title('Correlation of the R_curve')
plt.ylabel('Amplitude')
plt.xlabel('')
plt.show()

# Correlation of Green
GAutoCorrelate = sci.signal.correlate(G_curve, G_curve, mode = 'full')
#print("Lengden data0:",len(data0))
#print("Lengden autokorrelasjonen",len(GAutoCorrelate))
#print("Forholdet: ", len(data0)/len(GAutoCorrelate))
maxData0AutoCorrelate = np.max(GAutoCorrelate)
print(maxData0AutoCorrelate)
print("Autokorrelasjon G-Autocorrelated:", np.argmax(GAutoCorrelate) - np.floor(GAutoCorrelate.size/2))

x = np.arange(-len(GAutoCorrelate)/2, len(GAutoCorrelate)/2, 1)

plt.plot(x, GAutoCorrelate)
plt.title('Correlation of the G_curve')
plt.ylabel('Amplitude')
plt.xlabel('')
plt.show()

# Correlation of Blue
BAutoCorrelate = sci.signal.correlate(B_curve, B_curve, mode = 'full')
#print("Lengden data0:",len(data0))
#print("Lengden autokorrelasjonen",len(BAutoCorrelate))
#print("Forholdet: ", len(data0)/len(BAutoCorrelate))
maxData0AutoCorrelate = np.max(BAutoCorrelate)
print(maxData0AutoCorrelate)
print("Autokorrelasjon B-Autocorrelated:", np.argmax(BAutoCorrelate) - np.floor(BAutoCorrelate.size/2))

x = np.arange(-len(BAutoCorrelate)/2, len(BAutoCorrelate)/2, 1)

plt.plot(x, BAutoCorrelate)
plt.title('Correlation of the B_curve')
plt.ylabel('Amplitude')
plt.xlabel('')
plt.show()

