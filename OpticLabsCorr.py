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

# Correlation of Red
RAutoCorrelate = sci.signal.correlate(R_curve, R_curve, mode = 'full')
#print("Lengden data0:",len(data0))
#print("Lengden autokorrelasjonen",len(RAutoCorrelate))
#print("Forholdet: ", len(data0)/len(RAutoCorrelate))
maxData0AutoCorrelate = np.max(RAutoCorrelate)
print(maxData0AutoCorrelate)
print("Autokorrelasjon R-Autocorrelated:", np.argmax(RAutoCorrelate) - np.floor(RAutoCorrelate.size/2))

#plot the Correlation
plt.plot(RAutoCorrelate)
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

plt.plot(GAutoCorrelate)
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

plt.plot(BAutoCorrelate)
plt.title('Correlation of the B_curve')
plt.ylabel('Amplitude')
plt.xlabel('')
plt.show()

