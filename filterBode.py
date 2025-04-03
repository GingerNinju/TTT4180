import numpy as np
from scipy.fft import fft
import scipy as sci
import sys
import matplotlib.pyplot as plt


#Fetch data from csv file
filterA = np.genfromtxt('..\\Lab4\\FrekvensresponsFilterA.csv', delimiter=',')
filterB = np.genfromtxt('..\\Lab4\\FrekvensresponsFilterB.csv', delimiter=',')


'''
#Plot the data
plt.figure(figsize=(10, 5))
plt.title('Frequency response of filter A and B')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.xlim(0, 150)
plt.ylim(0.01, 1000)
plt.plot(filterA[:,0], filterA[:,1], label='Filter A')
plt.plot(filterB[:,0], filterB[:,1], label='Filter B')
'''
fig, axs = plt.subplots(2)

axs[0].plot(filterA[:,0], filterA[:,1])
axs[0].set_title('Filter A')
#scale y axis logarithmic
#axs[0].set_yscale('log')
axs[0].set_xscale('log')
axs[0].grid(True)




axs[1].plot(np.abs(filterB[:,0]), filterB[:,1])
axs[1].set_title('Filter B')
#axs[1].set_yscale('log')
axs[1].set_xscale('log')
axs[1].grid(True)

plt.show()