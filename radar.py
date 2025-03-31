import numpy as np
from scipy.fft import fft
import scipy as sci
import sys
import matplotlib.pyplot as plt


def raspi_import(path, channels=5):
    """
    Import data produced using adc_sampler.c.

    Returns sample period and a (`samples`, `channels`) `float64` array of
    sampled data from all `channels` channels.

    Example (requires a recording named `foo.bin`):
    ```
    >>> from raspi_import import raspi_import
    >>> sample_period, data = raspi_import('foo.bin')
    >>> print(data.shape)
    (31250, 5)
    >>> print(sample_period)
    3.2e-05

    ```
    """

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype='uint16').astype('float64')
        # The "dangling" `.astype('float64')` casts data to double precision
        # Stops noisy autocorrelation due to overflow
        data = data.reshape((-1, channels))

    # sample period is given in microseconds, so this changes units to seconds
    sample_period *= 1e-6
    return sample_period, data


# Import data from bin file
#if __name__ == '__main__':
#    sample_period, data = raspi_import(sys.argv[1] if len(sys.argv > 1)
#         else 'foo.bin')


#1139_Radar-30sec-AWAY_4.bin

sample_period, data = raspi_import('.\\radarData\\202503311106_radar-slow-AWAY.bin ', 1)
# print(data.shape)
# print(sample_period)

# print(data[0:100])
# data_list = data.tolist
# print(data_list)

#print(data_list[0:200])

#tid = np.arange(data.shape[0])*sample_period

#plt.plot(data[::5])
#plt.xlim(0,100)
#plt.show()


# tid = np.arange(data.shape[0])*sample_period
samples = data.shape[0]
cutoff_l = 15000 *5
cutoff_h = 90000 *5
dataDetrend = sci.signal.detrend(data, axis = 0)
dataAdj = dataDetrend * 3.3 / 4096
data0 = dataAdj[cutoff_l:cutoff_h:5]
data1 = dataAdj[cutoff_l+1:cutoff_h+1:5]
data2 = dataAdj[2::5]
data3 = dataAdj[3::5]
data4 = dataAdj[4::5]

data0Mean = np.mean(data0, axis = 0)
data1Mean = np.mean(data1, axis = 0)
data2Mean = np.mean(data2, axis = 0)
data3Mean = np.mean(data3, axis = 0)
data4Mean = np.mean(data4, axis = 0)

data0 = data0 - data0Mean
data1 = data1 - data1Mean
data2 = data2 - data2Mean
data3 = data3 - data3Mean
data4 = data4 - data4Mean

tid = np.arange(data0.shape[0]) # /31230
plt.plot(tid,data0)
plt.plot(tid,data1)
#plt.plot(tid,data2)
#plt.plot(tid,data3)
#plt.plot(tid,data4)
#plt.xlim(2,2.2)
plt.xlabel('Tid [s]')
plt.ylabel('Magnitude [V]')
plt.title('')
plt.grid()
plt.show()

dataC = data0 + 1j * data1

#FFT of the data
dataC_fft = np.fft.fftshift(np.fft.fft(dataC, axis=0))



# Calculate the sampling rate depening on Cutoff

sampling_rate = 31250
# Make x_axis m/s
x_axis = 31250 * 3e8 / 2 / 24.13e9



# x-axis in m/s
x = np.linspace(-x_axis/2, x_axis/2 , len(dataC_fft))


# Plot the FFT of the data logarithmic y-axis
plt.plot(x, np.abs(dataC_fft))
plt.xlim(-10, 10)
#log scale y axis
plt.yscale('log')
plt.xlabel('Speed [m/s]')   
plt.ylabel('Magnitude [V]')
plt.title('FFT of the data')
plt.grid()
plt.show()

'''
fig, axs = plt.subplots(2)

axs[0].plot(x,np.abs(dataC_fft))
axs[0].set_title('Q')
#scale y axis logarithmic
axs[0].set_yscale('log')

axs[1].plot(x,np.abs(dataC_fft))
axs[1].set_title('I')


plt.show()

'''
