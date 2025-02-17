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

sample_period, data = raspi_import('DATA_mStrom.bin', 1)
print(data.shape)
print(sample_period)

# print(data[0:100])
# data_list = data.tolist
# print(data_list)

#print(data_list[0:200])

#tid = np.arange(data.shape[0])*sample_period

#plt.plot(data[::5])
#plt.xlim(0,100)
#plt.show()


# tid = np.arange(data.shape[0])*sample_period


dataDetrend = sci.signal.detrend(data, axis = 0)
dataAdj = dataDetrend * 3.3 / 4096
data0 = dataAdj[100::5]
data1 = dataAdj[101::5]
data2 = dataAdj[102::5]
data3 = dataAdj[103::5]
data4 = dataAdj[104::5]
tid = np.arange(data0.shape[0])/31250
plt.plot(tid,data0)
plt.plot(tid,data1)
plt.plot(tid,data2)
#plt.plot(tid,data3)
#plt.plot(tid,data4)
plt.xlim(0,0.1)
plt.xlabel('Tid [s]')
plt.ylabel('Magnitude [V]')
plt.title('')
plt.grid()
plt.show()


'''
# data5fft = np.zeros_like(data5th, dtype=np.complex128) 
data5fft = fft(data5th, axis = 0)
#signalLength = data5th.size
freq = np.fft.fftfreq(31250,d=sample_period)


#print(data5fft[0:100])
#fig, axs = plt.subplots(2,1, sharex=True)
#axs[0].plot(freq, np.abs(data5fft))
#axs[1].plot(np.arange(len(data[::5])), data5th)
plt.plot(freq, data5fft)
plt.xlim(-1200,1200)
plt.ylabel('Magnitude [V/MHz]')
plt.xlabel('Frekvens [Hz]')
plt.title('Frekvensspektrum')
plt.grid()
#plt.xlim(0,100)
plt.show()

'''



 
