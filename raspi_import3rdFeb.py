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

sample_period, data = raspi_import('..\\1-Klapp_corner01.bin', 1)
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

tid = np.arange(data0.shape[0])/31230
plt.plot(tid,data0)
plt.plot(tid,data1)
plt.plot(tid,data2)
#plt.plot(tid,data3)
#plt.plot(tid,data4)
#plt.xlim(2,2.2)
plt.xlabel('Tid [s]')
plt.ylabel('Magnitude [V]')
plt.title('')
plt.grid()
plt.show()

'''
data0 = data0[round(31230*3/10):round(31230*4/10):]
data1 = data1[round(31230*3/10):round(31230*4/10):]
data2 = data2[round(31230*3/10):round(31230*4/10):]
''' 
# Correlation
data0AutoCorrelate = sci.signal.correlate(data0, data0, mode = 'full')
print("Lengden data0:",len(data0))
print("Lengden autokorrelasjonen",len(data0AutoCorrelate))
print("Forholdet: ", len(data0)/len(data0AutoCorrelate))
maxData0AutoCorrelate = np.max(data0AutoCorrelate)
print(maxData0AutoCorrelate)
print("Autokorrelasjon data0:", np.argmax(data0AutoCorrelate) - np.floor(data0AutoCorrelate.size/2))

data01Correlate = sci.signal.correlate(data0, data1, mode = 'full')
maxData01Correlate = np.max(data01Correlate)
# maxData01Correlate = maxData01Correlate - len(data1) 
#print(maxData01Correlate)
KorrRes01 = np.argmax(data01Correlate) - np.floor(data01Correlate.size/2)
print("Korrelasjon mellom data 0 og 1:", KorrRes01)

data02Correlate = sci.signal.correlate(data0, data2, mode = 'full')
maxData02Correlate = np.max(data02Correlate)
#print(maxData02Correlate)
KorrRes02 = np.argmax(data02Correlate) - np.floor(data02Correlate.size/2)
print("Korrelasjon mellom data 0 og 2:", KorrRes02)

data12Correlate = sci.signal.correlate(data1, data2, mode = 'full')
maxData12Correlate = np.max(data12Correlate)
#print(maxData12Correlate)
KorrRes12 = np.argmax(data12Correlate) - np.floor(data12Correlate.size/2)
print("Korrelasjon mellom data 1 og 2:", KorrRes12)

angle = np.arctan(np.sqrt(3)*((-KorrRes02-KorrRes01)/(-(KorrRes02)+KorrRes01-2*KorrRes12)))
if ((-(KorrRes02)+KorrRes01-2*KorrRes12)) < 0:
    angle = angle + np.pi
    print("Vinkel (pi added):", angle)
else: print("Vinkel:", angle)
# Convert angle to degrees
angle = angle*180/np.pi
print("Vinkel i grader:", angle)


plt.plot(data0AutoCorrelate)
plt.show()


# Standardavvik og varians av målinger
sample_period, vardata = raspi_import('..\\1-Klapp_corner01.bin', 1)

vardata0 = vardata[100::5]
vardata1 = vardata[101::5]
vardata2 = vardata[102::5]

vardata0Mean = np.mean(vardata0, axis = 0)
vardata1Mean = np.mean(vardata1, axis = 0)
vardata2Mean = np.mean(vardata2, axis = 0)

vardata0 = vardata0 - vardata0Mean
vardata1 = vardata1 - vardata1Mean
vardata2 = vardata2 - vardata2Mean

#Korrelasjon
vardata01Correlate = sci.signal.correlate(vardata0, vardata1, mode = 'full')
vardata02Correlate = sci.signal.correlate(vardata0, vardata2, mode = 'full')
vardata12Correlate = sci.signal.correlate(vardata1, vardata2, mode = 'full')

#Vinkel
KorrRes01 = np.argmax(vardata01Correlate) - np.floor(vardata01Correlate.size/2)
KorrRes02 = np.argmax(vardata02Correlate) - np.floor(vardata02Correlate.size/2)
KorrRes12 = np.argmax(vardata12Correlate) - np.floor(vardata12Correlate.size/2)
angle0 = np.arctan(np.sqrt(3)*((-KorrRes02-KorrRes01)/(-(KorrRes02)+KorrRes01-2*KorrRes12)))

sample_period, vardata = raspi_import('..\\2-Klapp_corner01.bin', 1)

vardata0 = vardata[100::5]
vardata1 = vardata[101::5]
vardata2 = vardata[102::5]

vardata0Mean = np.mean(vardata0, axis = 0)
vardata1Mean = np.mean(vardata1, axis = 0)
vardata2Mean = np.mean(vardata2, axis = 0)

vardata0 = vardata0 - vardata0Mean
vardata1 = vardata1 - vardata1Mean
vardata2 = vardata2 - vardata2Mean

#Korrelasjon
vardata01Correlate = sci.signal.correlate(vardata0, vardata1, mode = 'full')
vardata02Correlate = sci.signal.correlate(vardata0, vardata2, mode = 'full')
vardata12Correlate = sci.signal.correlate(vardata1, vardata2, mode = 'full')

#Vinkel
KorrRes01 = np.argmax(vardata01Correlate) - np.floor(vardata01Correlate.size/2)
KorrRes02 = np.argmax(vardata02Correlate) - np.floor(vardata02Correlate.size/2)
KorrRes12 = np.argmax(vardata12Correlate) - np.floor(vardata12Correlate.size/2)
angle1 = np.arctan(np.sqrt(3)*((-KorrRes02-KorrRes01)/(-(KorrRes02)+KorrRes01-2*KorrRes12)))

sample_period, vardata = raspi_import('..\\3-Klapp_corner01.bin', 1)

vardata0 = vardata[100::5]
vardata1 = vardata[101::5]
vardata2 = vardata[102::5]

vardata0Mean = np.mean(vardata0, axis = 0)
vardata1Mean = np.mean(vardata1, axis = 0)
vardata2Mean = np.mean(vardata2, axis = 0)

vardata0 = vardata0 - vardata0Mean
vardata1 = vardata1 - vardata1Mean
vardata2 = vardata2 - vardata2Mean

#Korrelasjon
vardata01Correlate = sci.signal.correlate(vardata0, vardata1, mode = 'full')
vardata02Correlate = sci.signal.correlate(vardata0, vardata2, mode = 'full')
vardata12Correlate = sci.signal.correlate(vardata1, vardata2, mode = 'full')

#Vinkel
KorrRes01 = np.argmax(vardata01Correlate) - np.floor(vardata01Correlate.size/2)
KorrRes02 = np.argmax(vardata02Correlate) - np.floor(vardata02Correlate.size/2)
KorrRes12 = np.argmax(vardata12Correlate) - np.floor(vardata12Correlate.size/2)
angle2 = np.arctan(np.sqrt(3)*((-KorrRes02-KorrRes01)/(-(KorrRes02)+KorrRes01-2*KorrRes12)))

sample_period, vardata = raspi_import('..\\4-Klapp_corner01.bin', 1)

vardata0 = vardata[100::5]
vardata1 = vardata[101::5]
vardata2 = vardata[102::5]

vardata0Mean = np.mean(vardata0, axis = 0)
vardata1Mean = np.mean(vardata1, axis = 0)
vardata2Mean = np.mean(vardata2, axis = 0)

vardata0 = vardata0 - vardata0Mean
vardata1 = vardata1 - vardata1Mean
vardata2 = vardata2 - vardata2Mean

#Korrelasjon
vardata01Correlate = sci.signal.correlate(vardata0, vardata1, mode = 'full')
vardata02Correlate = sci.signal.correlate(vardata0, vardata2, mode = 'full')
vardata12Correlate = sci.signal.correlate(vardata1, vardata2, mode = 'full')

#Vinkel
KorrRes01 = np.argmax(vardata01Correlate) - np.floor(vardata01Correlate.size/2)
KorrRes02 = np.argmax(vardata02Correlate) - np.floor(vardata02Correlate.size/2)
KorrRes12 = np.argmax(vardata12Correlate) - np.floor(vardata12Correlate.size/2)
angle3 = np.arctan(np.sqrt(3)*((-KorrRes02-KorrRes01)/(-(KorrRes02)+KorrRes01-2*KorrRes12)))

sample_period, vardata = raspi_import('..\\5-Klapp_corner01.bin', 1)

vardata0 = vardata[100::5]
vardata1 = vardata[101::5]
vardata2 = vardata[102::5]

vardata0Mean = np.mean(vardata0, axis = 0)
vardata1Mean = np.mean(vardata1, axis = 0)
vardata2Mean = np.mean(vardata2, axis = 0)

vardata0 = vardata0 - vardata0Mean
vardata1 = vardata1 - vardata1Mean
vardata2 = vardata2 - vardata2Mean

#Korrelasjon
vardata01Correlate = sci.signal.correlate(vardata0, vardata1, mode = 'full')
vardata02Correlate = sci.signal.correlate(vardata0, vardata2, mode = 'full')
vardata12Correlate = sci.signal.correlate(vardata1, vardata2, mode = 'full')

#Vinkel
KorrRes01 = np.argmax(vardata01Correlate) - np.floor(vardata01Correlate.size/2)
KorrRes02 = np.argmax(vardata02Correlate) - np.floor(vardata02Correlate.size/2)
KorrRes12 = np.argmax(vardata12Correlate) - np.floor(vardata12Correlate.size/2)
angle4 = np.arctan(np.sqrt(3)*((-KorrRes01-KorrRes02)/(-(KorrRes01)+KorrRes02-2*KorrRes12)))

#Varians og Standaravvik av vinkler
angleVar = np.var([angle0, angle1, angle2, angle3, angle4])
angleStd = np.std([angle0, angle1, angle2, angle3, angle4])
print("Vinkel var:", angleVar)
print("Vinkel std:", angleStd)


 
