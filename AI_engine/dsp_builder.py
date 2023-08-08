import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import re
import pytz
from datetime import datetime

import enum

from scipy import signal
from scipy.signal import butter, lfilter, cheby1, cheby2
from numpy import array

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def butter_bandstop_filter(data, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        i, u = signal.butter(order, [low, high], btype='bandstop')
        y = signal.lfilter(i, u, data)
        return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def cheby1(cutoff, max_ripple, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.cheby1(order, max_ripple, normal_cutoff, btype='low', analog=False)
    return b, a

def cheby1_filter(data, max_ripple, cutoff, fs, order=5):
    b, a = cheby1(cutoff, max_ripple, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def cheby2(cutoff, min_attenuation, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.cheby2(order, min_attenuation, normal_cutoff, btype='low', analog=False)
    return b, a

def cheby2_filter(data, min_attenuation, cutoff, fs, order=5):
    b, a = cheby2(cutoff, min_attenuation, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def sine_generator(fs, sinefreq, duration):
    T = duration
    nsamples = fs * T
    w = 2. * np.pi * sinefreq
    t_sine = np.linspace(0, T, nsamples, endpoint=False)
    y_sine = np.sin(w * t_sine)
    result = pd.DataFrame({ 
        'data' : y_sine} ,index=t_sine)
    return result

#data = pd.read_csv('https://raw.githubusercontent.com/iotanalytics/IoTTutorial/main/data/SCG_data.csv').drop('Unnamed: 0',1).to_numpy()[0:20,:1000]

#sigs = data[10,:]
"""
fs = 120
t = np.linspace(0, 1, 1000, False)  # 1 second
sigs = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t) + np.sin(2*np.pi*30*t) # 10 Hz + 20 Hz + 30 Hz

sigsLowpass = butter_lowpass_filter(sigs, 25, fs, order=5) # allow everything below 25 Hz
sigsBandpass = butter_bandpass_filter(sigs, 15, 25, fs, order=5) # allow in between 15 Hz & 25 Hz
sigsBandstop = butter_bandstop_filter(sigs, 15, 25, fs, order=5) # block in between 15 Hz & 25 Hz
sigsHighpass = butter_highpass_filter(sigs, 15, fs, order=5) # allow everything above 15 Hz
sigsRecover = sigsBandpass + sigsBandstop # Combine Bandpass & Bandstop to recover original signal



plt.figure(figsize=(12,5))
plt.title('raw signals')
plt.plot(sigs)
plt.show()

plt.figure(figsize=(12,5))
plt.title('sigsLowpass')
plt.plot(sigsLowpass)
plt.show()

plt.figure(figsize=(12,5))
plt.title('sigsBandpass')
plt.plot(sigsBandpass)
plt.show()

plt.figure(figsize=(12,5))
plt.title('sigsBandstop')
plt.plot(sigsBandstop)
plt.show()

plt.figure(figsize=(12,5))
plt.title('sigsHighpass')
plt.plot(sigsHighpass)
plt.show()

plt.figure(figsize=(12,5))
plt.title('sigsBandpass + sigsBandstop')
plt.plot(sigsRecover)
plt.show()
#"""

def run():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 60.0
    lowcut = 15.0
    highcut = 25.0

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6, 9]:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')

    # Filter a noisy signal.
    T = 1
    nsamples = 100 #int(T * fs)
    t = np.linspace(0, T, nsamples, endpoint=False)
    a = 0.02
    f0 = 20.0
    x = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t) + np.sin(2*np.pi*30*t) # 10 Hz + 20 Hz + 30 Hz
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Noisy signal')

    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
    plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.show()


run()