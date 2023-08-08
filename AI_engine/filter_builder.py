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

# Highpass filters

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def cheby1_highpass(cutoff, fs, max_rip=5, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.cheby1(N = order, rp = max_rip, Wn = normal_cutoff, btype='high', analog=False)
    return b, a

def cheby1_highpass_filter(data, max_ripple, cutoff, fs, order=5):
    b, a = cheby1_highpass(cutoff=cutoff, max_rip = max_ripple, fs=fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def cheby2_highpass(cutoff, fs, min_attn=5, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.cheby2(N = order, rs = min_attn, Wn = normal_cutoff, btype='high', analog=False)
    return b, a

def cheby2_highpass_filter(data, min_attenuation, cutoff, fs, order=5):
    b, a = cheby2_highpass(cutoff=cutoff, min_attn = min_attenuation, fs=fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

# Lowpass filters

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def cheby1_lowpass(cutoff, fs, max_rip=5, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.cheby1(N = order, rp = max_rip, Wn = normal_cutoff, btype='low', analog=False)
    return b, a

def cheby1_lowpass_filter(data, max_ripple, cutoff, fs, order=5):
    b, a = cheby1_lowpass(cutoff=cutoff, max_rip = max_ripple, fs=fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def cheby2_lowpass(cutoff, fs, min_attn=5, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.cheby2(N = order, rs = min_attn, Wn = normal_cutoff, btype='low', analog=False)
    return b, a

def cheby2_lowpass_filter(data, min_attenuation, cutoff, fs, order=5):
    b, a = cheby2_lowpass(cutoff=cutoff, min_attn = min_attenuation, fs=fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

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

def cheby1_bandpass(lowcut, highcut, fs, max_rip=5, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.cheby1(N = order, rp = max_rip, Wn = [low, high], btype='band', analog=False)
    return b, a

def cheby1_bandpass_filter(data, max_ripple, lowcut, highcut, fs, order=5):
    b, a = cheby1_bandpass(lowcut=lowcut, highcut=highcut, max_rip = max_ripple, fs=fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def cheby2_bandpass(lowcut, highcut, fs, min_attn=5, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.cheby2(N = order, rs = min_attn, Wn = [low, high], btype='band', analog=False)
    return b, a

def cheby2_bandpass_filter(data, min_attenuation, lowcut, highcut, fs, order=5):
    b, a = cheby2_bandpass(lowcut=lowcut, highcut=highcut, min_attn = min_attenuation, fs=fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

# Bandstop filters
def butter_bandstop(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    i, u = signal.butter(order, [low, high], btype='bandstop')
    return i, u

def butter_bandstop_filter(data, lowcut, highcut, fs, order=5):
    i, u = butter_bandstop(lowcut, highcut, fs, order=order)
    y = signal.lfilter(i, u, data)
    return y

def cheby1_bandstop(lowcut, highcut, fs, max_rip=5, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.cheby1(N = order, rp = max_rip, Wn = [low, high], btype='bandstop', analog=False)
    return b, a

def cheby1_bandstop_filter(data, max_ripple, lowcut, highcut, fs, order=5):
    b, a = cheby1_bandstop(lowcut=lowcut, highcut=highcut, max_rip = max_ripple, fs=fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def cheby2_bandstop(lowcut, highcut, fs, min_attn=5, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.cheby2(N = order, rs = min_attn, Wn = [low, high], btype='bandstop', analog=False)
    return b, a

def cheby2_bandstop_filter(data, min_attenuation, lowcut, highcut, fs, order=5):
    b, a = cheby2_bandstop(lowcut=lowcut, highcut=highcut, min_attn = min_attenuation, fs=fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

