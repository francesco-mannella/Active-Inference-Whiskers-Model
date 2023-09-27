import numpy as np
import pandas as pd
from scipy.fftpack import rfft, irfft, rfftfreq


def fft_filter(signal, h=0.001, min_freq=1, max_freq=12):
    """Bandpass filter using fast fourier transform

    Parameters:
      signal (array): The time series to be filtered.
      h (float): The integration time to be used.
      min_freq (float): Lower bound
      max_freq (float): Upper bound
    """

    # fourier transform of the signal
    f_signal = rfft(signal)

    # frequency index of each armonic
    freqs = rfftfreq(len(signal), h)

    # cut out-of bounds frequencies from
    # the transform
    cut_f_signal = f_signal.copy()
    cut_f_signal[freqs < min_freq] = 0
    cut_f_signal[freqs >= max_freq] = 0

    # fourier anti-transform - we finally
    # get the filtered signal
    cut_signal = irfft(cut_f_signal)

    return cut_signal, (f_signal ** 2) / h, freqs


def filtered(values, mode, h=0.001):
    """Filter a time series within one of three predefined groups"""

    if type(mode) == str:
        bounds = {
            "low": (0, 15),
            "medium": (15, 30),
            "high": (30, 60),
        }[mode]
    elif  type(mode) == tuple:
       bounds = mode 

    fltrd, psd, freqs = fft_filter(
        values, h=h, min_freq=bounds[0], max_freq=bounds[1]
    )
    return fltrd

def smooth_boundaries(x, prop=0.5):
    n = len(x)
    g = np.exp(-0.5*np.linspace(-3, 3, int(n*prop) ))
    g /= g.sum()
    mask = np.ones(n)
    mask = np.convolve(mask, g, mode="same")
    res = mask*x
    return res

def filter_whisk(d, value, mode, h=0.001):
    series = filtered(d[value].to_numpy(), mode, h=h)
    series = smooth_boundaries(series)
    d[value] = series
    return d 

def add_speed(data, value):

    x = np.diff(data[value].to_numpy(), prepend=0)
    x[:5] = np.mean(x[5:10])
    data["speed"] = x
    return data


def errors(data, value):
    x = data[value].to_numpy()
    x = (np.reshape(x, [-1, 1]) - (np.reshape(x, [1, -1])))**2
    x = x.mean(0)
    # th = x.min() + (x.max()-x.min())*0.4
    # x = np.maximum(0, x - th)
    data["errors"] = x
    return data


def band_filter(series, t="ts", variable="variable", value="value",
        mode="high", h=0.001, s=0.5):

    fvalue = "Filtered_Angle"
    fltrd = series
    fltrd[fvalue] = fltrd.groupby(variable).apply(lambda x: filter_whisk(x, value, mode, h=h))[value]
    fltrd["desynchronization"] = fltrd.groupby(t).apply(lambda x: errors(x, value))["errors"]   

    return fltrd

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    plt.close("all")

    t = np.linspace(0, 1, 1000)
    x = np.sin(2 * np.pi * t)
    x += np.cos(36 * np.pi * t)
    plt.figure()
    plt.plot(t, x)
    fltrd, psd, freqs = fft_filter(x, h=0.001, min_freq=0, max_freq=2)
    plt.plot(t, fltrd)

    fltrd, psd, freqs = fft_filter(x, h=0.001, min_freq=2, max_freq=40)
    plt.plot(t, fltrd)
    plt.show()
    plt.figure()
    plt.plot(np.arange(40), psd[:40])

    plt.figure()

    plt.plot(filtered(x, "high"))

    plt.show()
    
    
