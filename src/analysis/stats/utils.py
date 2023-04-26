import sys
import numpy as np
from scipy.interpolate import interp1d
import scipy.stats as sts

def trim_nans(data):
    d = np.copy(data)
    d[np.isnan(d)] = 0
    m = np.abs(np.diff(d)).mean()*40
    df = np.diff(d, prepend=0)
    start_idcs = np.argwhere(df > m).ravel()
    start = None if len(start_idcs) == 0 else start_idcs[0]
    end_idcs = np.argwhere(df < -m).ravel() 
    end = None if len(end_idcs) == 0 else end_idcs[-1]
    return start, end


def interpolate_whisker_angle(d, value="value", filter_len=20):
    ''' Fill gaps in a series with predictions coming from a local gaussian fitting
        of the known points
    Parameters:
        d: The time-series (1st colunm: ts, 2nd column: values)
        value: name of the value variable
    Returns:
        The filtered time series    
    '''    

    start, end = trim_nans(d[value])
    if start is None: start = d.index[0]    
    if end is None: end = d.index[-1]    
    idcs = d.index 
    # select items with value NaN
    nans = d[d[value].isnull()].index
    # select items with value not NaN
    nonans = d[~d[value].isnull()].index
   
    idcs = idcs[(idcs >=start) & (idcs <=end)]
    nans = nans[(nans >=start) & (nans <=end)]
    nonans = nonans[(nonans >=start) & (nonans <=end)]
    
    if nonans.shape[0] > 100:

        x, y = d.loc[nonans, ["ts", value]].to_numpy().T
        
        predict = interp1d(x, y, kind="nearest", bounds_error=False)
        d.loc[nans, value] = predict(d.loc[nans, "ts"].to_numpy())

        d.loc[idcs, value] = np.convolve(
                d.loc[idcs, value].to_numpy(),  
                np.ones(filter_len)/filter_len,
                "same"
                )
    else:
        d[value] = np.nan
    
    return d

# %%
def kde(d, x = "ts", value="touch", cut=False, kde_factor=0.1, min_kde=1e-3):
    
    # find timestamps related to touches  
    ts = d["ts"].iloc[:].to_numpy()
    tts = d[d[f"{value}"]>0]["ts"].iloc[:].to_numpy()
    
    # convolve the timeseries of touches 
    ma = np.convolve(d[value], np.ones(10)/10, "same")
    d["touch_ma"] = ma

    # exclude touches which are too rare (minimal density) 
    if tts.sum() > 1 and cut == True:
        d.loc[ma < 0.1, touch] = 0

    # density
    # compute again touch timestamps
    tts = d[d[f"{value}"]>0]["ts"].iloc[:].to_numpy()
    try:

        # gaussian kernel density
        fit = sts.gaussian_kde(tts, kde_factor)
        tkde = fit.evaluate(ts)

        # derivative of kde
        der = np.diff(tkde, prepend=0)
        signder = np.sign(der)
        sder = np.maximum(0, np.sign(np.diff(signder, prepend=0)))

        # divide in chunks based on kde increment onsets
        tchunks = np.cumsum(sder).astype(float)
        tchunks[tkde < min_kde] = 1e200
        tchunks -= np.min(tchunks)
        tchunks[tchunks==1e200] = np.nan
        d["touch_kde"] = tkde
        d["touch_chunks"] = tchunks

    except ValueError as e:

        exc_type, exc_obj, exc_tb = sys.exc_info()
        #print(f"{__file__}:{exc_tb.tb_lineno}  {e}")

        d["touch_kde"] = 0
        d["touch_chunks"] = 0

    return d


class SeriesAnalysis:

    @staticmethod
    def maxmins(series):
        derivatives = np.diff(series, prepend=0)
        deriv_signs = np.sign(derivatives)
        mms = np.sign(np.diff(deriv_signs))
        return mms

    @staticmethod
    def maxima(series):
        mms = SeriesAnalysis.maxmins(series)
        maxima = -np.hstack([np.minimum(0, mms), 0]) 
        return maxima

    @staticmethod
    def minima(series):
        mms = SeriesAnalysis.maxmins(series)
        maxima = np.hstack([np.maximum(0, mms), 0]) 
        return maxima

