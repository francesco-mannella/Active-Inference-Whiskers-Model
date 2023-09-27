# TODO: replace with crosscorr_new
import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d


# This code checks if the signal module has an attribute called
# "correlation_lags". If it does not, it defines a function called
# "correlation_lags" and adds it to the signal module as an attribute. The
# function takes three parameters: in1_size, in2_size, and mode. It creates two
# arrays, x and y, and returns the concatenation of the two arrays.
if not hasattr(signal, "correlation_lags"):
    def correlation_lags(in1_size, in2_size, mode="full"):
        x = -np.arange(in2_size)[::-1]
        y = np.arange(in1_size)

        return np.hstack([x, y[1:]])

    signal.correlation_lags = correlation_lags


def whiten(*args):
    """Standardize a group of time-series by subtracting the mean and dividing by the standard deviation"""
    data = np.vstack([x for x in args])

    means = np.mean(data, 1).reshape(-1, 1)
    stds = np.std(data, 1).reshape(-1, 1)
    stds[stds == 0] = 1

    return (data - means) / stds


def compute_crosscorrelation(x, y):
    """Compute the crosscorrelation between two time-series

    Parameters:
        x: the first time-series
        y: the second time-series

    Returns:
        lag: the lag of the max-peak corelation
        corr: the max-peak correlation
        corrs: all lagged correlations

    """

    x, y = whiten(x, y)

    n = len(x)
    corr = signal.correlate(x, y, mode="full")
    lags = signal.correlation_lags(len(x), len(y), mode="full")
    if np.max(corr) > 0: corr /= np.max(corr)
    lags = lags / n
    idxmax = np.argmax(corr)
    return lags[idxmax], corr[idxmax], corr


def roll_array(data, win_size, win_num):
    """Split a dataset into windowed datasets

    Parameters:
        data: the dataset to be split (by rows)
        win_size: the length of each window (number of rows)
        win_num: te total amount of windows (can be overlapped)

    Returns:
        rolled_data: A list of windowed datasets
        idcs: an array of the indices of window centres


    Example:

        rolled_data, idcs = roll_array(np.arange(20), 3, 4)

        rolled_data:
            [array([0, 1, 2]),
            array([4, 5, 6]),
            array([ 8,  9, 10]),
            array([12, 13, 14]),
            array([16, 17, 18])]
        idcs:
            array([ 0,  4,  8, 12, 16])


    """
    data_size = len(data)
    if data_size < (win_size + win_num):
        raise "The size of the window is big or you requested too much windows"

    last = data_size - win_size
    gap = last // win_num
    idcs = np.arange(0, last + 1, gap)
    rolled = [data[x: (x + win_size)] for x in idcs]
    return rolled, idcs


def moving_crosscorrelation(x, y, win_size, win_num, corr_th=0.5):
    """Compute the max lagged correlation over a number of rolled windows of
        a dataset of two timeseries

    Parameters:
        x: the first time-series
        y: the second time-series
        win_size: the length of each window
        win_num: te total amount of windows (can be overlapped)
        corr_th: a minimum correlation below which lag is allways considered 0

    Returns:
        lag_and_corrs: a dataset of four rows:
            1: indices of window centres
            2: lags filtered by correlation (0 if below corr_th)
            3: lags
            4: correlations at peak

    """

    data = np.vstack([x, y]).T.copy()

    rolled_data, idcs = roll_array(data, win_size, win_num)

    # comute crosscorr on each window
    lags_and_corrs = np.vstack([compute_crosscorrelation(*batch.T)[:2] for batch in rolled_data])

    # add lag filtered by corr threshold
    lags, corrs = lags_and_corrs.T
    lags_and_corrs = np.hstack(
        [np.reshape(lags * (corrs > corr_th), (-1, 1)), lags_and_corrs]
    )

    # add indices of crosscorr centres
    lags_and_corrs = np.hstack(
        [np.reshape(idcs, (-1, 1)) + win_size // 2 + 1, lags_and_corrs]
    )

    return lags_and_corrs.T



def crosscorrs(ts: np.ndarray, x1: np.ndarray, x2: np.ndarray,
                            win_num, ws, corr_th):
    """
    Calculates the cross-correlation between two arrays x1 and x2, using a moving window of size ws and win_num windows.
    The correlation threshold is set to corr_th. The function returns an array of values interpolated from the moving cross-correlation.
    
    Parameters:
    ts (np.ndarray): array of timestamps
    x1 (np.ndarray): array of values
    x2 (np.ndarray): array of values
    win_num (int): number of windows
    ws (int): window size
    corr_th (float): correlation threshold
    
    Returns:
    values (np.ndarray): array of interpolated values
    """
    n = x1.size
    lags_and_corrs = moving_crosscorrelation(x1, x2, win_num=win_num, win_size=ws)
    idcs, clags, *_ = lags_and_corrs

    # This code creates a function 'f' that interpolates the values in the
    # 'clags' array using the corresponding indices in the 'idcs' array. The
    # 'values' array is then populated with the interpolated values from the
    # 'ts' array.
    f = interp1d(idcs, clags, bounds_error=False, fill_value=np.nan)
    values = f(ts)

    values[idcs.astype(int)] = clags
    return values

def compute_correlation_desynchronization(
        ts: np.ndarray, desynchronization: np.ndarray, touch: np.ndarray,
        win_gap: float = 0.02,
        win_size: float = None,
        corr_th: float = 0.5) -> np.ndarray:
    """Compute correlation between desynchronization and touch data.

    Args:
        ts: np.ndarray [N-timepoints] timestamps for the data
        desynchronization: np.ndarray [N-timepoints] 'difference' data for whiskers
        touch: np.ndarray [N-timepoints] 'touch' data for whiskers
        win_gap: Gap between windows in seconds.
        win_size: Size of windows in seconds. If None, defaults to 1/5 of the total length.
        corr_th: Correlation threshold.

    Returns:
        np.ndarray: Correlation values.
    """
    l = len(desynchronization)
    dt = np.diff(ts)[0] / 2

    wn = int(l // (win_gap / dt))
    if win_size is None:
        ws = l // 5
    else:
        ws = int(win_size / dt)

    corrs = crosscorrs(
            ts,
            desynchronization, 
            touch,
            win_num=wn, ws=ws, corr_th=corr_th)

    return corrs

