# TODO: replace with crosscorr_new
import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d


if not hasattr(signal, "correlation_lags"):
    def correlation_lags(in1_size, in2_size, mode="full"):
        x = -np.arange(in2_size)[::-1]
        y = np.arange(in1_size)

        return np.hstack([x, y[1:]])


    signal.correlation_lags = correlation_lags


def whiten(*args):
    """Standardize together a group of time-series"""
    data = np.vstack([x for x in args])

    means = np.mean(data, 1).reshape(-1, 1)
    stds = np.std(data, 1).reshape(-1, 1)
    stds[stds == 0] = 1

    return (data - means) / stds


def max_corr_gap(x, y):
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


def moving_max_corr_gap(x, y, win_size, win_num, corr_th=0.5):
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
    lags_and_corrs = np.vstack([max_corr_gap(*batch.T)[:2] for batch in rolled_data])

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



def crosscorr_whisker_touch(ts: np.ndarray, diff: np.ndarray, touch: np.ndarray,
                            win_num, ws, corr_th):
    """
    TODO - this function was 'crosscorrs'
    Args:
        ts:
        diff:
        touch:
        win_num:
        ws:
        corr_th:

    Returns:

    """
    n = diff.size
    lags_and_corrs = moving_max_corr_gap(diff, touch, win_num=win_num, win_size=ws)
    idcs, clags, *_ = lags_and_corrs
    f = interp1d(idcs, clags, bounds_error=False, fill_value=np.nan)
    values = f(ts)
    values[idcs.astype(int)] = clags
    return values


def compute_correlation_desynchronization_touch(
        ts, desynchronization, touch,
        win_gap=0.02,
        win_size=None,
        corr_th=0.5,):
    """
    TODO: this was compute_corrs_in_series

    Args:
        ts: np.ndarray [N-timepoints] timestamps for the data
        desynchronization: np.ndarray [N-timepoints] 'difference' data for whiskers
        touch: np.ndarray [N-timepoints] 'touch' data for whiskers
        win_gap: ?
        win_size: ?
        dt: ?
        corr_th: ?

    Returns:

    """
    l = len(desynchronization)
    # TODO: dt was hardcoded at 0.001, however the difference
    # between timepoins is 0.002 in the data, should it be
    # np.diff(ts) or np.diff(ts) / 2?
    dt = np.diff(ts)[0] / 2

    wn = int(l // (win_gap / dt))
    if win_size is None:
        ws = l // 5
    else:
        ws = int(win_size / dt)

    corrs = crosscorr_whisker_touch(
            ts,
            desynchronization, 
            touch,
            win_num=wn, ws=ws, corr_th=corr_th)

    return corrs

