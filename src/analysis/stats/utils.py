import sys
import numpy as np
from scipy.interpolate import interp1d
import scipy.stats as sts


def trim_nans(data):
    d = np.copy(data)
    d[np.isnan(d)] = 0
    m = np.abs(np.diff(d)).mean() * 40
    df = np.diff(d, prepend=0)
    start_idcs = np.argwhere(df > m).ravel()
    start = None if len(start_idcs) == 0 else start_idcs[0]
    end_idcs = np.argwhere(df < -m).ravel()
    end = None if len(end_idcs) == 0 else end_idcs[-1]
    return start, end


def interpolate_whisker_angle(d, value='value', filter_len=20):
    """Fill gaps in a series with predictions coming from a local gaussian fitting
        of the known points
    Parameters:
        d: The time-series (1st colunm: ts, 2nd column: values)
        value: name of the value variable
    Returns:
        The filtered time series
    """

    start, end = trim_nans(d[value])
    if start is None:
        start = d.index[0]
    if end is None:
        end = d.index[-1]
    idcs = d.index
    # select items with value NaN
    nans = d[d[value].isnull()].index
    # select items with value not NaN
    nonans = d[~d[value].isnull()].index

    idcs = idcs[(idcs >= start) & (idcs <= end)]
    nans = nans[(nans >= start) & (nans <= end)]
    nonans = nonans[(nonans >= start) & (nonans <= end)]

    if nonans.shape[0] > 100:

        x, y = d.loc[nonans, ['ts', value]].to_numpy().T

        predict = interp1d(x, y, kind='nearest', bounds_error=False)
        d.loc[nans, value] = predict(d.loc[nans, 'ts'].to_numpy())

        d.loc[idcs, value] = np.convolve(
            d.loc[idcs, value].to_numpy(),
            np.ones(filter_len) / filter_len,
            'same',
        )
    else:
        d[value] = np.nan

    return d


def find_maxima(time_series, convs=10):
    """
    Find the local maxima of a time series.

    Parameters
    ----------
    time_series : list
        A list of values representing a time series.
    convs : int, optional
        The number of convolutions to perform (default is 10).

    Returns
    -------
    maxima_indices : list
        A list of indices of the local maxima in the time series.
    maxima_values : list
        A list of values of the local maxima in the time series.
    """
    # Convert time series to numpy array
    time_series = np.array(time_series)
    size = len(time_series) // 50
    for k in range(convs):
        time_series = np.convolve(
            time_series, np.ones(size) / size, mode='same'
        )

    # Find the indices of local maxima
    maxima_indices = (
        np.where(
            (time_series[1:-1] > time_series[:-2])
            & (time_series[1:-1] > time_series[2:])
        )[0]
        + 1
    )

    # Return the values at the maxima indices
    maxima_values = time_series[maxima_indices]

    return maxima_indices, maxima_values


# %%
def kde(d, x='ts', value='touch', cut=False, kde_factor=0.1, min_kde=1e-3, touch_chunks_oscill=True):
    """
    Computes a kernel density estimation (KDE) of the given data.
    
    Parameters
    ----------
    d : pandas.DataFrame
        DataFrame containing the data to be analyzed.
    x : str, optional
        Name of the column containing the timestamps.
    value : str, optional
        Name of the column containing the values to be analyzed.
    cut : bool, optional
        Whether to exclude touches which are too rare (minimal density).
    kde_factor : float, optional
        Factor used to compute the KDE.
    min_kde : float, optional
        Minimum KDE value.
    touch_chunks_oscill : bool, optional
        Whether to divide the data into chunks based on the oscillations of the data.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the KDE and the chunks of the data.
    """

    # find timestamps related to touches
    ts = d['ts'].iloc[:].to_numpy()
    tts = d[d[f'{value}'] > 0]['ts'].iloc[:].to_numpy()

    # convolve the timeseries of touches
    ma = np.convolve(d[value], np.ones(10) / 10, 'same')
    d['touch_ma'] = ma

    # # exclude touches which are too rare (minimal density)
    # if tts.sum() > 1 and cut == True:
    #     d.loc[ma < 0.1, touch] = 0
    #
    if touch_chunks_oscill == False:
        try:
            # density
            # compute again touch timestamps
            tts = d[d[f'{value}'] > 0]['ts'].iloc[:].to_numpy()

            # gaussian kernel density
            fit = sts.gaussian_kde(tts, kde_factor)
            tkde = fit.evaluate(ts)
            d['touch_kde'] = tkde

            # derivative of kde
            der = np.diff(tkde, prepend=0)
            signder = np.sign(der)
            sder = np.maximum(0, np.sign(np.diff(signder, prepend=0)))

            # divide in chunks based on kde increment onsets
            tchunks = np.cumsum(sder).astype(float)
            tchunks[tkde < min_kde] = 1e200
            tchunks -= np.min(tchunks)
            tchunks[tchunks == 1e200] = np.nan
            d['touch_chunks'] = tchunks

        except ValueError as e:

            d['touch_kde'] = 0
            d['touch_chunks'] = -2

    else:
        d['touch_kde'] = 0

        oscill = d.groupby('ts')['Angle'].mean().to_numpy()
        idcs, _ = SeriesAnalysis.maxima(oscill)
        chunks = np.zeros_like(oscill)
        if len(tts) > 0:
            chunks[idcs] = 1
            chunks = np.cumsum(chunks)
            chunk0 = chunks[ts == tts[0]]
            chunks -= chunk0
        d['touch_chunks'] = chunks

    return d
