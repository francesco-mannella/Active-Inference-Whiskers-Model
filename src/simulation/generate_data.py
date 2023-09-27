import sys

sys.path.append('Code')

import src.simulation.run_simulation as run
from src.simulation.params import Params
from src.analysis.stats.fourier_transform import band_filter
from src.analysis.stats.crosscorr import (
    compute_correlation_desynchronization,
)
from src.params import datasetpath
from src.tools import make_simulation_gif_video

import pandas as pd
import numpy as np
import regex as re
import copy, os, sys

# %%


def run_trial(
    sid,
    seed,
    noise,
    obj,
    visual_trial,
    touch_trial,
    trial,
    filter_frequency_bounds,
    param_dict=None,
    log=True,
    decision_db=False,
    plot_frames=False,
):
    """Runs a single trial of a simulation.

    Parameters
    ----------
    sid : int
        Subject ID.
    seed : int
        Random seed.
    noise : float
        Noise level.
    obj : int
        Object ID.
    visual_trial : bool
        Whether the trial is a visual trial.
    touch_trial : bool
        Whether the trial is a touch trial.
    trial : int
        Trial number.
    filter_frequency_bounds : tuple
        Frequency bounds for filtering.
    param_dict : dict, optional
        Dictionary of parameters.
    log : bool, optional
        Whether to log the trial.
    decision_db : bool, optional
        Whether to return a decision database.
    plot_frames : bool, optional
        Whether to plot the frames.

    Returns
    -------
    db : pandas.DataFrame
        Database of the trial.
    db_decision : pandas.DataFrame, optional
        Database of the trial decisions.
    """

    params = None
    if param_dict is not None:
        params = Params()
        for key, value in param_dict.items():
            setattr(params, key, value)

    db_orig, frames = run.run_simulation(
        obj,
        visual_trial,
        touch_trial,
        noise=noise,
        plot=plot_frames,
        jupyter=False,
        seed=seed,
        params=copy.deepcopy(params),
        log=log,
    )
    

    db = run.build_df(
        db_orig,
        mod='desynchronization',
        pred=True,
        filter_frequency_bounds=filter_frequency_bounds,
        params=params,
    )

    if decision_db:
        db_decision = run.build_df1(db_orig, params=params)

    db['touches_in_ts'] = (
        db.groupby('ts')['touch']
        .apply(lambda x: pd.Series(sum(x) * np.ones_like(x)))
        .to_numpy()
    )
    db['object'] = obj
    db['trial_id'] = f'{seed:08d}'
    type_ = (
        'T'
        if touch_trial is True and visual_trial is False
        else 'M'
        if touch_trial is True and visual_trial is True
        else 'V'
    )

    db['type'] = type_
    tid = f'{type_}{obj:02d}-{sid}-{seed:08d}-{trial:04d}'
    db.loc[db.Direction == 'right', 'postid'] = tid + '_right'
    db.loc[db.Direction == 'left', 'postid'] = tid + '_left'

    db['tid'] = tid
    db['sid'] = sid
    db['ts'] = db['ts']
    db = db[
        [
            'sid',
            'trial_id',
            'tid',
            'type',
            'Direction',
            'postid',
            'ts',
            'Whiskers',
            'object',
            'touch',
            'touches_in_ts',
            'Angle',
            'Filtered_Angle',
            'Desynchronization',
            'PredAmpl',
            'PredAngle',
            'PredError',
        ]
    ]


    if not frames is None:
        make_simulation_gif_video(
            folder='.',
            root=tid,
            frames=frames,
        )

    if decision_db:
        return db, db_decision
    return db


def sim(
    mod='desynchronization',
    trials=1,
    seed=None,
    sid='',
    noise=0,
    filter_frequency_bounds=(4, 12),
    params=None,
    save_data=True,
    decision_db=False,
    plot_frames=False,
    partial=None,
):
    """
    This function runs a simulation of the given model with the given parameters.

    Parameters
    ----------
    mod : str, optional
        The model to simulate. Default is 'desynchronization'.
    trials : int, optional
        The number of trials to run. Default is 1.
    seed : int, optional
        The seed to use for the simulation. Default is None.
    sid : str, optional
        The session ID for the simulation. Default is ''.
    noise : float, optional
        The amount of noise to add to the simulation. Default is 0.
    filter_frequency_bounds : tuple, optional
        The frequency bounds to use for the filter. Default is (4, 12).
    params : dict, optional
        The parameters to use for the simulation. Default is None.
    save_data : bool, optional
        Whether or not to save the data from the simulation. Default is True.
    decision_db : bool, optional
        Whether or not to save the decision data from the simulation. Default is False.
    plot_frames : bool, optional
        Whether or not to plot the frames from the simulation. Default is False.
    partial : list, optional
        The list of tuples to use for partial simulation. Default is None.

    Returns
    -------
    data : pandas.DataFrame
        The data from the simulation.
    ddata : pandas.DataFrame
        The decision data from the simulation.
    """
    if seed is None:
        seed = np.frombuffer(os.urandom(4), dtype=np.uint32)[0]


    if sid == '':
        sid = f'{trials:04d}-{seed:08d}-{sid}'

    data = []
    ddata = []
    s = 0


    for trial in range(trials):
        for obj in [0, 1]:
            for visual_trial in [True, False]:
                for touch_trial in [True]:
                    
                    if partial is not None:
                        if not (visual_trial, obj) in [tuple(x) for x in partial]:
                            continue

                    dbo = run_trial(
                        sid,
                        seed,
                        noise,
                        obj,
                        visual_trial,
                        touch_trial,
                        trial,
                        filter_frequency_bounds=filter_frequency_bounds,
                        param_dict=copy.deepcopy(params),
                        decision_db=decision_db,
                        plot_frames=plot_frames,
                    )

                    if decision_db:
                        db, ddb = dbo
                        ddb["tid"] = db.tid.unique()[0]
                        ddb["type"] = db.type.unique()[0]
                        ddb["object"] = db.object.unique()[0]
                    else:
                        db = dbo

                    if params is not None:
                        for param, param_value in params.items():
                            if np.isscalar(param_value):
                                db[param] = param_value

                    data.append(db)
                    if decision_db: ddata.append(ddb)
                    print(db['tid'].unique()[0])
        seed += 1

    data = pd.concat(data)
    if decision_db: ddata = pd.concat(ddata)
    if save_data:
        data.to_csv(datasetpath / f'sim-{sid}_filtered_df.csv')
    return data if not decision_db else data, ddata


if __name__ == '__main__':

    # import necessary libraries
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import matplotlib

    # set matplotlib to use QtAgg
    matplotlib.use('QtAgg')

    # set parameters
    params = {
        'eta_a': 0.02,
        'cicles': 8,
        'relative_time_switched_on': 0,
        'stime': 9000,
    }

    # run trial
    df = run_trial(
        sid='200',
        seed=1,
        noise=0.02,
        obj=1,
        visual_trial=True,
        filter_frequency_bounds=(10, 40),
        touch_trial=True,
        trial=True,
        param_dict=params,
    )

    # create figure
    fig = plt.figure()

    # add first subplot
    ax = fig.add_subplot(211)
    sns.lineplot(
        ax=ax,
        data=df,
        x='ts',
        y='Angle',
        hue='Whiskers',
        palette='rocket',
    )

    # add second subplot
    ax = fig.add_subplot(212)
    sns.lineplot(
        ax=ax,
        data=df,
        x='ts',
        y='Desynchronization',
        hue='Whiskers',
        palette='rocket',
    )

    # show plot
    plt.show()


class SeriesAnalysis:
    """A class for analyzing a series of data points"""

    @staticmethod
    def maxmins(series):
        """Calculates the maxmins of a series of data points"""
        derivatives = np.diff(series, prepend=0)
        deriv_signs = np.sign(derivatives)
        mms = np.sign(np.diff(deriv_signs))
        return mms

    @staticmethod
    def maxima(series):
        """Calculates the maxima of a series of data points"""
        mms = SeriesAnalysis.maxmins(series)
        maxima = -np.hstack([np.minimum(0, mms), 0]) 
        return maxima

    @staticmethod
    def minima(series):
        """Calculates the minima of a series of data points"""
        mms = SeriesAnalysis.maxmins(series)
        maxima = np.hstack([np.maximum(0, mms), 0]) 
        return maxima
