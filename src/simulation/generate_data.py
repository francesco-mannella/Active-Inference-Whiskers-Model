import sys

import simulation.run_simulation as run
from simulation.params import Params
from analysis.stats.fourier_transform import band_filter
from analysis.stats.crosscorr import (
    compute_correlation_desynchronization_touch,
)
from utils.params import datasetpath

import pandas as pd
import numpy as np
import regex as re


def run_trial(
    sid,
    seed,
    noise,
    obj,
    visual_trial,
    touch_trial,
    trial,
    param_dict=None,
    log=True,
):
 
    params = None
    if param_dict is not None:
        params = Params()
        for key, value in param_dict.items():
            setattr(params, key, value)

    db, _ = run.run_simulation(
        obj,
        visual_trial,
        touch_trial,
        noise=noise,
        plot=False,
        jupyter=False,
        seed=seed,
        params=params,
        log=log,
    )

    db = run.build_df(
        db,
        filtered=True,
        mod="desynchronization",
        pred=True,
        win_gap=0.01,
        win_size=0.8,
        params = params,
    )

    db["touches_in_ts"] = (
        db.groupby("ts")["touch"]
        .apply(
            lambda x: pd.Series(sum(x) * np.ones_like(x))
        )
        .to_numpy()
    )
    db["object"] = obj
    db["trial_id"] = f"{seed:08d}"
    type_ = (
        "T"
        if touch_trial is True and visual_trial is False
        else "M"
        if touch_trial is True and visual_trial is True
        else "V"
    )

    db["type"] = type_
    tid = f"{type_}{obj:02d}-{seed:08d}-{trial:04d}"
    db.loc[db.Direction == "right", "postid"] = (
        tid + "_right"
    )
    db.loc[db.Direction == "left", "postid"] = tid + "_left"

    db["tid"] = tid
    db["sid"] = sid
    db["ts"] = db["ts"]
    db = db[
        [
            "sid",
            "trial_id",
            "tid",
            "type",
            "Direction",
            "postid",
            "ts",
            "Whiskers",
            "object",
            "touch",
            "touches_in_ts",
            "Angle",
            "Filtered_Angle",
            "Desynchronization",
            "PredAngle",
            "PredError",
        ]
    ]
    return db


def sim(mod="desynchronization", trials=1, seed=0, sid="", noise=0, params=None):

    seed = seed * 1000

    sid = f"{trials:04d}-{seed:08d}-{sid}"

    data = []
    s = 0
    for obj in [1]:
        for visual_trial in [True, False]:
            for touch_trial in [True]:
                for trial in range(trials):

                    db = run_trial(
                        sid,
                        seed,
                        noise,
                        obj,
                        visual_trial,
                        touch_trial,
                        trial,
                        param_dict = params,
                    )

                    data.append(db)
                    print(db["tid"].unique()[0])
                    seed += 1

    data = pd.concat(data)
    data.to_csv(datasetpath / f"sim-{sid}_filtered_df.csv")
    return data


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("QtAgg")

    params = {"eta_a":0.02, "cicles": 8,  "relative_time_switched_on": 0, "stime": 9000 }
    df = run_trial(sid="200", seed=1, noise=0.02, obj=1, visual_trial=False, 
            touch_trial=True, trial=True, param_dict=params )
    
    fig = plt.figure()
    ax = fig.add_subplot(211)
    sns.lineplot(
        ax=ax,
        data=df,
        x='ts',
        y='Angle',
        hue='Whiskers',
        palette='rocket',
    )
    ax = fig.add_subplot(212)
    sns.lineplot(
        ax=ax,
        data=df,
        x='ts',
        y='Desynchronization',
        hue='Whiskers',
        palette='rocket',
    )
    plt.show()
