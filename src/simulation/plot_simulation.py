import numpy as np
import gym, RatSim
from simulation.ActiveInverenceModel import GM
from simulation.params import Params
from tools import make_simulation_gif_video
from simulation.run_simulation import build_df, build_df1
from analysis.stats.fourier_transform import band_filter
from analysis.stats.crosscorr import compute_correlation_desynchronization_touch
import pandas as pd
import seaborn as sns
import os

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as GS


def init_plt():
    LITTLE_SIZE = 6
    SMALL_SIZE = 8
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 11

    plt.rc("font", size=LITTLE_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_behav_data(data, pm=None):

    pm = Params() if pm is None else pm

    fig, ax = plt.subplots(5, 1, figsize=(8, 10))
    fig.tight_layout(pad=5)
    plt.close(fig)
    gs = GS(40, 80)
    palette = sns.color_palette("hls", 3)

    # ---- manage data into a pandas db ----

    ## Build with whisker labels
    labels = [f"whisk_{x}_{y}" for x in ["left", "right"] for y in range(3)]
    df = pd.DataFrame({l: k for k, l in zip(data["gp_x"].T, labels)})

    ## Add timestep index
    t = np.arange(len(data["gp_x"]))
    df["ts"] = t / 1000

    ## Melt into one dependent variable for angles
    df = df.melt(id_vars=["ts"], value_vars=labels)
    df = df.rename(columns={"variable": "Whiskers", "value": "Angle"})

    ## Add touch variable
    df["touch"] = np.nan
    for i, w in enumerate(labels):
        df.loc[df["Whiskers"] == w, "touch"] = data["touch_sensor"][:, i]

    # Compute crosscorr analysis in db
    df_flt = band_filter(df, variable="Whiskers", value="Angle", mode=(2, 5))
    df_flt["corrs"] = compute_correlation_desynchronization_touch(
            ts=df_flt["ts"].to_numpy(), 
            desynchronization=df_flt["desynchronization"].to_numpy(),
            touch=df_flt["touch"].to_numpy(),
            win_num=80)

    # data for level 1
    df1 = pd.DataFrame(
        {l: k for k, l in zip(data["gm_mu_y"].T, ["Object 1", "Object 2"])}
    )
    t = np.arange(len(data["gp_x"]))
    df1["ts"] = t / 1000
    df1 = df1.melt(id_vars=["ts"], value_vars=["Object 1", "Object 2"])
    df1 = df1.rename(columns={"variable": "Objects", "value": "Decision_prob"})

    # -------------------------------------

    # find timestep of begin of visual input
    vis = np.argwhere(data["obj"].mean(-1) > 0)
    if vis.shape[0] > 0:
        visual_time = vis[0][0]
    else:
        visual_time = -10
    visual_time /= 1000

    # find timestep of first touch
    touch = data["touch_sensor"].sum(-1)
    tch = np.argwhere(touch > 0)
    if tch.shape[0] > 0:
        tch_time = tch[0][0]
    else:
        tch_time = -10
    tch_time /= 1000

    # plot
    ax[0].cla()
    ax[0].set_title("Whiskers movements")
    ax[0].set_ylabel("Angle")

    sns.lineplot(
        ax=ax[0],
        data=df,
        x="ts",
        y="Angle",
        hue="Whiskers",
        # palette=palette, legend=True )
        legend=True,
    )

    sns.scatterplot(
        ax=ax[0],
        data=df.query("touch>0"),
        x="ts",
        y="Angle",
        hue="Whiskers",
        # palette=palette, legend=True )
        legend=True,
    )

    ax[0].plot([visual_time, visual_time], [-10, 10], lw=3, c="green")
    ax[0].plot([tch_time, tch_time], [-10, 10], lw=3, c="red")

    xmax = df["ts"].max()
    ax[0].set_xlim([-xmax * 0.1, xmax * 1.3])
    ax[0].set_ylim([-1.2, 1.5])
    ax[0].get_legend().remove()
    ax[0].set_xlabel("")
    ax[0].set_xticks([])
    ax[0].set_xticklabels([])
    ax[0].legend(labels=labels)

    # Subplot 2
    ax[1].cla()
    ax[1].set_title("Decision layer")
    ax[1].set_ylabel("Probability")
    sns.lineplot(
        ax=ax[1], data=df1, x="ts", y="Decision_prob", hue="Objects", legend=True
    )
    ax[1].plot([visual_time, visual_time], [0, 1], lw=3, c="green")
    ax[1].plot([tch_time, tch_time], [0, 1], lw=3, c="red")
    ax[1].set_xlim([-xmax * 0.1, xmax * 1.3])
    ax[1].get_legend().remove()
    ax[1].set_xlabel("")
    ax[1].set_yticks([0, 1])
    ax[1].set_xticks([])
    ax[1].set_xticklabels([])

    # Subplot 3
    ax[2].cla()
    ax[2].set_title("Whiskers filtered movements")
    ax[2].set_ylabel("Angle increment")
    sns.lineplot(
        ax=ax[2],
        data=df_flt,
        x="ts",
        y="Angle",
        hue="Whiskers",
    )  # palette=palette)
    ax[2].plot(
        [visual_time, visual_time],
        [df_flt["Angle"].min(), df_flt["Angle"].max()],
        lw=3,
        c="green",
    )
    ax[2].plot(
        [tch_time, tch_time],
        [df_flt["Angle"].min(), df_flt["Angle"].max()],
        lw=3,
        c="red",
    )
    ax[2].set_xlim([-xmax * 0.1, xmax * 1.3])
    ax[2].get_legend().remove()
    ax[2].set_xlabel("")
    ax[2].set_xticks([])
    ax[2].set_xticklabels([])

    # Subplot 4
    ax[3].cla()
    ax[3].set_title("Deviations from\nfiltered movement means")
    ax[3].set_ylabel("Difference between angle increments")
    sns.lineplot(
        ax=ax[3],
        data=df_flt,
        x="ts",
        y="desynchronization",
        hue="Whiskers",
        # palette=palette, ci=None)
        errorbar=None,
    )
    ax[3].plot(
        [visual_time, visual_time],
        [df_flt["desynchronization"].min(), df_flt["desynchronization"].max()],
        lw=3,
        c="green",
    )
    ax[3].plot(
        [tch_time, tch_time],
        [df_flt["desynchronization"].min(), df_flt["desynchronization"].max()],
        lw=3,
        c="red",
    )
    n = df_flt.shape[0]
    ax[3].set_xlim([-xmax * 0.1, xmax * 1.3])
    ax[3].get_legend().remove()
    ax[3].set_xlabel("")
    ax[3].set_xticklabels([])

    # Subplot 5
    ax[4].cla()
    ax[4].set_title("Lags in crosscorrelations\nbetween touches and movements")
    ax[4].set_ylabel("Correlation lag")
    sns.lineplot(
        ax=ax[4],
        data=df_flt,
        x="ts",
        y="corrs",
        hue="Whiskers",
        # palette=palette, ci=None)
        palette=palette,
        errorbar=None,
    )
    ax[4].plot(
        [visual_time, visual_time],
        [df_flt["corrs"].min(), df_flt["corrs"].max()],
        lw=3,
        c="green",
    )
    ax[4].plot(
        [tch_time, tch_time],
        [df_flt["corrs"].min(), df_flt["corrs"].max()],
        lw=3,
        c="red",
    )
    means = df_flt.groupby("ts")[["ts", "corrs"]].mean()
    ax[4].fill_between(
        means["ts"],
        np.zeros_like(means["corrs"]),
        means["corrs"],
        lw=3,
        fc="Gray",
        ec="black",
    )
    ax[4].set_xlim([0, pm.stime * 1.3 / 1000])
    # ax[4].set_ylim([-1, 1])
    ax[4].get_legend().remove()

    return fig


def plot_simplified(data, pm=None):

    pm = Params() if pm is None else pm

    df = build_df(data, disable_filtering=True)
    df1 = build_df1(data)

    fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    gs = GS(40, 80)
    palette = sns.color_palette("hls", 6)
    visual_time = pm.time_light_switched_on * 0.001
    tch_time = df.query("touch>0")["ts"].min()

    # subplot 1
    ax[0].cla()
    ax[0].set_title("Whiskers movements")
    ax[0].set_ylabel("Angle")

    sns.lineplot(ax=ax[0], data=df, x="ts", y="Angle", hue="Whiskers", palette=palette)

    sns.scatterplot(
        ax=ax[0],
        data=df,
        x="ts",
        y="touch_angle",
        hue="Whiskers",
        palette=palette,
        ec="black",
        s=5,
        legend=False,
    )

    h, l = ax[0].get_legend_handles_labels()
    ax[0].get_legend().remove()
    ax[0].plot([visual_time, visual_time], [-1.5, 1.5], lw=3, c="green")
    ax[0].plot([tch_time, tch_time], [-1.5, 1.5], lw=3, c="red")

    xmax = df["ts"].max()
    ax[1].set_xlim([-xmax * 0.1, xmax * 1.3])
    ax[0].set_ylim([-1.55, 2.55])
    ax[0].set_xlabel("")
    ax[0].set_xticks([])
    ax[0].set_xticklabels([])
    ax[0].legend(h, l, ncol=2, title="Whiskers")

    # Subplot 2
    ax[1].cla()
    ax[1].set_title("Decision layer")
    ax[1].set_ylabel("Probability")
    sns.lineplot(
        ax=ax[1], data=df1, x="ts", y="Decision_prob", hue="Objects", legend=True
    )
    ax[1].plot([visual_time, visual_time], [0, 1], lw=3, c="green")
    ax[1].plot([tch_time, tch_time], [0, 1], lw=3, c="red")
    ax[1].set_xlim([-xmax * 0.1, xmax * 1.3])
    ax[1].get_legend().remove()
    ax[1].set_xlabel("")
    ax[1].set_yticks([0, 1])
    ax[1].set_xticks([])
    ax[1].set_xticklabels([])

    fig.tight_layout(pad=0.5)

    return df


def plot_data(
    data, mod="desynchronization", return_type="fig", filtered=True, 
    win_gap=0.01, win_size=0.8, pm=None):
   
    init_plt()

    pm = Params() if pm is None else pm

    df = build_df(data, mod=mod, filtered=filtered, win_gap=win_gap, win_size=win_size)
    df1 = build_df1(data)

    fig, ax = plt.subplots(6, 1, figsize=(5, 8), sharex=True)
    palette = sns.color_palette("hls", 6)
    visual_time = pm.time_light_switched_on * 0.001
    try:
        tch_time = df.query("touch>0")["ts"].iloc[0]
    except IndexError:
        tch_time = -1
    # subplot 1
    ax[0].cla()
    ax[0].set_title("Whiskers movements")
    ax[0].set_ylabel("Angle")

    sns.lineplot(ax=ax[0], data=df, x="ts", y="Angle", hue="Whiskers", palette=palette)

    df.loc[np.isnan(df["touch_angle"]), "touch_angle"] = np.inf
    sns.scatterplot(
        ax=ax[0],
        data=df,
        x="ts",
        y="touch_angle",
        hue="Whiskers",
        palette=palette,
        alpha=0.1,
        legend=False,
    )

    h, l = ax[0].get_legend_handles_labels()
    ax[0].plot([visual_time, visual_time], [-1.1, 2], lw=3, c="green")
    ax[0].plot([tch_time, tch_time], [-1.1, 2], lw=3, c="red")

    xmax = df["ts"].max()
    ax[0].set_xlim([-xmax * 0.1, xmax * 1.1])
    ax[0].set_ylim([-1.3, 2.2])
    ax[0].set_xlabel("")
    ax[0].set_xticks([])
    ax[0].set_xticklabels([])

    ax[0].legend(bbox_to_anchor=(1.4, 0.95), title="Whiskers")

    # Subplot 2
    ax[1].cla()
    ax[1].set_title("Decision layer")
    ax[1].set_ylabel("Probability")
    sns.lineplot(
        ax=ax[1], data=df1, x="ts", y="Decision_prob", hue="Objects", legend=True
    )
    ax[1].plot([visual_time, visual_time], [0, 1], lw=3, c="green")
    ax[1].plot([tch_time, tch_time], [0, 1], lw=3, c="red")
    ax[1].set_ylim([-0.1, 1.1])
    ax[1].set_xlim([-xmax * 0.1, xmax * 1.1])
    ax[1].get_legend().remove()
    ax[1].set_xlabel("")
    ax[1].set_yticks([0, 1])
    ax[1].set_xticks([])
    ax[1].set_xticklabels([])

    # Subplot 3
    ax[2].cla()
    ax[2].set_title("Whiskers filtered movements")
    ax[2].set_ylabel("Angle increment")
    sns.lineplot(
        ax=ax[2], data=df, x="ts", y="Filtered_Angle", hue="Whiskers", palette=palette
    )
    mn = df["Filtered_Angle"].min()
    mx = df["Filtered_Angle"].max()
    gap = mx - mn
    ax[2].plot(
        [visual_time, visual_time], [mn - gap * 0.1, mx + gap * 0.1], lw=3, c="green"
    )
    ax[2].plot([tch_time, tch_time], [mn - gap * 0.1, mx + gap * 0.1], lw=3, c="red")
    ax[2].set_xlim([-xmax * 0.1, xmax * 1.1])
    ax[2].get_legend().remove()
    ax[2].set_xlabel("")
    ax[2].set_xticks([])
    ax[2].set_xticklabels([])

    var = mod.title()
    # Subplot 4
    ax[3].cla()
    ax[3].set_title("Deviations from\nfiltered movement means")
    ax[3].set_ylabel("Difference between\nangle increments")
    sns.lineplot(
        ax=ax[3], data=df, x="ts", y=var, hue="Whiskers", palette=palette, errorbar=None
    )
    mn = df[var].min()
    mx = df[var].max()
    gap = mx - mn
    ax[3].plot(
        [visual_time, visual_time], [mn - gap * 0.1, mx + gap * 0.1], lw=3, c="green"
    )
    ax[3].plot([tch_time, tch_time], [mn - gap * 0.1, mx + gap * 0.1], lw=3, c="red")
    n = df.shape[0]
    ax[3].set_xlim([-xmax * 0.1, xmax * 1.1])
    ax[3].get_legend().remove()
    ax[3].set_xlabel("")
    ax[3].set_xticklabels([])

    # Subplot 5
    var = "Corrs_react"
    ax[4].cla()
    ax[4].set_title("Positive lags in crosscorrelations\nbetween touches and movements")
    ax[4].set_ylabel("Correlation lag")
    sns.lineplot(
        ax=ax[4],
        data=df,
        x="ts",
        y=var,
        hue="Whiskers",
        palette=palette,
        errorbar=None,
        zorder=5,
    )
    

    mn = df[var].min()
    mx = df[var].max()
    gap = mx - mn
    ax[4].plot(
        [visual_time, visual_time], [mn - gap * 0.1, mx + gap * 0.1], lw=3, c="green", zorder=0,
    )
    ax[4].plot([tch_time, tch_time], [mn - gap * 0.1, mx + gap * 0.1], lw=3, c="red", zorder=0)
    means = df.groupby("ts")[["ts", var]].mean()
    ax[4].fill_between(
        means["ts"],
        np.zeros_like(means[var]),
        means[var],
        lw=3,
        fc="Gray",
        ec="black",
        zorder=10,
    )
    ax[4].set_xlim([-xmax * 0.1, xmax * 1.1])
    ax[4].get_legend().remove()
    
    # Subplot 6
    var = "Corrs_pred"
    ax[5].cla()
    ax[5].set_title("Negative lags in crosscorrelations\nbetween touches and movements")
    ax[5].set_ylabel("Correlation lag")
    sns.lineplot(
        ax=ax[5],
        data=df,
        x="ts",
        y=var,
        hue="Whiskers",
        palette=palette,
        errorbar=None,
        zorder=5,
    )
    

    mn = df[var].min()
    mx = df[var].max()
    gap = mx - mn
    ax[5].plot(
        [visual_time, visual_time], [mn - gap * 0.1, mx + gap * 0.1], lw=3, c="green", zorder=0,
    )
    ax[5].plot([tch_time, tch_time], [mn - gap * 0.1, mx + gap * 0.1], lw=3, c="red", zorder=0)
    means = df.groupby("ts")[["ts", var]].mean()
    ax[5].fill_between(
        means["ts"],
        np.zeros_like(means[var]),
        means[var],
        lw=3,
        fc="Gray",
        ec="black",
        zorder=10,
    )
    ax[5].set_xlim([-xmax * 0.1, xmax * 1.1])
    ax[5].get_legend().remove()

    fig.tight_layout(pad=0.1)
    if return_type == "fig":
        return fig
    else:
        return df
