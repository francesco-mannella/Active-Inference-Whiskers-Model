import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as GS
import numpy as np
import gym, RatSim
from simulation.ActiveInverenceModel import GM
from simulation.params import Params
from tools import make_simulation_gif_video
from analysis.stats.fourier_transform import band_filter
from analysis.stats.crosscorr import (
    compute_correlation_desynchronization_touch,
)
from RatSim.envs.mkvideo import vidManager
import pandas as pd
import seaborn as sns
import os, regex as re
import tools as tools
import gc
from PIL import Image

plt.ioff()


def concat_frames(frms1, frms2):
    frms = []
    for im1, im2 in zip(frms1, frms2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        frms.append(dst)
    return frms


class RendererOnstep:

    labels = [f'whisk_{x}_{y}' for x in ['left', 'right'] for y in range(3)]

    def __init__(self):
        self.fig = plt.figure(figsize=(12, 5))
        self.gs = GS(10, 30)
        self.ax1 = None
        self.ax2 = None
        self.vm = vidManager(self.fig, name='onstep', duration=400)

    def __call__(self, data, touches, decisions):
        self.data = data
        self.touches = touches
        self.decisions = decisions
        
        self.fig.clear()
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)

        df = self.mkDataFrame()

        palette = sns.color_palette('hls', 6)
        sns.lineplot(
            ax=self.ax1,
            data=df,
            x='ts',
            y='Angle',
            hue='Whiskers',
            palette=palette,
            legend=None,
            errorbar=None,
        )
        sp = sns.scatterplot(
            ax=self.ax1,
            data=df,
            x='ts',
            y='touch',
            hue='Whiskers',
            palette=palette,
            legend=False,
            alpha=0.3,
        )
        self.ax1.set_xlim(self.xlim)
        self.ax1.set_ylim([-1.2, 2.4])

        objs = ['obj0', 'obj1']
        df = pd.DataFrame({o: x for o, x in zip(objs, self.decisions.T)})
        t = np.arange(len(self.decisions))
        df['ts'] = t / 1000
        df = df.melt(id_vars=['ts'], value_vars=objs)

        df = df.rename(columns={'variable': 'Objects', 'value': 'decision'})
        sns.lineplot(
            ax=self.ax2,
            data=df,
            x='ts',
            y='decision',
            hue='Objects',
            legend=None,
            errorbar=None,
        )
        self.ax2.set_xlim(self.xlim)
        self.ax2.set_ylim([-0.2, 1.2])
        self.fig.canvas.draw()
        self.vm.save_frame()

    def mkDataFrame(self):
        df = pd.DataFrame({l: k for k, l in zip(self.data.T, self.labels)})
        t = np.arange(len(self.data))
        df['ts'] = (t) / 1000
        df['touch'] = 0

        df = df.melt(id_vars=['ts'], value_vars=self.labels)
        df = df.rename(columns={'variable': 'Whiskers', 'value': 'Angle'})
        if df.shape[0] > 1:
            for i, w in enumerate(self.labels):
                df.loc[df['Whiskers'] == w, 'touch'] = (
                    self.touches[:, i] * df.loc[df['Whiskers'] == w, 'Angle']
                )
        df.loc[df['touch'] == 0, 'touch'] = np.nan

        return df


def run_simulation(
    obj=0,
    visual_trial=False,
    touch_trial=False,
    noise=0,
    seed=None,
    plot=True,
    jupyter=True,
    log=True,
    params=None,
):
    env = None
    env = gym.make('RatSim-v0', disable_env_checker=True)
    env = env.unwrapped

    pm = params if params is not None else Params()
    pm.time_obj_movement += int((pm.phase) / (pm.dt * pm.scale))

    pm.time_light_switched_on = (
            pm.time_obj_movement + 
            pm.relative_time_switched_on
            )


    pm.time_light_switched_on += int((pm.phase) / (pm.dt * pm.scale))

    pm.obj = obj

    if seed is None:
        seed = np.frombuffer(os.urandom(4), dtype=np.uint32)[0]

    # initializing random state
    rng = np.random.RandomState(seed)
    if log == True:
        pbar = tools.PBar(
            f'rendering simulation):',
            n_ticks=40,
            n_counts=pm.stime // 10,
            n_decimals=1,
        )

    if plot is True or jupyter is True:
        env.reset()
        renderer_onstep = RendererOnstep()
        renderer_onstep.xlim = [0, pm.stime / 1000]
        env.renderer_figsize = (5, 5)
    else:
        env.reset()

    env.noise = noise
    env.set_dt(0.5 * np.pi * pm.freq * pm.cicles)

    # initialization of generative model
    gm = GM(
        dt=pm.dt,
        eta_a=pm.eta_a,
        Sigma_s_p=pm.Sigma_s_p,
        Sigma_s_t=pm.Sigma_s_t,
        Sigma_s_v=pm.Sigma_s_v,
        Sigma_mu=pm.Sigma_mu,
        eta_mu=pm.eta_mu,
        eta_dmu=pm.eta_dmu,
        nu=pm.nu,
        Sigma_nu=pm.Sigma_nu,
        eta_nu=pm.eta_nu,
        nu_obj0=pm.nu_obj0,
        nu_obj1=pm.nu_obj1,
        tau_mu_y=pm.tau_mu_y,
        Sigma_mu_y=pm.Sigma_mu_y,
        eta_mu_y=pm.eta_mu_y,
        eta_dmu_y=pm.eta_dmu_y,
        eta_nu2=pm.eta_nu2,
    )

    # initialization of the prior based on the parameter set before
    gm.nu2 = 1 if pm.object_prior == 0 else 0 if pm.object_prior == 1 else 0.5
    if pm.object_prob_prior is not None:
        gm.nu2 = pm.object_prob_prior

    # initialization of variables that have to be saved for graphs
    data = {
        'touch_sensor': np.zeros([pm.stime, 6]),
        'gm_touch_sensor': np.zeros([pm.stime, 6]),
        'gm_mu_x': np.zeros([pm.stime]),
        'gm_mu': np.zeros([pm.stime, 6]),
        'gm_dmu': np.zeros([pm.stime, 6]),
        'gm_nu': np.zeros([pm.stime, 6]),
        'gm_mu_y': np.zeros([pm.stime, 2]),
        'gm_nu2': np.zeros([pm.stime]),
        'gm_PE_s_t': np.zeros([pm.stime, 6]),
        'gm_PE_s_p': np.zeros([pm.stime, 6]),
        'gm_PE_s_v': np.zeros([pm.stime, 2]),
        'gm_PE_nu': np.zeros([pm.stime, 6]),
        'gp_x': np.zeros([pm.stime, 6]),
        'gp_dx': np.zeros([pm.stime, 6]),
        'gp_alpha': np.zeros([pm.stime, 6]),
        'obj': np.zeros([pm.stime, 2]),
        'frames': [],
    }

    # environment initialization
    state = env.reset(pm.obj)
    angle_prevs = state['JOINT_POSITIONS'][:-1]
    init_pos = pm.stop_pos - pm.obj_mov_length
    pos = env.move_object(
        'box',
        [0, init_pos + (not touch_trial) * pm.notouch_initial_displacement],
        angle=0.1 * noise * rng.randn(),
    )
    path = pm.stop_pos - pos[1]
    mov_increment = path / pm.obj_mov_duration
    touch_control = np.zeros(6)
    visual_input = [0, 0]

    # preactivation
    Sigma_vision_saved = gm.Sigma_s_v
    gm.Sigma_s_v = 100000

    Sigma_t_saved = gm.Sigma_s_t
    gm.Sigma_s_t = 100000

    for t in range(pm.stime // 6):

        # -------------------------------------------------------------------------
        # Step

        angle_prevs = state['JOINT_POSITIONS'][:-1]

        state, rew, info, done = env.step(pm.action)

        touches = state['TOUCH_SENSORS']

        # noise in the receptor
        angle_velocities = (
            state['JOINT_POSITIONS'][:-1] - angle_prevs
        ) / pm.dt + 0.5 * rng.randn(*angle_prevs.shape)
        pm.action[:6] += gm.update(
            touch_sensory_states=touches,
            proprioceptive_sensory_states=angle_velocities,
            x=env.oscillator,
            visual_input=visual_input,
        )

    for t in range(pm.stime):

        # -------------------------------------------------------------------------
        # Scheduling of the simulation

        # specification of object movement
        if pm.moving_object == True:
            if (
                pos[1] < (pm.stop_pos + pm.stop_pos_adj * pm.obj + 0.2*pm.stop_pos*noise*rng.rand())
                and t < pm.time_obj_movement + pm.obj_mov_duration
                and t > pm.time_obj_movement
            ):
                # moving the object forward
                pos = env.move_object('box', [0, mov_increment])

            if t == pm.time_light_switched_on:
                if visual_trial == True:
                    visual_input[pm.obj] = 1
                    gm.Sigma_s_v = Sigma_vision_saved

            if t == pm.time_obj_movement:
                gm.Sigma_s_t = Sigma_t_saved


        # -------------------------------------------------------------------------
        # Step

        angle_prevs = state['JOINT_POSITIONS'][:-1]

        state, rew, info, done = env.step(pm.action)

        touches = state['TOUCH_SENSORS']
        if sum(touches) > 0:
            gm.Sigma_s_t = Sigma_t_saved

        # noise in the receptor
        angle_velocities = (
            state['JOINT_POSITIONS'][:-1] - angle_prevs
        ) / pm.dt + 0.05 * rng.randn(*angle_prevs.shape)
        pm.action[:6] += gm.update(
            touch_sensory_states=touches,
            proprioceptive_sensory_states=angle_velocities,
            x=env.oscillator,
            visual_input=visual_input,
        )

        # -------------------------------------------------------------------------
        # Storing data

        data['gm_touch_sensor'][t] = gm.s_t
        data['gm_mu_x'][t] = gm.mu_x
        data['gm_mu'][t] = gm.mu
        data['gm_dmu'][t] = gm.dmu
        data['gm_nu'][t] = gm.nu
        data['gm_mu_y'][t] = gm.mu_y
        data['gm_nu2'][t] = gm.nu2
        data['gm_PE_s_t'][t] = gm.PE_s_t
        data['gm_PE_s_p'][t] = gm.PE_s_p
        data['gm_PE_s_v'][t] = gm.PE_s_v
        data['gm_PE_nu'][t] = gm.PE_nu
        data['gp_x'][t] = state['JOINT_POSITIONS'][:6]
        data['gp_dx'][t] = np.copy(angle_velocities)
        data['gp_alpha'][t] = pm.action[:6]
        data['touch_sensor'][t] = touches
        data['obj'][t] = visual_input

        # Plotting
        if plot is True or jupyter is True:
            if t % 200 == 0 or t == pm.stime - 1:
                renderer_onstep(
                    data['gp_x'][: t + 1],
                    data['touch_sensor'][: t + 1],
                    data['gm_mu_y'][: t + 1],
                )
                env.render('offline')
        if log == True:
            if t % (pm.stime // 10) == 0 or t == pm.stime - 1:
                pbar.print(t // 10 + 1)

    data["params"] = pm 
    if plot is True or jupyter is True:

        env.renderer.vm.frames = concat_frames(
            env.renderer.vm.frames,
            renderer_onstep.vm.frames,
        )

        gif = env.renderer.video(jupyter=jupyter)
        return data, gif
    else:
        return data, None


def gaussian_filter(x, smoothing=0.5, prec=0.8):
    n = len(x)
    rng = int(n * (1 - prec))
    s = n * smoothing
    flt = np.exp(-0.5 * (s**-2) * (np.linspace(0, 1, rng) - 0.5) ** 2)
    flt /= flt.sum()
    return np.convolve(x, flt, 'same')


def get_sm_var(data, var, var_name, params=None):

    ## Build with whisker labels
    labels = [f'{x}_{y}' for x in ['left', 'right'] for y in range(3)]

    df = pd.DataFrame({l: k for k, l in zip(data[var].T, labels)})

    ## Add timestep index
    t = np.arange(len(data[var]))
    dt = 0.0001
    if params is not None:
        dt = params.dt * params.scale
    df['ts'] = t * dt

    df = df.melt(id_vars=['ts'], value_vars=labels)
    df = df.rename(columns={'variable': 'Whiskers', 'value': var_name})
    return df


def build_df(
    data,
    filtered=True,
    mod='desynchronization',
    pred=True,
    win_gap=0.05,
    win_size=0.3,
    disable_filtering=False,
    params=None,
):
    dfv = []
    for var, var_name in zip(
        ['gp_x', 'touch_sensor', 'gm_mu', 'gm_PE_s_t'],
        ['Angle', 'touch', 'PredAngle', 'PredError'],
    ):
        cdf = get_sm_var(data, var, var_name, params)
        dfv.append(cdf)

    df = dfv[0]
    for cdf in dfv[1:]:
        df = df.merge(cdf)

    df['touch_per_whisk'] = df['touch']
    df['touch_angle'] = df['Angle']
    df.loc[df['touch'] == 0, 'touch_angle'] = np.nan

    if not disable_filtering:

        ts_num = df.groupby('ts').first().shape[0]
        ts_min = df.groupby('ts').first().reset_index()['ts'].min()
        ts_max = df.groupby('ts').first().reset_index()['ts'].max()
        ts_rng = ts_max - ts_min

        df_flt = band_filter(
            df.copy(),
            variable='Whiskers',
            value='Angle',
            filtered=filtered,
            mode=(0.5, 80),
            h=ts_rng / ts_num,
        )

        df_flt['ttouch'] = df_flt.groupby('Whiskers')['touch'].transform(
            lambda x: gaussian_filter(x.to_numpy(), prec=0.95, smoothing=0.1)
        )

        df_flt['corrs'] = compute_correlation_desynchronization_touch(
            ts=df_flt['ts'].to_numpy(),
            desynchronization=df_flt['desynchronization'].to_numpy(),
            touch=df_flt['touch'].to_numpy(),
            win_gap=win_gap,
            win_size=win_size,
        )

        df['Filtered_Angle'] = df_flt['Angle']

    df['Speed'] = df_flt['speed']
    df['Accel'] = df_flt['accel']
    df['Momentum'] = df_flt['momentum']
    df['Desynchronization'] = df_flt['desynchronization']
    df['Corrs'] = df_flt['corrs']
    df['Direction'] = [re.sub(r'^(\w+)_\d', r'\1', s) for s in df['Whiskers']]
    df['Corrs'] = df_flt['corrs']
    df['Corrs_pred'] = np.minimum(0, df_flt['corrs'])
    df['Corrs_react'] = np.maximum(0, df_flt['corrs'])
    df['touch_per_whisk'] = df['touch']

    def dirtouch(x):
        x['touch'] = x['touch'].max()
        return x

    df = df.groupby(['ts', 'Direction']).apply(dirtouch)
    return df


def build_df1(data):

    t = np.arange(len(data['gm_mu_y']))
    df = pd.DataFrame(data['gm_mu_y'])
    df.columns = ['obj0', 'obj1']
    df['ts'] = t / 1000
    df = df.melt(
        id_vars='ts',
        value_vars=['obj0', 'obj1'],
        var_name='Objects',
        value_name='Decision_prob',
    )
    return df


# %%

if __name__ == '__main__':

    # %%
    
    from simulation.params import Params
    from simulation.plot_simulation import plot_simplified
    from simulation.plot_simulation import plot_data
    
    SMALL_SIZE = 7
    MEDIUM_SIZE = 11
    BIGGER_SIZE = 18
    
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # %%
    dfs = []
    plot = True
    seed = 3
    visual = True
    obj = 0 
    data, frames = run_simulation(
        obj=obj,
        visual_trial=visual,
        touch_trial=True,
        noise=0.05,
        seed=seed,
        plot=plot,
        jupyter=False,
    )
    # %%
    df = build_df(
        data,
        mod='desynchronization',
        filtered=True,
        win_gap=0.01,
        win_size=0.8,
    )
    
    f = plot_data(data, return_type='fig', pm=data["params"])
    f.savefig(
        f"data_s{obj}_{'touch' if visual is False else 'visual'}.png"
    )
    
    if not frames is None:
        make_simulation_gif_video(
            folder='.',
            root=f"s{obj}_{'touch' if visual is False else 'visual'}",
            frames=frames,
        )
