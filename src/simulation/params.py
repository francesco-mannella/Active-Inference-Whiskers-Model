import numpy as np

class Params:
    def __init__(self):
        # whiskers order : "whisk_left_1", "whisk_left_2", "whisk_left_3", "whisk_right_1", "whisk_right_2", "whisk_right_3"

        # initializing 6 whiskers amplitudes, 6 central values of oscillation
        # action[12:15] sohuld be the 2 coordinates of rat position and the
        # angle describing head rotation
        self.action = np.zeros(15)
        self.action[:6] = np.ones(6) * np.pi * [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
        self.action[6:12] = np.ones(6) * np.pi * [0.0, 0.1, 0.35, 0.0, 0.1, 0.35]
        self.stop_pos = -4.8
        self.stop_pos_adj = 0
        self.obj_mov_length = 1 
        self.obj_mov_duration = 263.158 
        self.notouch_initial_displacement = -0.2
        
        # Objects representations in terms of whisking amplitudes
        self.nu_obj0 = np.array([0.000, 0.100, 0.400, 0.100, 0.600, 0.600])
        self.nu_obj1 = np.array([0.100, 0.600, 0.600, 0.000, 0.100, 0.400])

        # the ones representing the two objects
        self.action[:6] = np.ones(6)

        # integration time interval
        self.dt = 0.004
        self.scale = 0.05 # 4/100 -- actual timesteps are 0.004*(4/100)
        self.cicles = 5
        self.freq = self.dt/(2*np.pi)
        self.phase = 0
        # simulation time stamps
        self.stime = 12000
        # set the type of object in front of the agent. Object 1 is larger than object 0
        self.obj = 0
        self.moving_object = True
        # angle
        self.obj_angle = 0
        self.time_obj_movement = 1000
        self.relative_time_switched_on = -self.obj_mov_duration

        # variable setting agent's prior.
        # 0 means that the agent expect to have in front object 0,
        # 1 means that it expect object 1, while 2 stands for no prior.
        self.object_prior = 2
        self.object_prob_prior = None
        # parameter that, if set to False, turn off (increase) gm.Sigma_nu leaving out
        # the race model part and the eventual prior present.
        self.prior = True
        # set the time in which the gm.Sigma_nu are decreased again to the default values

        # action parameters
        self.eta_a = 0.01
        # sensory input parameters
        self.Sigma_s_p = 0.002  # proprioception
        self.Sigma_s_t = 0.02  # touch
        self.Sigma_s_v = 0.0005  # vision
        self.Sigma_s_v_no_v = 100000  # vision
        # whiskers dynamics parameters
        self.Sigma_mu = 0.003
        self.eta_mu = 0.003
        self.eta_dmu = 0.2
        # whisking amplitudes parameters
        self.nu = 0.8 * np.ones(6)
        self.Sigma_nu = 0.02 # * 1e200
        self.eta_nu = 0.02
        # parameters regulating dynamics decision making between two objects
        self.tau_mu_y = 10.0
        self.Sigma_mu_y = [0.003, 0.003]
        self.eta_mu_y = [0.001, 0.001]
        self.eta_dmu_y = [.2, .2]
        self.eta_nu2 = 0.35
