"""
Copyright 2021 Federico Maggiore, Francesco Mannella

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to
do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np


def sech(x):
    x[np.abs(x) > 0.71e3] = 0.71e3 * np.sign(x[np.abs(x) > 0.71e3])
    return 1 / np.cosh(x)


def tanh(x):
    return np.tanh(x)


def rad_dist(x, s=1):
    return np.exp(-0.5 * (s**-2) * np.linalg.norm(x) ** 10)


# Generative model class
class GM:
    def __init__(
        self,
        dt,
        eta_a,
        Sigma_s_p,
        Sigma_s_t,
        Sigma_s_v,
        Sigma_mu,
        eta_mu,
        eta_dmu,
        nu,
        Sigma_nu,
        eta_nu,
        nu_obj0,
        nu_obj1,
        tau_mu_y,
        Sigma_mu_y,
        eta_mu_y,
        eta_dmu_y,
        eta_nu2,
        eta_x=0.0001,
        Sigma_mu_x=0.000025,
    ):

        # Size of a simulation step
        self.dt = dt

        # ACTION
        # Action variable (in this case the action is intended as the increment of the variable that the agent is allowed to modified)
        self.da = 0.0
        # Action gradient descent weight
        self.eta_a = eta_a

        # SENSORY INPUTS
        # Variances (inverse of precisions) of sensory proprioceptive inputs
        self.Sigma_s_p = Sigma_s_p
        # Variances (inverse of precisions) of sensory touch inputs
        self.Sigma_s_t = Sigma_s_t
        # Variances (inverse of precisions) of vision inputs
        self.Sigma_s_v = Sigma_s_v

        # CPG
        # Internal variable of CPG sensory input
        self.mu_x = 0.0
        # Variance of CPG sensory input
        self.Sigma_mu_x = Sigma_mu_x
        # CPG sensory input gradient descent weight
        self.eta_x = eta_x

        # WHISKERS DYNAMICS
        # Internal representation of whiskers base angle
        self.mu = np.zeros(len(nu))
        # Internal representation of whiskers base angle velocity
        self.dmu = np.zeros(len(nu))
        # Internal variables precisions
        self.Sigma_mu = Sigma_mu
        # \vec{mu} and \vec{mu'} gradient descent weights
        self.eta_mu = eta_mu
        self.eta_dmu = eta_dmu

        # OSCILLATION AMPLITUDES
        # Parameter that regulates whiskers amplitude oscillation
        self.nu = np.array(nu)
        # Variance of nu
        self.Sigma_nu = Sigma_nu
        # \nu parameter gradient descent wheight
        self.eta_nu = eta_nu
        # matrix operator that applied to mu_y gives agent's nu predictions
        self.g_nu = np.vstack([nu_obj0, nu_obj1])

        # Internal vector in which each component represents how much the agent is certain regarding
        # hypothesis of object 0 and hypothesis of object 1 (respectively [1,0] correspond to object
        # 0 while [0,1] to object 1). This is a variable also used to predict an already processed
        # visual input s_v, through function g_s_v, and the vector nu controlling the amplitude of
        # whisking of each whisker
        self.mu_y = np.array([0.5, 0.5])
        # First order variable used to define race model
        self.dmu_y = np.array([0.0, 0.0])
        # Parameter that regulates time scale of race model dynamics
        self.tau_mu_y = tau_mu_y
        # mu_y dynamics precision
        self.Sigma_mu_y = np.array(Sigma_mu_y)
        # \vec{mu_y} gradient descent weights
        self.eta_mu_y = np.array(eta_mu_y)
        # \vec{dmu_y} gradient descent weights
        self.eta_dmu_y = np.array(eta_dmu_y)

        # Internal variable directing race model
        self.nu2 = 0.5
        # Gradient descent weight
        self.eta_nu2 = eta_nu2

    # Touch function
    def g_touch(self, x, v, prec=50):
        return sech(prec * v) * (0.5 * tanh(prec * x) + 0.5)

    # Derivative of the touch function with respect to v
    def dg_touch_dv(self, x, v, prec=50):
        return (
            -prec
            * sech(prec * v)
            * tanh(prec * v)
            * (0.5 * tanh(prec * x) + 0.5)
        )

    # Derivative of the touch function with respect to x
    def dg_touch_dx(self, x, v, prec=50):
        return sech(prec * v) * 0.5 * prec * (sech(prec * x)) ** 2

    # Derivative with respect to vector y of the mapping function between y and nu
    # (for now is a simple matrix 6x2)
    def dg_nu_dy(self, g_nu):
        return g_nu

    def f_mu_y(self, nu2):
        return np.array([nu2, 1.0 - nu2])

    # Function that implement the update of internal variables.
    def update(
        self,
        touch_sensory_states,
        proprioceptive_sensory_states,
        x,
        visual_input=np.zeros(2),
    ):
        # touch_sensory_states  and proprioceptive_sensory_states arguments come from GP (both arrays have dimension equal to the number of whiskers)
        # Returns action increment

        # x = np.array(x)    # ?needed?
        self.s_p = proprioceptive_sensory_states
        self.s_t = touch_sensory_states
        self.s_v = visual_input
        self.s_x = x

        self.PE_mu = self.dmu - (self.nu * self.mu_x - self.mu)
        self.PE_s_t = self.s_t - self.g_touch(x=self.mu, v=self.dmu)
        self.PE_s_p = self.s_p - self.dmu
        self.PE_s_x = self.s_x - self.mu_x
        self.PE_s_v = self.s_v - self.mu_y
        self.PE_nu = self.nu - np.dot(self.mu_y, self.g_nu)

        self.PE_mu_y = (
            self.dmu_y - (self.f_mu_y(self.nu2) - self.mu_y) / self.tau_mu_y
        )

        self.dF_dmu = (
            +self.PE_mu / self.Sigma_mu
            - self.dg_touch_dx(x=self.mu, v=self.dmu)
            * self.PE_s_t
            / self.Sigma_s_t
        )

        self.dF_d_dmu = (
            +self.PE_mu / self.Sigma_mu
            - self.PE_s_p / self.Sigma_s_p
            - self.dg_touch_dv(x=self.mu, v=self.dmu)
            * self.PE_s_t
            / self.Sigma_s_t
        )

        self.dF_dmu_x = -self.PE_s_x / self.Sigma_mu_x - np.sum(
            self.nu * self.PE_mu / self.Sigma_mu
        )

        numu_dist = np.abs(np.diff(self.mu_y).sum()) < 0.6
        numu_amp = 1
        self.dF_dnu = -self.mu_x * self.PE_mu / self.Sigma_mu + self.PE_nu / (
            self.Sigma_nu + numu_amp * (numu_dist)
        )

        self.dF_dmu_y = (
            -self.PE_s_v / self.Sigma_s_v
            - np.dot(
                self.dg_nu_dy(self.g_nu),
                self.PE_nu / (self.Sigma_nu + numu_amp * (1 - numu_dist)),
            )
            - self.PE_mu_y / self.Sigma_mu_y
        )

        self.dF_d_dmu_y = +self.PE_mu_y / self.Sigma_mu_y

        self.dF_dnu2 = (
            -self.PE_mu_y[0] / self.Sigma_mu_y[0]
            + self.PE_mu_y[1] / self.Sigma_mu_y[1]
        )

        # Action update
        # case with dg/da = 1
        self.da = (
            -self.dt
            * self.eta_a
            * (
                self.mu_x * self.PE_s_p / self.Sigma_s_p
                + self.PE_s_t / self.Sigma_s_t
            )
        )
        # case with real dg/da
        # self.da = -self.dt*eta_a*x*self.dg_dv(x=self.mu, v=self.dmu)*self.PE_s[1]/self.Sigma_s[1]

        # Learning internal parameter nu
        self.nu += -self.dt * self.eta_nu * self.dF_dnu
        self.nu = np.maximum(0, np.minimum(1, self.nu))
        self.mu += self.dt * (self.dmu - self.eta_mu * self.dF_dmu)
        self.mu = np.maximum(0, np.minimum(1, self.mu))
        self.dmu += -self.dt * self.eta_dmu * self.dF_d_dmu
        self.mu_x += -self.dt * self.eta_x * self.dF_dmu_x
        self.mu_y += +self.dt * (self.dmu_y - self.eta_mu_y * self.dF_dmu_y)
        self.dmu_y += -self.dt * self.eta_dmu_y * self.dF_d_dmu_y
        self.mu_y = np.maximum(0, np.minimum(1, self.mu_y))
        self.nu2 += -self.dt * self.eta_nu2 * self.dF_dnu2
        self.nu2 = np.maximum(0, np.minimum(1, self.nu2))

        return self.da
