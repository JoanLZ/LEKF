#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
from manifpy import SO3, SO3Tangent
from matplotlib import pyplot as plt

import lekf
import state
import measurement


def f(X, U, W, dt):
    g = np.array([0, 0, -9.81])
    ω = U.ω_m - W.ω_wn - W.ω_rw
    a =

    X_out = state.State(X.R+())


# Constants for simulation
_IMU_RATE = 1000  # [Hz]
_OPTITRACK_RATE = 100  # [Hz]
assert _IMU_RATE % _OPTITRACK_RATE == 0, 'Imu rate should be a multiple of optitrack rate.'

# Sigma
w_sigmas = measurement.ImuNoise(0.1*np.ones(3), 0.1*np.ones(3),
                                0.01*np.ones(3), 0.01*np.ones(3))
v_sigmas = measurement.OptitrackNoise(0.1*np.ones(SO3.Dof()), 0.1*np.ones(3))

if __name__ == "__main__":
    '''Initialisation'''
    dt_imu = 1/_IMU_RATE # [s]
    dt_ot = 1/_OPTITRACK_RATE # [s]
    t_imu = 0 # imu tracking time [s]
    t_ot = 0 # ot tracking time [s]

    X = state.State(SO3.Random(), np.random.rand(3), np.random.rand(3),
                    np.random.rand(3), np.random.rand(3))
    U = measurement.ImuMeasurement(np.zeros(3), np.zeros(3))
    W = measurement.ImuNoise(np.zeros(3), np.zeros(3),
                             np.zeros(3), np.zeros(3))
    Y = measurement.OptitrackMeasurement(X.R, X.p)
    V = measurement.OptitrackNoise(np.zeros(3), np.zeros(3))

    fig, ax = plt.subplot()
    while True:
        '''Loop'''
        '''IMU TIME!'''
        # W update
        W.a_wn = w_sigmas.a_wn * np.random.rand(3)
        W.ω_wn = w_sigmas.ω_rw * np.random.rand(3)
        W.a_rw = w_sigmas.a_rw * np.random.rand(3)
        W.ω_rw = w_sigmas.ω_rw * np.random.rand(3)
        # U update
        ω_nom = np.array([0, 0, 1])
        a_nom = np.array([-np.cos(dt_imu), -np.sin(dt_imu), 0])
        g = np.array([0, 0, -9.81])
        U.a_m = X.R.inverse().act(a_nom-g)+W.a_wn+w_sigmas.a_rw
        U.ω_m = ω_nom + W.ω_wn + W.ω_rw

        X, _, _, _ = lekf.f(X, U, W, dt_imu)
        # lekf.predict(U, dt_imu)
        t_imu = t_imu + dt_imu
        if t_imu >= t_ot:
            '''OPTITRACK TIME!'''
            # V update
            V.R_wn = v_sigmas.R_wn * np.random.rand(SO3.Dof())
            V.p_wn = v_sigmas.p_wn * np.random.rand(3)

            Y, _, _ = lekf.h(X, V)
            # lekf.correct(Y)
            t_ot = t_ot + dt_ot

        # Visualisation
