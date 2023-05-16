#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
from manifpy import SO3

from lekf import LEKF
import state
import measurement

# Constants for simulation
_IMU_RATE = 1000  # [Hz]
_OPTITRACK_RATE = 100  # [Hz]
_TIME = 15  # [s]
# Sigmas
w_sigmas = measurement.ImuNoise(0.1*np.ones(3), 0.1*np.ones(3),
                                0.01*np.ones(3), 0.01*np.ones(3))
v_sigmas = measurement.OptitrackNoise(0.1*np.ones(SO3.DoF), 0.1*np.ones(3))

# Initialitzation of covariances
#first sigma, then sigma² = variance, and then the covariance matrix.

# Covariance of the State

p_sigmas = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
                     #[std_error R,std_error v,std_error p,std_error ab,std_error wb ] 

P0 = np.diagflat(np.squares(p_sigmas))

# Covariance of the Measurment IMU
q_sigmas = np.array([0.1,0.1,0.1,0.1,0.1,0.1])
                     #[std_error ab,std_error ωb] 

Q0 = np.diagflat(np.squares(q_sigmas))

# Covariance of the IMU bias
w_sigmas = np.array([0.1,0.1,0.1,0.1,0.1,0.1])
                     #[std_error R,std_error v]

W0 = np.diagflat(np.squares(w_sigmas))

# Covariance of the Measurment Optitrack
v_sigmas = np.array([0.1,0.1,0.1,0.1,0.1,0.1])
                     #[std_error R,std_error v,std_error p,std_error ab,std_error wb ] 

V0 = np.diagflat(np.squares(v_sigmas))

g = np.array([0, 0, -9.81])

def update(X, t):
    pass #TODO

def observe(X):
    pass #TODO


X_list = []
U_list = []
Y_list = []
X_est_list = []
P_est_list = []

if __name__ == "__main__":
    '''Initialisation'''
    dt_imu = 1/_IMU_RATE  # [s]
    dt_ot = 1/_OPTITRACK_RATE  # [s]
    dt = dt_imu*dt_ot  # [s]

    t_imu = 0  # imu tracking time [s]
    t_ot = 0  # ot tracking time [s]
    t = 0  # global tracking [s]

    X = state.State(SO3.Identity(), np.array([1, 1, 0]), np.zeros(3),
                    np.zeros(3), np.zeros(3))

    lekf = LEKF(X,P0,Q0,W0,V0) #TODO

    '''Simulation loop'''
    for t in np.arange(0, _TIME, dt):

        if t >= t_imu:
            '''Imu data'''
            X, U = update(X, t)
            X_list.append(X)
            U_list.append(U)
            lekf.predict(U, dt_imu)
            X_est_list.append(lekf.X)
            P_est_list.append(lekf.P)

            t_imu = t_imu + dt_imu

        if t >= t_ot:
            '''Optitrack data'''
            Y = observe(X)
            Y_list.append(Y)
            lekf.correct(Y)
            X_est_list.append(lekf.X)
            P_est_list.append(lekf.P)

            t_ot = t_ot + dt_ot

    '''Data process'''
