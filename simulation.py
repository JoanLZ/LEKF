#!/usr/bin/python3

import numpy as np

from lekf import LEKF
from state import State, StateTangent
from input import Input
from output import Output, OutputTangent
from pertubation import Pertubation

from sim_config import *

σ_P = np.array([σ_P_R*np.ones(3),
                σ_P_v*np.ones(3),
                σ_P_p*np.ones(3),
                σ_P_ab*np.ones(3),
                σ_P_ωb*np.ones(3)]).flatten()
P0 = np.diagflat(np.square(σ_P))

σ_Q = np.array([σ_Q_am*np.ones(3),
                σ_Q_ωm*np.ones(3)]).flatten()
Q0 = np.diagflat(np.square(σ_Q))

σ_W = np.array([σ_W_aw*np.ones(3),
                σ_W_ωw*np.ones(3),
                σ_W_ar*np.ones(3),
                σ_W_ωr*np.ones(3)]).flatten()
W0 = np.diagflat(np.square(σ_W))

σ_V = np.array([σ_V_Rw*np.ones(3),
                σ_V_pw*np.ones(3)]).flatten()
V0 = np.diagflat(np.square(σ_V))

def command(X: State, t: float) -> any:
    a = np.array([-np.cos(t), -np.sin(t), 0])
    ω = np.ones(3)
    return a, ω

def update(X: State, t: float, dt: float) -> any:

    R  = X.get_R()
    v  = X.get_v().coeffs_copy()
    ab = X.get_ab().coeffs_copy()
    ωb = X.get_ωb().coeffs_copy()

    a, ω = command(X, t)

    # Pertubation W
    W = Pertubation.Identity()
    if IMU_NOISE:
        W = Pertubation.Random(np.array([σ_sim_aw,
                                         σ_sim_ωw,
                                         σ_sim_ar,
                                         σ_sim_ωr]))
    aw = W.get_aw().coeffs_copy()
    ωw = W.get_ωw().coeffs_copy()
    ar = W.get_ar().coeffs_copy()
    ωr = W.get_ωr().coeffs_copy()

    # New input U
    am = R.inverse().act(a-G)+ab+aw
    ωm = ω + ωb + ωw
    U = Input(np.array([am, ωm]).flatten())

    # New step ΔX
    coeffs = np.array([ω*dt, a*dt,
                       v*dt+0.5*a*(dt**2),
                       ar, ωr]).flatten()
    ΔX = StateTangent(coeffs)

    return X+ΔX, U


def observe(X: State) -> Output:
    Y = Output.Bundle(X.get_R(), X.get_p())

    # Noise
    V = OutputTangent.Identity()
    if OPTITRACK_NOISE:
        coeffs = np.array([np.random.normal(0,σ_sim_Rw,3),
                           np.random.normal(0,σ_sim_pw,3)]).flatten()
        V = OutputTangent(coeffs)
    return Y+V


X_sim_list = []  # List of simulated states
U_sim_list = []  # List of simulated Input (IMU measurement)
Y_sim_list = []  # List of simulated Ouput (Optitrack measurement)
X_pre_list = []  # List of estimated states after prediction
P_pre_list = []  # List of estimated covariance after prediction
X_cor_list = []  # List of estimated states after correction
P_cor_list = []  # List of estimated covariance after correction
Z_list = []  # List of innovation
H_list = []  # List of expectation
T_imu  = []
T_ot   = []


if __name__ == "__main__":
    '''Initialisation'''
    dt_imu = 1/IMU_RATE  # [s]
    dt_ot = 1/OPTITRACK_RATE  # [s]

    t_imu = 0  # imu tracking time [s]
    t_ot = 0  # ot tracking time [s]
    t = 0  # global tracking [s]

    # Definig initial state and covariances as lie ektended kalman filter class
    lekf = LEKF(X0, P0, Q0, W0, V0)

    '''Simulation loop'''
    while t != TIME:
        t = min(t_imu, t_ot, TIME)
        if t >= t_imu:
            '''Imu data'''
            T_imu.append(t_imu)
            X, U = update(X, t_imu, dt_imu)
            X_sim_list.append(X)  # storing simulated values of X
            U_sim_list.append(U)  # storing simulated values of u (IMU)

            # Prediction
            if DO_PREDICTION:
                lekf.predict(U, dt_imu)
                X_pre_list.append(lekf.X)  # storing estimated values of X
                P_pre_list.append(lekf.P)  # storing estimated values of P

            t_imu = t_imu + dt_imu

        if t >= t_ot:
            '''Optitrack data'''
            T_ot.append(t_imu)

            Y = observe(X)
            Y_sim_list.append(Y)

            # Correction
            if DO_CORRECTION:
                e, _, _ = lekf.h(lekf.X, OutputTangent.Identity())
                H_list.append(e)
                z, _, _ = lekf.z(Y)
                Z_list.append(z)
                lekf.correct(Y)
                X_cor_list.append(lekf.X)
                P_cor_list.append(lekf.P)

            t_ot = t_ot + dt_ot

    from matplotlib import pyplot as plt
    x = [X.get_p().coeffs()[0] for X in X_sim_list]
    y = [X.get_p().coeffs()[1] for X in X_sim_list]
    xe = [X.get_p().coeffs()[0] for X in X_pre_list]
    ye = [X.get_p().coeffs()[1] for X in X_pre_list]
    plt.plot(x,y, xe, ye)
    plt.show()

    dist = [np.linalg.norm((X.get_p()-Xe.get_p()).coeffs()) for X, Xe in zip(X_sim_list,X_pre_list)]
    plt.plot(T_imu,dist)
    plt.show()

    dist = [np.linalg.norm((X._coeffs-Xe._coeffs)) for X, Xe in zip(X_sim_list,X_pre_list)]
    plt.plot(T_imu,dist)
    plt.show()

    z_roll  = [zi.get_ΔRm().coeffs()[0] for zi in Z_list]
    z_pitch = [zi.get_ΔRm().coeffs()[1] for zi in Z_list]
    z_yaw   = [zi.get_ΔRm().coeffs()[2] for zi in Z_list]
    z_x = [zi.get_Δpm().coeffs()[0] for zi in Z_list]
    z_y = [zi.get_Δpm().coeffs()[1] for zi in Z_list]
    z_z = [zi.get_Δpm().coeffs()[2] for zi in Z_list]

    plt.plot( T_ot, z_pitch, color='red', label='Z_R_x', ls = "-")
    plt.plot( T_ot ,z_roll, color='green', label='Z_R_y',ls = "-")
    plt.plot( T_ot ,z_yaw, color='blue', label='Z_R_z',ls = "-")

    plt.plot( T_ot ,z_x, color='red', label='Z_x_p',ls = ":")
    plt.plot( T_ot ,z_y, color='green', label='Z_y_p',ls = ":")
    plt.plot( T_ot ,z_z, color='blue', label='Z_z_p',ls = ":")

    plt.legend()
    plt.show()
