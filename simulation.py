#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
from manifpy import SO3, SO3Tangent

from lekf import LEKF
import state
import measurement

# RATE & TIME
_IMU_RATE = 1000  # [Hz]
_OPTITRACK_RATE = 100  # [Hz]
_TIME = 15  # [s]

# NOISE
_IMU_NOISE = True
_OPTITRACK_NOISE = True

# Sigmas
w_sigmas = measurement.ImuNoise(6.3e-5*np.ones(3), 8.7e-5*np.ones(3),
                                4e-4*np.ones(3), 3.9e-5*np.ones(3))
v_sigmas = measurement.OptitrackNoise(
    SO3Tangent(0.001*np.ones(SO3.DoF)), 0.001*np.ones(3))

# Initialitzation of covariances
# first sigma, then sigma² = variance, and then the covariance matrix.

# Covariance of the State

p_sigmas = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
# [std_error R,std_error v,std_error p,std_error ab,std_error wb ]

P0 = np.diagflat(np.square(p_sigmas))

# Covariance of the Measurement IMU
q_sigmas = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
# [std_error ab,std_error ωb]

Q0 = np.diagflat(np.square(q_sigmas))

# Covariance of the IMU bias
w_joan_sigmas = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
# [std_error R,std_error v]

W0 = np.diagflat(np.square(w_joan_sigmas))

# Covariance of the Measurment Optitrack
v_joan_sigmas = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
# [std_error R,std_error v,std_error p,std_error ab,std_error wb ]

V0 = np.diagflat(np.square(v_joan_sigmas))

g = np.array([0, 0, -9.81])


def update(X, U_t, dt):
    U = measurement.ImuMeasurement()
    Un = measurement.ImuNoise()
    X_o = state.State()
    if _IMU_NOISE:
        Un.a_wn = w_sigmas.a_wn*np.random.uniform(-1,1,3) # random value between -1 and 1 for each var.
        Un.ω_wn = w_sigmas.ω_wn*np.random.uniform(-1,1,3)
        Un.a_rw = w_sigmas.a_rw*np.random.uniform(-1,1,3)
        Un.ω_rw = w_sigmas.ω_rw*np.random.uniform(-1,1,3)
    # Command U
    U.a_m = X.R.inverse().act(U_t.a_m-g)+X.a_b+Un.a_wn
    U.ω_m = U_t.ω_m + X.ω_b + Un.ω_wn
    # New state X
    X_o.R = X.R.rplus(SO3Tangent(U_t.ω_m*dt))
    X_o.v = X.v + U_t.a_m*dt
    X_o.p = X.p + X.v*dt + U_t.a_m*((dt**2)/2)
    X_o.a_b = X.a_b + Un.a_rw
    X_o.ω_b = X.ω_b + Un.ω_rw
    return X_o, U


def observe(X):
    Y = measurement.OptitrackMeasurement()
    Yn = measurement.OptitrackNoise()
    if _OPTITRACK_NOISE:
        Yn.R_wn = SO3Tangent(v_sigmas.R_wn.coeffs_copy()*np.random.uniform(-1,1,3)) # random value between -1 and 1 for each var.
        Yn.p_wn = v_sigmas.p_wn*np.random.uniform(-1,1,3)
    Y.R_m = X.R+Yn.R_wn
    Y.p_m = X.p+Yn.p_wn
    return Y

X_list = [] # List to stock simulated states
U_list = [] # List to stock simulated control (IMU measurement)
Y_list = [] # List to stock simulated OptiTrack 
X_est_list = [] # List to stock estimated states
P_est_list = [] # List to stock estimated covariance

if __name__ == "__main__":
    '''Initialisation'''
    dt_imu = 1/_IMU_RATE  # [s]
    dt_ot = 1/_OPTITRACK_RATE  # [s]
    dt = dt_imu*dt_ot  # [s]

    t_imu = 0  # imu tracking time [s]
    t_ot = 0  # ot tracking time [s]
    t = 0  # global tracking [s]

    X = state.State(SO3.Identity(), np.array([0, 1, 0]), np.zeros(3),
                    np.zeros(3), np.zeros(3)) #inicial state. We define the velocity on vy beacuse the circle we want to simulate.

    lekf = LEKF(X, P0, Q0, W0, V0) # Definig initial state and covariances as lie ektended kalman filter class

    '''Simulation loop'''
    for t in np.arange(0, _TIME, dt): 

        if t >= t_imu:
            '''Imu data'''
            # True acc & angular velocity
            U_t = measurement.ImuMeasurement() # Inicialitzating of U(t) = [0(3x3), 0(3x3)]
            U_t.a_m = np.array([-np.cos(t_imu), -np.sin(t_imu), 0]) #Expressing a circle around z axis by the accel.
            U_t.ω_m = np.array([0, 0, 1]) #rotation around z.
            X, U = update(X, U_t, dt_imu)
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
            #X_est_list.append(lekf.X)
            #P_est_list.append(lekf.P)

            t_ot = t_ot + dt_ot

    '''Data process'''

import get_x_n_y

x_r, y_r = get_x_n_y.get_X_n_Y(X_list)
x_est, y_est = get_x_n_y.get_X_n_Y(X_est_list)

# Plotting both the curves simultaneously
plt.plot(x_r, y_r, color='red', label='real', ls = "-")
plt.plot(x_est, y_est, color='blue', label='estimated',ls = ":")

plt.legend()
plt.show()