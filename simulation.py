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
_TIME = 120 # [s]

# SET NOISE
_IMU_NOISE = True
_OPTITRACK_NOISE = True

# SET CORRECTION
_CORRECTION = True

# GRAVITY
g = np.array([0, 0, -9.81])

# Sigmas for the simulated noise
# Noise inside IMU commands. White and Random Walk
Sigma_W_a_wn = 6.3e-5 #m/s²
Sigma_W_ω_wn = 8.7e-5 #rad/s
Sigma_W_a_rw = 0#4e-4   #m/s²
Sigma_W_ω_rw = 0# 3.9e-5 #rad/s

w_sigmas = measurement.ImuNoise(Sigma_W_a_wn*np.ones(3), Sigma_W_ω_wn*np.ones(3), # a_wn, w_wn
                                Sigma_W_a_rw*np.ones(3), Sigma_W_ω_rw *np.ones(3))   # a_rw, w_wr

# Noise inside OptiTrack Measurments 
v_sigmas = measurement.OptitrackNoise(SO3Tangent(0.001*np.ones(SO3.DoF)), 0.001*np.ones(3)) # R_wn, p_wn

# Initialitzation of covariances first sigma (std. deviation), then sigma² (variance), and then the covariance matrix.

# Sigmas of P
Sigma_P_R = 1e-1; # Initial Value of std. deviation of the orientation.
Sigma_P_v = 1e-1; # Initial Value of std. deviation of the lineal velocity.
Sigma_P_p = 1e-1; # Initial Value of std. deviation of the position.
Sigma_P_ab = 0#1e-4; # Initial Value of std. deviation of the lineal acceleration bias.
Sigma_P_ωb = 0#1e-4; # Initial Value of std. deviation of the angular velocity bias.

P_sigmas = np.array([Sigma_P_R, Sigma_P_R, Sigma_P_R, Sigma_P_v, Sigma_P_v, Sigma_P_v, Sigma_P_p, Sigma_P_p, Sigma_P_p, Sigma_P_ab, Sigma_P_ab, Sigma_P_ab, Sigma_P_ωb, Sigma_P_ωb, Sigma_P_ωb])

# Covariance matrix P

P0 = np.diagflat(np.square(P_sigmas))

# Sigmas of Q
Sigma_Q_a_wn = Sigma_W_a_wn
Sigma_Q_ω_wn = Sigma_W_ω_wn

Q_sigmas = np.array([Sigma_Q_a_wn, Sigma_Q_a_wn, Sigma_Q_a_wn, Sigma_Q_ω_wn, Sigma_Q_ω_wn, Sigma_Q_ω_wn])

# Covariance matrix Q

Q0 = np.diagflat(np.square(Q_sigmas))

# Sigmas of W

W_joan_sigmas = np.array([Sigma_W_a_rw, Sigma_W_a_rw, Sigma_W_a_rw, Sigma_W_ω_rw, Sigma_W_ω_rw, Sigma_W_ω_rw])

# Covariance matrix W

W0 = np.diagflat(np.square(W_joan_sigmas))

# Sigmas of V
Sigma_V_R = 4e-3
Sigma_V_ω = 1e-2

v_joan_sigmas = np.array([Sigma_V_R , Sigma_V_R , Sigma_V_R , Sigma_V_ω, Sigma_V_ω, Sigma_V_ω])

# Covariance matrix V

V0 = np.diagflat(np.square(v_joan_sigmas))

# Updadte function simulates the inputs of the measurments done by IMU.

def update(X, U_t, dt):
    U = measurement.ImuMeasurement()
    Un = measurement.ImuNoise()
    X_o = state.State()
    if _IMU_NOISE:
        # Un.a_wn = w_sigmas.a_wn*np.random.uniform(-1,1,3) # random value between -1 and 1 for each var.
        # Un.ω_wn = w_sigmas.ω_wn*np.random.uniform(-1,1,3)
        # Un.a_rw = w_sigmas.a_rw*np.random.uniform(-1,1,3)
        # Un.ω_rw = w_sigmas.ω_rw*np.random.uniform(-1,1,3)
        Un.a_wn = np.random.uniform(0,w_sigmas.a_wn,3) 
        Un.ω_wn = np.random.uniform(0,w_sigmas.ω_wn,3)
        Un.a_rw = np.random.uniform(0,w_sigmas.a_rw,3)
        Un.ω_rw = np.random.uniform(0,w_sigmas.ω_rw,3)
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
        Yn.R_wn = SO3Tangent(np.random.uniform(0,v_sigmas.R_wn.coeffs_copy(),3)) 
        Yn.p_wn = np.random.uniform(0,v_sigmas.p_wn,3)
    Y.R_m = X.R+Yn.R_wn # ?
    Y.p_m = X.p+Yn.p_wn # ?
    return Y

X_list = [] # List to stock simulated states
U_list = [] # List to stock simulated control (IMU measurement)
Y_list = [] # List to stock simulated OptiTrack 
X_est_list = [] # List to stock estimated states
P_est_list = [] # List to stock estimated covariance
Z_list = [] # List of z
H_list = [] # List of h

if __name__ == "__main__":
    '''Initialisation'''
    dt_imu = 1/_IMU_RATE  # [s]
    dt_ot = 1/_OPTITRACK_RATE  # [s]
    dt = dt_imu*dt_ot  # [s]

    t_imu = 0  # imu tracking time [s]
    t_ot = 0  # ot tracking time [s]
    t = 0  # global tracking [s]

    X = state.State(SO3.Identity(), np.array([0, -1, 0]), np.zeros(3),
                    np.zeros(3), np.zeros(3)) #inicial state. We define the velocity on vy beacuse the circle we want to simulate.

    X0 = state.State(SO3.Identity(), np.array([0.1, -1.1, 0.1]), 0.01*np.ones(3),
                    np.zeros(3), np.zeros(3)) #inicial state. We define the velocity on vy beacuse the circle we want to simulate.


    lekf = LEKF(X0, P0, Q0, W0, V0) # Definig initial state and covariances as lie ektended kalman filter class

    '''Simulation loop'''
    for t in np.arange(0, _TIME, dt): 

        if t >= t_imu:
            '''Imu data'''
            # True acc & angular velocity
            U_t = measurement.ImuMeasurement() # Inicialitzating of U(t) = [0(3x3), 0(3x3)]
            U_t.a_m = np.array([-2, 3, 0]) #Expressing...
            #U_t.a_m = np.array([np.cos(t_imu), np.sin(t_imu), 0]) #Expressing a circle around z axis by the accel.
            U_t.ω_m = np.array([0, 0, 0]) #rotation around z.
            X, U = update(X, U_t, dt_imu)
            X_list.append(X) #storing real values of X
            U_list.append(U) #storing real values of u (IMU)
            lekf.predict(U, dt_imu)
            X_est_list.append(lekf.X) #sotring estimated values of X
            P_est_list.append(lekf.P) #sotring estimated values of P

            t_imu = t_imu + dt_imu
        
        if t >= t_ot:
        
            '''Optitrack data'''
        
            Y = observe(X)
        
            Y_list.append(Y)
            
            V = measurement.OptitrackNoise()

            h, _, _ = lekf.h(lekf.X, V)
            
            H_list.append(h)

            z, _, _ = lekf.z(Y)

            Z_list.append(z)

            if _CORRECTION:
                lekf.correct(Y)

            
            #X_est_list.append(lekf.X
            # )
        
            #P_est_list.append(lekf.P
            t_ot = t_ot + dt_ot

    '''Data process'''

import get_x_n_y

x_r, y_r = get_x_n_y.get_X_n_Y(X_list)
x_est, y_est = get_x_n_y.get_X_n_Y(X_est_list)

z_x = [ze_i.R_wn.coeffs()[0] for ze_i in Z_list]
z_y = [ze_i.R_wn.coeffs()[1] for ze_i in Z_list]
z_z = [ze_i.R_wn.coeffs()[2] for ze_i in Z_list]
z_x_px = [ze_i.p_wn[0] for ze_i in Z_list]
z_x_py = [ze_i.p_wn[1] for ze_i in Z_list]
z_x_pz = [ze_i.p_wn[2] for ze_i in Z_list]

h_x = [he_i.R_m.coeffs()[0] for he_i in H_list]
h_y = [he_i.R_m.coeffs()[1] for he_i in H_list]
h_z = [he_i.R_m.coeffs()[2] for he_i in H_list]
h_x_px = [he_i.p_m[0] for he_i in H_list]
h_x_py = [he_i.p_m[1] for he_i in H_list]
h_x_pz = [he_i.p_m[2] for he_i in H_list]

# d = [np.linalg.norm(xe_i.p-x_i.p) for xe_i, x_i in zip(X_est_list,X_list)]
t_imu = np.arange(0, _TIME, dt_imu)
t_ot = np.arange(0,_TIME,dt_ot)
# # Plotting both the curves simultaneously

plt.plot(x_r, y_r, color='red', label='real', ls = "-")
plt.plot(x_est, y_est, color='blue', label='estimated',ls = ":")

plt.legend()
plt.show()


P_p = [np.diag((x_i)[6:9,6:9]) for x_i in P_est_list]

P_p_x = [(x_i)[0] for x_i in P_p]
P_p_y = [(x_i)[1] for x_i in P_p]
P_p_z = [(x_i)[2] for x_i in P_p]

plt.semilogy(t_imu, P_p_x, color ='red', label='P_px', ls = "-")
plt.semilogy(t_imu, P_p_y,  color='blue', label='P_py',ls = "--")
plt.semilogy(t_imu, P_p_z, color='green', label='P_pz', ls = "-")

plt.legend()
plt.show()


# # Plotting correction (z), variables: Z.R (x,y,z) and Z.p (x,y,z)
plt.plot( t_ot, z_x, color='red', label='Z_R_x', ls = "-")
plt.plot( t_ot ,z_y, color='green', label='Z_R_y',ls = "-")
plt.plot( t_ot ,z_z, color='blue', label='Z_R_z',ls = "-")

plt.plot( t_ot ,z_x_px, color='red', label='Z_x_p',ls = ":")
plt.plot( t_ot ,z_x_py, color='green', label='Z_y_p',ls = ":")
plt.plot( t_ot ,z_x_pz, color='blue', label='Z_z_p',ls = ":")

plt.legend()
plt.show()

# Plotting (h), variables: h.R (x,y,z) and h.p(x,y,z)
plt.plot( t_ot, h_x, color='red', label='h_R_x', ls = "-")
plt.plot( t_ot ,h_y, color='green', label='h_R_y',ls = "-")
plt.plot( t_ot ,h_z, color='blue', label='h_R_z',ls = "-")

plt.plot( t_ot ,h_x_px, color='red', label='h_x_p',ls = ":")
plt.plot( t_ot ,h_x_py, color='green', label='h_y_p',ls = ":")
plt.plot( t_ot ,h_x_pz, color='blue', label='h_z_p',ls = ":")

plt.legend()
plt.show()