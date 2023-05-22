
import numpy as np
from manifpy import *

from measurement import *
from state import *

_ZERO = np.zeros([3, 3])


class LEKF:
    # Constructor

    def __init__(self, x0, P0, Q, W, V) -> None:
        self.x = x0  # [State]
        self.P = P0  # [Covariance]
        self.Q = Q  # [Covariance]
        self.W = W  # [Covariance]
        self.V = V  # [Covariance]

    # User function

    def predict(self, u, dt):

        # Initialize noise vector w with all the noises, White and random Walk, at 0.
        # As the atributes are expected to be 3x3 array, the np.zero of dimension 3 is used.

        w = ImuBiasRandomWalk()

        # The prediction is based on the dynamics function With 0 noise.
        # The obtained values are the estimated x and the jacobians of the dynamcs Wrt its inputs.
        xo, J_xo_x, J_xo_u, J_xo_w = self.f(self.x, u, w, dt)

        # The prediction step also has to upload the value of the covariance matrix P (matrix of the state).
        # We do this by taking in account all the inputs covariances and the jacobians mentioned before.
        # P_out = J_xo_x @ self.P @ J_xo_x.transpose() + J_xo_u @ self.Q @ J_xo_u.transpose() + \
        #     J_xo_w @ self.w @ J_xo_w.transpose()
        P_out = J_xo_x @ self.P @ J_xo_x.transpose() + J_xo_u @ self.Q @ J_xo_u.transpose() +J_xo_w @ self.W @ J_xo_w.transpose()

        self.x = xo  # We upload the estimated state value.
        # We upload the covariance matrix of the estimated value.
        # self.P = P_out
        self.P = (P_out+P_out.T)/2

    def correct(self, y):
        # innovation
        z, J_z_x, J_z_y = self.innovation(y)
        Z = J_z_x @ self.P @ J_z_x.transpose() + J_z_y @ self.V @ J_z_y.transpose()

        # kalman gain
        K = - self.P @ J_z_x.transpose() @ np.linalg.inv(Z)

        # update step
        dx = K @ z
        
        dR = dx[0:3]
        dv = dx[3:6]
        dp = dx[6:9]
        dab = dx[9:12]
        dωb = dx[12:15]

        # update
        Ro  = self.x.R.rplus(SO3Tangent(dR))
        po  = self.x.p + dp
        vo  = self.x.v + dv
        abo = self.x.a_b + dab
        ωbo = self.x.ω_b + dωb
        
        xo = State(Ro, vo, po, abo, ωbo)

        P_out = self.P - K @ Z @ K.transpose()

        # output
        self.x = xo
        # self.P = P_out
        self.P = (P_out+P_out.T)/2

    # Acceleration
    def a(self, x, u):
        g = np.array([0, 0, -9.81])
        # compute the true a
        # remove bias and Whiet noise from measurement
        # equation
        Δa = u.a_m - x.a_b - u.a_w
        # jacobian
        J_Δa_am = np.identity(len(Δa))  # Jacobian of the difference Wrt a_m
        J_Δa_ab = -np.identity(len(Δa))  # Jacobian of the difference Wrt a_b
        J_Δa_aWn = -np.identity(len(Δa))  # Jacobian of the difference Wrt a_Wn
        # Change the frame of the IMU from Wrt body to Wrt World frame
        # Defining the Jacobians for manif to compute.
        J_RΔa_R = np.ndarray([x.R.DoF, x.R.DoF])
        # Defining the Jacobians for manif to compute.
        J_RΔa_Δa = np.ndarray([x.R.DoF, Δa.size])

        # Computing the frame change
        # a = R * ( Δa ) + g. The result is a 3d vector.
        a = x.R.act(Δa, J_RΔa_R, J_RΔa_Δa) + g

        # jacobian

        # From the paper: J_R*v_R = ... = -R[v]x. In this case J_a_R = -R[ da ]x = -R*[a_m -a_b]x
        J_a_R = J_RΔa_R

        # J_a_am = J_R(am-ab-aWn)+g_(am-ab-aWn) @ J_(am-ab-aWn)_am
        J_a_am = J_RΔa_Δa @ J_Δa_am

        # J_a_ab = J_R(am-ab-aWn)+g_(am-ab-aWn) @ J_(am-ab-aWn)_ab
        J_a_ab = J_RΔa_Δa @ J_Δa_ab

        # J_a_aWn = J_R(am-ab-aWn)+g_(am-ab-aWn) @ J_(am-ab-aWn)_aWn
        J_a_aWn = J_RΔa_Δa @ J_Δa_aWn

        return a, J_a_R, J_a_am, J_a_ab, J_a_aWn

    # Omega

    def ω(self, x, u):

        # compute true value ω
        # remove bias and Whiet noise from measurement
        # equation
        # w = W_m - W_b defining Wmeasured and Wbias as a 3D vector.
        # J_ω_R = np.zeros([SO3.DoF,SO3.DoF])
        J_ω_ωm = np.zeros(3)
        # ω = x.R.act(u.ω_m,J_ω_R,J_ω_ωm) - x.ω_b - w.ω_Wn
        ω = u.ω_m - x.ω_b - u.ω_w
        # jacobian
        J_ω_ωm = np.identity(3)
        J_ω_ωb = -np.identity(3)
        J_ω_ωWn = -np.identity(3)

        return ω, J_ω_ωm, J_ω_ωb, J_ω_ωWn

    # Dynamics of system

    def f(self, x, u, w, dt):

        # input and output states
        X_i = State(x.R, x.v, x.p, x.a_b, x.ω_b)
        X_o = X_i

        # compute real values of u
        a, J_a_R, J_a_am, J_a_ab, _ = self.a(X_i, u)  # real a and its jacobians
        ω, J_ω_ωm, J_ω_ωb, _ = self.ω(X_i, u)  # real ω and its jacobians

        # compute value
        # equation
        # Defining the Jacobians for manif to compute.
        J_Ro_R = np.ndarray([SO3.DoF, X_i.R.DoF]) # J_Roωdt_R
        # Defining the Jacobians for manif to compute.
        J_Ro_ωdt = np.ndarray([SO3.DoF, ω.size]) # J_Roωdt_ωdt

        X_o.R = X_i.R.rplus(SO3Tangent(ω*dt), J_Ro_R, J_Ro_ωdt)  # R (+) w*dt = RExp(Wdt)
        # jacobian
        # We computed the jacobian of the plus operation Wrt to Wdt, not w. So lets compute jacobian of Wdt Wrt w and then apply chain rule.
        
        #J_Expωdt_ωdt = SO3Tangent(ω*dt).rjac()

        J_ωdt_ω = dt*np.identity(3) 
        
        J_Ro_ω = J_Ro_ωdt @ J_ωdt_ω 
        #J_Ro_ω = J_Ro_ωdt @ J_Expωdt_ωdt @ J_ωdt_ω  # Chain rule
        
        J_R_ωm = J_Ro_ω @ J_ω_ωm  # Chain rule

        # equation
        X_o.v = X_i.v + a*dt  # v =  v + a*dt
        # jacobian
        J_v_v = np.identity(3)
        J_v_a = dt*np.identity(3)
        # equation
        X_o.p = X_i.p + X_i.v*dt + 0.5*a*dt**2  # p =  p + v*dt + 0.5*a*dt²
        # jacobian
        J_p_p = np.identity(3)
        J_p_v = dt*np.identity(3)
        J_p_a = 0.5*np.identity(3)*dt**2 + J_v_a*dt # J_p_a = J_
        # equation
        X_o.a_b = X_i.a_b + w.a_rw  # a_b =  a_b + a_r
        # jacobian
        J_ab_ab = np.identity(3)
        J_ab_ar = np.identity(3)
        # equation
        X_o.ω_b = X_i.ω_b + w.ω_rw  # ω_b =  ω_b + ω_r
        # jacobian
        J_ωb_ωb = np.identity(3)
        J_ωb_ωr = np.identity(3)

        # Assemble big jacobians: J_f_x, J_f_u, J_f_r

        # Jacobians of R Wrt state vars
        J_R_ωb = J_Ro_ω @ J_ω_ωb

        # Jacobians of v Wrt state vars

        # J_v+adt_adt @ J_adt_a @ J_R(am-ab)+g_R = J_v+adt_a @ J_R(am-ab)+g_R
        J_v_R = J_v_a @ J_a_R

        J_v_ab = J_v_a @ J_a_ab

        # Jacobians of p Wrt state vars
        J_p_R = J_p_a @ J_a_R
        J_p_ab = J_p_a @ J_a_ab

        # J_f_x = np.array([[J_Ro_R,  _ZERO,   _ZERO,     _ZERO,     J_R_ωb],
        #                   [   J_v_R, J_v_v,    _ZERO,     J_v_ab,     _ZERO],
        #                   [   J_p_R, J_p_v,    J_p_p,     J_p_ab,     _ZERO],
        #                   [    _ZERO,  _ZERO,  _ZERO,     J_ab_ab,    _ZERO],
        #                   [    _ZERO,  _ZERO,  _ZERO,     _ZERO,    J_ωb_ωb]])

        J_f_x = np.zeros([15, 15])
        J_f_x[0:3, 0:3] = J_Ro_R
        J_f_x[0:3, 12:15] = J_R_ωb
        J_f_x[3:6, 0:3] = J_v_R
        J_f_x[3:6, 3:6] = J_v_v
        J_f_x[3:6, 9:12] = J_v_ab
        J_f_x[6:9, 0:3] = J_p_R
        J_f_x[6:9, 3:6] = J_p_v
        J_f_x[6:9, 6:9] = J_p_p
        J_f_x[6:9, 9:12] = J_p_ab
        J_f_x[9:12, 9:12] = J_ab_ab
        J_f_x[12:15, 12:15] = J_ωb_ωb

        # Jacobians of v Wrt IMU measurments
        J_v_am = J_v_a @ J_a_am

        # Jacobians of p Wrt IMU measurments
        J_p_am = J_p_a @ J_a_am

        # J_f_u = np.array([[_ZERO,  J_R_ωm],
        #                   [J_v_am,   _ZERO],
        #                   [J_p_am,   _ZERO],
        #                   [_ZERO,   _ZERO],
        #                   [_ZERO,   _ZERO]])

        J_f_u = np.zeros([15, 6])
        J_f_u[0:3, 3:6] = J_R_ωm
        J_f_u[3:6, 0:3] = J_v_am
        J_f_u[6:9, 0:3] = J_p_am

        # J_f_W = np.array([[_ZERO,    _ZERO],
        #                  [_ZERO,    _ZERO],
        #                  [_ZERO,    _ZERO],
        #                  [J_ab_ar,    _ZERO],
        #                  [_ZERO,  J_ωb_ωr]])

        J_f_W = np.zeros([15, 6])
        J_f_W[9:12, 0:3]  = J_ab_ar
        J_f_W[12:15, 3:6] = J_ωb_ωr

        return X_o, J_f_x, J_f_u, J_f_W

    # Expectation

    def expectation(self, x):

        # orientation
        eR     = x.R
        # jacobian
        J_eR_R = np.identity(3)

        # postion
        ep = x.p
        # jacobian
        J_ep_p = np.identity(3)

        e = OptitrackMeasurement(eR, ep)

        J_e_x = np.zeros([6, 15])

        J_e_x[0:3, 0:3] = J_eR_R
        J_e_x[3:6, 6:9] = J_ep_p

        return e, J_e_x

    # Correction

    def innovation(self, y):
        v = OptitrackNoise()

        # expectation
        e, J_e_x = self.expectation(self.x)

        # Defining the Jacobians for manif to compute.
        J_zR_yR = np.ndarray([SO3.DoF, SO3.DoF])
        # Defining the Jacobians for manif to compute.
        J_zR_eR = np.ndarray([SO3.DoF, SO3.DoF])

        # orientation
        # R_z = y.R (-) Ye.R_m
        zR = y.R.rminus(e.R, J_zR_yR, J_zR_eR)

        # position
        zp = y.p - e.p
        J_zp_yp =  np.identity(3)
        J_zp_ep = -np.identity(3)

        # chain rule
        J_z_e = np.zeros([6,6])
        J_z_e[0:3,0:3] = J_zR_eR
        J_z_e[3:6,3:6] = J_zp_ep
        
        J_z_y = np.zeros([6,6])
        J_z_y[0:3,0:3] = J_zR_yR
        J_z_y[3:6,3:6] = J_zp_yp

        # output        
        z = np.zeros([6])
        z[0:3] = zR.coeffs()
        z[3:6] = zp

        J_z_x = J_z_e @ J_e_x
        
        return z, J_z_x, J_z_y


# Observation system
