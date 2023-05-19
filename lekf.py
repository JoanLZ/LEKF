
import numpy as np
from manifpy import *

from measurement import *
from state import *

_ZERO = np.zeros([3, 3])


class LEKF:
    # Constructor

    def __init__(self, X0, P0, Q, W, V) -> None:
        self.X = X0  # [State]
        self.P = P0  # [Covariance]
        self.Q = Q  # [Covariance]
        self.W = W  # [Covariance]
        self.V = V  # [Covariance]

    # User function

    def predict(self, U, dt):

        # Initialize covariance matrix W whit all the noises, white and random walk, at 0.
        # As the atributes are expected to be 3x3 array, the np.zero of dimension 3 is used.

        W = ImuNoise()

        # The prediction is based on the dynamics function with 0 noise.
        # The obtained values are the estimated X and the jacobians of the dynamcs wrt its inputs.
        X_out, F_x, F_u, F_w = self.f(self.X, U, W, dt)

        # The prediction step also has to upload the value of the covariance matrix P (matrix of the state).
        # We do this by taking in account all the inputs covariances and the jacobians mentioned before.
        # P_out = F_x @ self.P @ F_x.transpose() + F_u @ self.Q @ F_u.transpose() + \
        #     F_w @ self.W @ F_w.transpose()
        P_out = F_x @ self.P @ F_x.transpose() #+ F_u @ self.Q @ F_u.transpose() #+F_w @ self.W @ F_w.transpose()

        self.X = X_out  # We upload the estimated state value.
        # We upload the covariance matrix of the estimated value.
        self.P = P_out

    def correct(self, Y):
        z, J_z_x, J_z_v = self.z(Y)

        Z = J_z_x @ self.P @ J_z_x.transpose() + J_z_v @ self.V @ J_z_v.transpose()

        K = - self.P @ J_z_x.transpose() @ np.linalg.inv(Z)

        R_out = self.X.R + \
            SO3Tangent(K[:3, :3]@z.R_wn.coeffs_copy() + K[:3, 3:]@z.p_wn)
        v_out = self.X.v + K[3:6, :3]@z.R_wn.coeffs_copy() + K[3:6, 3:]@z.p_wn
        p_out = self.X.p + K[6:9, :3]@z.R_wn.coeffs_copy() + K[6:9, 3:]@z.p_wn
        ab_out = self.X.a_b + \
            K[9:12, :3]@z.R_wn.coeffs_copy() + K[9:12, 3:]@z.p_wn
        ωb_out = self.X.ω_b + \
            K[12:, :3]@z.R_wn.coeffs_copy() + K[12:, 3:]@z.p_wn

        X_out = State(R_out, v_out, p_out, ab_out, ωb_out)

        P_out = self.P - K @ Z @ K.transpose()

        self.X = X_out
        self.P = P_out

    # Acceleration

    def a(self, X, U, W):
        g = np.array([0, 0, -9.81])
        # compute the true a
        # remove bias and whiet noise from measurement
        # equation
        Δa = U.a_m-X.a_b-W.a_wn
        # jacobian
        J_Δa_am = np.identity(len(Δa))  # Jacobian of the difference wrt a_m
        J_Δa_ab = -np.identity(len(Δa))  # Jacobian of the difference wrt a_b
        J_Δa_awn = -np.identity(len(Δa))  # Jacobian of the difference wrt a_wn
        # Change the frame of the IMU from wrt body to wrt world frame
        # Defining the Jacobians for manif to compute.
        J_RΔa_R = np.ndarray([X.R.DoF, X.R.DoF])
        # Defining the Jacobians for manif to compute.
        J_RΔa_Δa = np.ndarray([X.R.DoF, Δa.size])

        # Computing the frame change
        # a = R * ( Δa ) + g. The result is a 3d vector.
        a = X.R.act(Δa, J_RΔa_R, J_RΔa_Δa) + g

        # jacobian

        # From the paper: J_R*v_R = ... = -R[v]x. In this case J_a_R = -R[ da ]x = -R*[a_m -a_b]x
        J_a_R = J_RΔa_R

        # J_a_am = J_R(am-ab-awn)+g_(am-ab-awn) @ J_(am-ab-awn)_am
        J_a_am = J_RΔa_Δa @ J_Δa_am

        # J_a_ab = J_R(am-ab-awn)+g_(am-ab-awn) @ J_(am-ab-awn)_ab
        J_a_ab = J_RΔa_Δa @ J_Δa_ab

        # J_a_awn = J_R(am-ab-awn)+g_(am-ab-awn) @ J_(am-ab-awn)_awn
        J_a_awn = J_RΔa_Δa @ J_Δa_awn

        return a, J_a_R, J_a_am, J_a_ab, J_a_awn

    # Omega

    def ω(self, X, U, W):

        # compute true value ω
        # remove bias and whiet noise from measurement
        # equation
        # w = w_m - w_b defining wmeasured and wbias as a 3D vector.
        # J_ω_R = np.zeros([SO3.DoF,SO3.DoF])
        J_ω_ωm = np.zeros(3)
        # ω = X.R.act(U.ω_m,J_ω_R,J_ω_ωm) - X.ω_b - W.ω_wn
        ω = U.ω_m - X.ω_b - W.ω_wn
        # jacobian
        J_ω_ωm = np.identity(3)
        J_ω_ωb = -np.identity(3)
        J_ω_ωwn = -np.identity(3)

        return ω, J_ω_ωm, J_ω_ωb, J_ω_ωwn

    # Dynamics of system

    def f(self, X, U, W, dt):

        X_o = State(X.R, X.v, X.p, X.a_b, X.ω_b)
        # compute real values of U
        a, J_a_R, J_a_am, J_a_ab, _ = self.a(X_o, U, W)  # real a and its jacobians
        ω, J_ω_ωm, J_ω_ωb, _ = self.ω(X_o, U, W)  # real ω and its jacobians
        # compute value
        # equation
        # Defining the Jacobians for manif to compute.
        J_RExp_R = np.ndarray([SO3.DoF, X_o.R.DoF]) # J_RExpωdt_R
        # Defining the Jacobians for manif to compute.
        J_RExp_ωdt = np.ndarray([SO3.DoF, ω.size]) # J_RExpωdt_ωdt
        X_o.R = X_o.R.rplus(SO3Tangent(ω*dt), J_RExp_R,
                          J_RExp_ωdt)  # R (+) w*dt = RExp(wdt)
        # jacobian
        # We computed the jacobian of the plus operation wrt to wdt, not w. So lets compute jacobian of wdt wrt w and then apply chain rule.
        
        #J_Expωdt_ωdt = SO3Tangent(ω*dt).rjac()

        J_ωdt_ω = dt*np.identity(3) 
        
        J_RExp_ω = J_RExp_ωdt @ J_ωdt_ω 
        #J_RExp_ω = J_RExp_ωdt @ J_Expωdt_ωdt @ J_ωdt_ω  # Chain rule
        
        J_R_ωm = J_RExp_ω @ J_ω_ωm  # Chain rule

        # equation
        X_o.v = X_o.v + a*dt  # v =  v + a*dt
        # jacobian
        J_v_v = np.identity(3)
        J_v_a = dt*np.identity(3)
        # equation
        X_o.p = X_o.p + X_o.v*dt + 0.5*a*dt**2  # p =  p + v*dt + 0.5*a*dt²
        # jacobian
        J_p_p = np.identity(3)
        J_p_v = dt*np.identity(3)
        J_p_a = 0.5*np.identity(3)*dt**2 + J_v_a*dt # J_p_a = J_
        # equation
        X_o.a_b = X_o.a_b + W.a_rw  # a_b =  a_b + a_r
        # jacobian
        J_ab_ab = np.identity(3)
        J_ab_ar = np.identity(3)
        # equation
        X_o.ω_b = X_o.ω_b + W.ω_rw  # ω_b =  ω_b + ω_r
        # jacobian
        J_ωb_ωb = np.identity(3)
        J_ωb_ωr = np.identity(3)

        # Assemble big jacobians: J_f_x, J_f_u, J_f_r

        # Jacobians of R wrt state vars
        J_R_ωb = J_RExp_ω @ J_ω_ωb

        # Jacobians of v wrt state vars

        # J_v+adt_adt @ J_adt_a @ J_R(am-ab)+g_R = J_v+adt_a @ J_R(am-ab)+g_R
        J_v_R = J_v_a @ J_a_R

        J_v_ab = J_v_a @ J_a_ab

        # Jacobians of p wrt state vars
        J_p_R = J_p_a @ J_a_R
        J_p_ab = J_p_a @ J_a_ab

        # J_f_x = np.array([[J_RExp_R,  _ZERO,   _ZERO,     _ZERO,     J_R_ωb],
        #                   [   J_v_R, J_v_v,    _ZERO,     J_v_ab,     _ZERO],
        #                   [   J_p_R, J_p_v,    J_p_p,     J_p_ab,     _ZERO],
        #                   [    _ZERO,  _ZERO,  _ZERO,     J_ab_ab,    _ZERO],
        #                   [    _ZERO,  _ZERO,  _ZERO,     _ZERO,    J_ωb_ωb]])

        J_f_x = np.zeros([15, 15])
        J_f_x[0:3, 0:3] = J_RExp_R
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

        # Jacobians of v wrt IMU measurments
        J_v_am = J_v_a @ J_a_am

        # Jacobians of p wrt IMU measurments
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

        # J_f_w = np.array([[_ZERO,    _ZERO],
        #                  [_ZERO,    _ZERO],
        #                  [_ZERO,    _ZERO],
        #                  [J_ab_ar,    _ZERO],
        #                  [_ZERO,  J_ωb_ωr]])

        J_f_w = np.zeros([15, 6])
        J_f_w[9:12, 0:3] = J_ab_ar
        J_f_w[12:15, 3:6] = J_ωb_ωr

        return X_o, J_f_x, J_f_u, J_f_w

    # Expectation

    def h(self, X, V):
        # compute value
        # equation
        # Defining the Jacobians for manif to compute.
        J_Re_R = np.ndarray([SO3.DoF, SO3.DoF])
        # Defining the Jacobians for manif to compute.
        J_Re_Rwn = np.ndarray([SO3Tangent.DoF, SO3Tangent.DoF])
        # R_e = R (+) R_wn
        R_e = X.R.rplus(V.R_wn, J_Re_R, J_Re_Rwn)

        # equation
        p_e = X.p + V.p_wn
        # jacobian
        J_pe_p = np.identity(3)
        J_pe_pwn = np.identity(3)

        Y_e = OptitrackMeasurement(R_e, p_e)

        # J_h_x = np.array([[J_Re_R, _ZERO, _ZERO, _ZERO, _ZERO],
        #                   [_ZERO, _ZERO, J_pe_p, _ZERO, _ZERO]])

        J_h_x = np.zeros([6, 15])

        J_h_x[0:3, 0:3] = J_Re_R
        J_h_x[3:6, 6:9] = J_pe_p

        # J_h_v = np.array([[J_Re_Rwn, _ZERO],
        #                  [_ZERO, J_pe_pwn]])

        J_h_v = np.zeros([6, 6])

        J_h_v[0:3, 0:3] = J_Re_Rwn
        J_h_v[3:6, 3:6] = J_pe_pwn

        return Y_e, J_h_x, J_h_v

    # Correction

    def z(self, Y):
        V = OptitrackNoise()
        Y_e, H_x, H_v = self.h(self.X, V)

        # Defining the Jacobians for manif to compute.
        J_Rz_Rm = np.ndarray([SO3.DoF, SO3.DoF])
        # Defining the Jacobians for manif to compute.
        J_Rz_Re = np.ndarray([SO3.DoF, SO3.DoF])

        # R_z = Y.R (-) Ye.R_m
        R_z = Y.R_m.rminus(Y_e.R_m, J_Rz_Rm, J_Rz_Re)

        J_Re_R = H_x[0:3, 0:3]
        J_Rz_R = J_Rz_Re @ J_Re_R

        J_Re_Rwn = H_v[0:3, 0:3]
        J_Rz_Rwn = J_Rz_Re @ J_Re_Rwn

        # Z.p = Y.p_m - R_wn
        p_z = Y.p_m - Y_e.p_m
        J_pz_pe = -np.identity(3)

        J_pe_p = H_x[3:6, 6:9]
        J_pz_p = J_pz_pe @ J_pe_p

        J_pe_pwn = H_v[3:6, 3:6]
        J_pz_pwn = J_pz_pe @ J_pe_pwn

        z = OptitrackNoise(R_z, p_z)

        # J_z_x = np.array([[J_Rz_R, _ZERO,  _ZERO, _ZERO, _ZERO],
        #                   [ _ZERO, _ZERO, J_pz_p, _ZERO, _ZERO]])

        J_z_x = np.zeros([6, 15])

        J_z_x[0:3, 0:3] = J_Rz_R
        J_z_x[3:6, 6:9] = J_pz_p

        # J_z_v = np.array([[J_Rz_Rwn,    _ZERO],
        #                   [   _ZERO, J_pz_pwn]])

        J_z_v = np.zeros([6, 6])

        J_z_v[0:3, 0:3] = J_Rz_Rwn
        J_z_v[3:6, 3:6] = J_pz_pwn

        return z, J_z_x, J_z_v


# Observation system
