
import numpy as np
from manifpy import *

from measurement import *
from state import *

_ZERO = np.zeros([3, 3])



class LEKF:
    # Constructor
    def __init__(self, X0, P0, Q, W, V) -> None:
        self.X = X0 # [State]
        self.P = P0 # [Covariance]
        self.Q = Q  # [Covariance]
        self.W = W  # [Covariance]
        self.V = V  # [Covariance]

    # User function
    def predict(self, U, dt):

        W = ImuNoise(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
        X_out, F_x, F_u, F_w = self.f(self.X, U, W, dt)

        P_out = F_x @ self.P @ F_x.transpose() + F_u @ self.Q @ F_u.transpose() + \
            F_w @ self.W @ F_w.transpose()

        self.X = X_out
        self.P = P_out

    def correct(self, Y):
        
        V = OptitrackNoise(np.zeros(SO3.DoF),np.zeros(3))


    # Method

    def a(self, X, U, W):
        g = np.array([0, 0, -9.81])
        # compute the true a
        # remove bias from measurement
        # equation
        Δa = U.a_m-X.a_b-W.a_wn
        # jacobian
        J_Δa_am = np.identity(len(Δa))  # Jacobian of the differnce wrt a_m
        J_Δa_ab = -np.identity(len(Δa))  # Jacobian of the differnce wrt a_b
        J_Δa_awn = -np.identity(len(Δa))  # Jacobian of the differnce wrt a_wn
        # compute true value
        # equation
        # Defining the Jacobians for manif to compute.
        J_RΔa_R = np.ndarray([X.R.DoF, X.R.DoF])
        # Defining the Jacobians for manif to compute.
        J_RΔa_Δa = np.ndarray([X.R.DoF, Δa.size])
        # a = R * ( Δa ) + g. The result is a 3d vector.
        a = X.R.act(Δa, J_RΔa_R, J_RΔa_Δa) + g
        # jacobian
        # From the paper: J_R*v_R = ... = -R[v]x. In this case J_a_R = -R[ da ]x = -R*[a_m -a_b]x
        J_a_R = J_RΔa_R
        # J_a_am = J_R(am-ab-awn)+g_(am-ab-awn) dot J_(am-ab-awn)_am
        J_a_am = J_RΔa_Δa @ J_Δa_am
        # J_a_ab = J_R(am-ab-awn)+g_(am-ab-awn) dot J_(am-ab-awn)_ab
        J_a_ab = J_RΔa_Δa @ J_Δa_ab
        # J_a_awn = J_R(am-ab-awn)+g_(am-ab-awn) dot J_(am-ab-awn)_awn
        J_a_awn = J_RΔa_Δa @ J_Δa_awn
        return a, J_a_R, J_a_am, J_a_ab, J_a_awn

    def ω(self, X, U, W):

        # compute true value
        # equation
        # w = w_m - w_b defining wmeasured and wbias as a 3D vector.
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
        a, J_a_R, J_a_am, J_a_ab, _ = self.a(X, U, W)  # real a
        ω, J_ω_ωm, J_ω_ωb, _ = self.omega(X, U, W)  # real ω
        # compute value
        # equation
        # Defining the Jacobians for manif to compute.
        J_RExp_R = np.ndarray([SO3.DoF, X.R.DoF])
        # Defining the Jacobians for manif to compute.
        J_RExp_ωdt = np.ndarray([SO3.DoF, ω.size])
        X_o.R = X.R.rplus(SO3Tangent(ω*dt), J_RExp_R,
                          J_RExp_ωdt)  # R (+) w*dt = RExp(wdt)
        # jacobian
        # We computed the jacobian of the plus operation wrt to wdt, not w. So lets compute jacobian of wdt wrt w and then apply chain rule.
        J_ωdt_ω = dt*np.identity(3)
        J_RExp_ω = J_RExp_ωdt @ J_ωdt_ω  # Chain rule
        J_R_ωm = J_RExp_ω @ J_ω_ωm
        # equation
        X_o.v = X.v + a*dt  # v =  v + a*dt
        # jacobian
        J_v_v = np.identity(3)
        J_v_a = dt*np.identity(3)
        # equation
        X_o.p = X.p + X.v*dt + 0.5*a*dt**2  # p =  p + v*dt + 0.5*a*dt²
        # jacobian
        J_p_p = np.identity(3)
        J_p_v = dt*np.identity(3)
        J_p_a = 0.5*np.identity(3)*dt**2
        # equation
        X_o.a_b = X.a_b + W.a_rw  # a_b =  a_b + a_r
        # jacobian
        J_ab_ab = np.identity(3)
        J_ab_ar = np.identity(3)
        # equation
        X_o.ω_b = X.ω_b + W.ω_rw  # ω_b =  ω_b + ω_r
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

        # J_f_x

        J_f_x = np.array([[J_RExp_R,  _ZERO,  _ZERO,    _ZERO,  J_R_ωb],
                          [   J_v_R, J_v_v,  _ZERO,  J_v_ab,    _ZERO],
                          [   J_p_R, J_p_v, J_p_p,  J_p_ab,    _ZERO],
                          [    _ZERO,  _ZERO,  _ZERO, J_ab_ab,    _ZERO],
                          [    _ZERO,  _ZERO,  _ZERO,    _ZERO, J_ωb_ωb]])

        # Jacobians of v wrt IMU measurments
        J_v_am = J_v_a @ J_a_am

        # Jacobians of p wrt IMU measurments
        J_p_am = J_p_a @ J_a_am

        # J_f_u
        J_f_u = np.array([[  _ZERO, J_R_ωm],
                          [J_v_am,   _ZERO],
                          [J_p_am,   _ZERO],
                          [  _ZERO,   _ZERO],
                          [  _ZERO,   _ZERO]])
        # J_f_w
        J_f_w = np.array([[  _ZERO,    _ZERO],
                         [   _ZERO,    _ZERO],
                         [   _ZERO,    _ZERO],
                         [J_ab_ar,    _ZERO],
                         [   _ZERO, J_ωb_ωr]])

        return X_o, J_f_x, J_f_u, J_f_w

    def h(self, X, V):
        # Defining the Jacobians for manif to compute.
        J_Re_R = np.ndarray([SO3.DoF, SO3.DoF])
        # Defining the Jacobians for manif to compute.
        J_Re_Rwn = np.ndarray([SO3.DoF, SO3Tangent.DoF])
        # R_e = R (+) R_wn
        R_e = X.R.rplus(V.R_wn,J_Re_R,J_Re_Rwn)

        # P_e = p + p_wn
        p_e = X.p + V.p_wn

        J_pe_p = np.identity(3)
        J_pe_pwn = np.identity(3)
        
        Y_e = OptitrackMeasurement(R_e,p_e)
        
        J_h_x = np.array([[J_Re_R, _ZERO],
                          [_ZERO, J_pe_p]])
        J_h_v = np.array([[J_Re_Rwn, _ZERO],
                         [_ZERO, J_pe_pwn]])
        return Y_e, J_h_x, J_h_v
    
    def z(self, Y):
        V = OptitrackNoise(np.zeros(3), np.zeros(3))
        Y_e, H_x, H_v = self.h(self.X,V)

        # Defining the Jacobians for manif to compute.
        J_Rz_Rm = np.ndarray([SO3.DoF, SO3.DoF])
        # Defining the Jacobians for manif to compute.
        J_Rz_Re = np.ndarray([SO3.DoF, SO3.DoF])
        # Z = Y.R (+) R_wn
        R_z = Y.R_m.rminus(Y_e.R_m,J_Rz_Rm,J_Rz_Re)
        J_Re_R = H_x[0,0]
        J_Rz_R = J_Rz_Re @ J_Re_R

        J_Re_Rwn = H_v[0,0]
        J_Rz_Rwn = J_Rz_Re @ J_Re_Rwn
        
        # Z.p= Y.(+) R_wn
        p_z = Y.p_m - Y_e.p_m
        J_pz_pe = -np.identity(3)

        J_pe_p = H_x[1,1]
        J_pz_p = J_pz_pe @ J_pe_p

        J_pe_pwn = H_v[1,1]
        J_pz_pwn = J_pz_pe @ J_pe_pwn

        z = OptitrackNoise(R_z, p_z)

        J_z_x = np.array([[J_Rz_R, _ZERO,  _ZERO, _ZERO, _ZERO],
                          [ _ZERO, _ZERO, J_pz_p, _ZERO, _ZERO]])
        
        J_z_v = np.array([[J_Rz_Rwn,    _ZERO],
                          [   _ZERO, J_pz_pwn]])

        return z, J_z_x, J_z_v




        


# Observation system
