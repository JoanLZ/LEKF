
import numpy as np
from manifpy import *

from input import Input
from pertubation import Pertubation
from output import Output, OutputTangent
from state import State, StateTangent
from manifpy import SO3Tangent, R3Tangent, R3

class LEKF:

    # Constructor
    def __init__(self, X0: State, P0, Q, W, V) -> None:
        self.X = X0  # [State]
        self.P = P0  # [Covariance]
        self._Q = Q  # [Covariance]
        self._W = W  # [Covariance]
        self._V = V  # [Covariance]

    # User functions
    def predict(self, U: Input, dt: float) -> None:
        '''Make a prediction of the state knowing the command applied on the system.
        1. Compute the estimated state + the jacobians of the function.
        2. Update the covariance matrix of the estimated state.
        '''

        # f() is define as a function with noise. Set the noise at 0.
        # TODO[Hugo]: f() is only use for the prediction (where it is considere as 0).
        #             Remove the noise W.
        W = Pertubation.Identity()

        X_out, F_x, F_u, F_w = self.f(self.X, U, W, dt)
        P_out = F_x @ self.P  @ F_x.transpose() \
              + F_u @ self._Q @ F_u.transpose() \
              + F_w @ self._W @ F_w.transpose()

        self.X = X_out
        self.P = (P_out+P_out.T)/2 # Way to ensure symmetry of the matrix

    def correct(self, Y: Output) -> None:
        '''Make a correction of the state knowing the current measurements.
        1. Compute the innovation + the jacobians of the innovation.
        2. Compute the covariance matrix of the innovation.
        3. Compute the Kalman gain.
        4. Update the estimated dtate and its covariance matrix.
        '''

        z, J_z_x, J_z_v = self.z(Y)

        Z = J_z_x @ self.P @ J_z_x.transpose() + J_z_v @ self._V @ J_z_v.transpose()

        K = - self.P @ J_z_x.transpose() @ np.linalg.inv(Z)

        X_out = self.X+StateTangent(z.__rmatmul__(K))
        P_out = self.P - K @ Z @ K.transpose()

        self.X = X_out
        self.P = (P_out+P_out.T)/2 # Way to ensure symmetry of the matrix

    # Acceleration
    def a(self, X: State, U: Input, W: Pertubation) -> any:

        _G = np.array([0, 0, -9.81])

        R  = X.get_R()
        ab = X.get_ab().coeffs_copy()
        am = U.get_am().coeffs_copy()
        aw = W.get_aw().coeffs_copy()

        # Equation & jacobians of Δa 
        Δa = am-ab-aw
        J_Δa_am =  np.identity(len(Δa))  # Jacobian of Δa wrt am
        J_Δa_ab = -np.identity(len(Δa))  # Jacobian of Δa wrt ab
        J_Δa_aw = -np.identity(len(Δa))  # Jacobian of Δa wrt aw

        # Equation & jacobians of a 
        J_a_R  = np.ndarray([len(Δa),   R.DoF]) # Jacobian of a wrt R
        J_a_Δa = np.ndarray([len(Δa), len(Δa)]) # Jacobian of a wrt Δa
        a = R.act(Δa, J_a_R, J_a_Δa) + _G

        J_a_am = J_a_Δa @ J_Δa_am # Jacobian of a wrt am
        J_a_ab = J_a_Δa @ J_Δa_ab # Jacobian of a wrt ab
        J_a_aw = J_a_Δa @ J_Δa_aw # Jacobian of a wrt aw

        return a, J_a_R, J_a_am, J_a_ab, J_a_aw
    
    # Omega
    def ω(self, X: State, U: Input, W: Pertubation) -> any:

        ωm = U.get_ωm().coeffs_copy()
        ωb = X.get_ωb().coeffs_copy()
        ωw = W.get_ωw().coeffs_copy()

        # Equation & jacobians of ω 
        ω = ωm - ωb - ωw
        J_ω_ωm =  np.identity(len(ω)) # Jacobian of ω wrt ωm
        J_ω_ωb = -np.identity(len(ω)) # Jacobian of ω wrt ωb
        J_ω_ωw = -np.identity(len(ω)) # Jacobian of ω wrt ωw

        return ω, J_ω_ωm, J_ω_ωb, J_ω_ωw

    # Dynamics of system
    def f(self, X: State, U: Input, W: Pertubation, dt: float) -> any:

        R  = X.get_R()
        v  = X.get_v()
        p  = X.get_p()
        ab = X.get_ab()
        ωb = X.get_ωb()
        ar = W.get_ar()
        ωr = W.get_ωr()

        # get true acceleration and linear velocity with their jacobians
        a, J_a_R, J_a_am, J_a_ab, J_a_aw = self.a(X, U, W)
        ω, J_ω_ωm, J_ω_ωb, J_ω_ωw        = self.ω(X, U, W)

        # Equation & jacobians of Ro
        J_Ro_R   = np.ndarray([SO3Tangent.DoF,  R.DoF]) # Jacobian of Ro wrt R
        J_Ro_ωdt = np.ndarray([SO3Tangent.DoF, ω.size]) # Jacobian of Ro wrt ωdt
        Ro = R.rplus(SO3Tangent(ω*dt), J_Ro_R,J_Ro_ωdt)

        J_ωdt_ω = dt*np.identity(len(ω)) # Jacobian of ωdt wrt ω
        J_Ro_ω  = J_Ro_ωdt @ J_ωdt_ω     # Jacobian of Ro wrt ω
        J_Ro_ωm =   J_Ro_ω @ J_ω_ωm      # Jacobian of Ro wrt ωm
        J_Ro_ωb =   J_Ro_ω @ J_ω_ωb      # Jacobian of Ro wrt ωb
        J_Ro_ωw =   J_Ro_ω @ J_ω_ωw      # Jacobian of Ro wrt ωw

        # Equation & jacobians of vo
        J_vo_v   = np.ndarray([R3Tangent.DoF,  R.DoF]) # Jacobian of vo wrt v
        J_vo_adt = np.ndarray([R3Tangent.DoF, len(a)]) # Jacobian of vo wrt adt
        vo = v.rplus(R3Tangent(a*dt), J_vo_v, J_vo_adt)

        J_adt_a = dt*np.identity(len(a)) # Jacobian of adt wrt a
        J_vo_a  = J_vo_adt @ J_adt_a     # Jacobian of vo wrt a
        J_vo_R  = J_vo_a @ J_a_R         # Jacobian of vo wrt R
        J_vo_am = J_vo_a @ J_a_am        # Jacobian of vo wrt am
        J_vo_ab = J_vo_a @ J_a_ab        # Jacobian of vo wrt ab
        J_vo_aw = J_vo_a @ J_a_aw        # Jacobian of vo wrt aw

        # Equation & jacobians of po
        J_po_p = np.identity(3)             # Jacobian of po wrt p
        J_po_v = dt*np.identity(3)          # Jacobian of po wrt v
        J_po_a = 0.5*np.identity(3)*(dt**2) # Jacobian of po wrt a
        po = R3(p.coeffs() + v.coeffs()*dt + 0.5*a*(dt**2))

        J_po_R  = J_po_a @ J_a_R  # Jacobian of po wrt R
        J_po_am = J_po_a @ J_a_am # Jacobian of po wrt am
        J_po_ab = J_po_a @ J_a_ab # Jacobian of po wrt ab
        J_po_aw = J_po_a @ J_a_aw # Jacobian of po wrt aw

        # Equation & jacobians of abo
        J_abo_ab = np.ndarray([R3Tangent.DoF, len(a)]) # Jacobian of abo wrt ab
        J_abo_ar = np.ndarray([R3Tangent.DoF, len(a)]) # Jacobian of abo wrt ar
        abo = ab.rplus(ar, J_abo_ab, J_abo_ar)

        # Equation & jacobians of ωbo
        J_ωbo_ωb = np.ndarray([R3Tangent.DoF, len(ω)]) # Jacobian of ωbo wrt ωb
        J_ωbo_ωr = np.ndarray([R3Tangent.DoF, len(ω)]) # Jacobian of ωbo wrt ωr
        ωbo = ωb.rplus(ωr, J_ωbo_ωb, J_ωbo_ωr)

        # J_f_x = np.array([[J_Ro_R,  _ZERO,  _ZERO,    _ZERO,  J_Ro_ωb],
        #                   [J_vo_R, J_vo_v,  _ZERO,  J_vo_ab,    _ZERO],
        #                   [J_po_R, J_po_v, J_po_p,  J_po_ab,    _ZERO],
        #                   [ _ZERO,  _ZERO,  _ZERO, J_abo_ab,    _ZERO],
        #                   [ _ZERO,  _ZERO,  _ZERO,    _ZERO, J_ωbo_ωb]])

        J_f_x = np.zeros([15, 15])
        J_f_x[0:3, 0:3]     = J_Ro_R
        J_f_x[0:3, 12:15]   = J_Ro_ωb
        J_f_x[3:6, 0:3]     = J_vo_R
        J_f_x[3:6, 3:6]     = J_vo_v
        J_f_x[3:6, 9:12]    = J_vo_ab
        J_f_x[6:9, 0:3]     = J_po_R
        J_f_x[6:9, 3:6]     = J_po_v
        J_f_x[6:9, 6:9]     = J_po_p
        J_f_x[6:9, 9:12]    = J_po_ab
        J_f_x[9:12, 9:12]   = J_abo_ab
        J_f_x[12:15, 12:15] = J_ωbo_ωb

        # J_f_u = np.array([[  _ZERO, J_Ro_ωm],
        #                   [J_vo_am,   _ZERO],
        #                   [J_po_am,   _ZERO],
        #                   [  _ZERO,   _ZERO],
        #                   [  _ZERO,   _ZERO]])

        J_f_u = np.zeros([15, 6])
        J_f_u[0:3, 3:6] = J_Ro_ωm
        J_f_u[3:6, 0:3] = J_vo_am
        J_f_u[6:9, 0:3] = J_po_am

        # J_f_w = [[   _ZERO, J_Ro_ωwn,     _ZERO,     _ZERO],
        #          [J_vo_awn,    _ZERO,     _ZERO,     _ZERO],
        #          [J_po_awn,    _ZERO,     _ZERO,     _ZERO],
        #          [   _ZERO,    _ZERO, J_abo_arw,     _ZERO],
        #          [   _ZERO,    _ZERO,     _ZERO, J_ωbo_ωrw]]

        J_f_w = np.zeros([15, 12])
        J_f_w[0:3, 3:6]    = J_Ro_ωw
        J_f_w[3:6, 0:3]    = J_vo_aw
        J_f_w[6:9, 0:3]    = J_po_aw
        J_f_w[9:12, 6:9]   = J_abo_ar
        J_f_w[12:15, 9:12] = J_ωbo_ωr

        Xo = State.Bundle(Ro,vo, po, abo, ωbo)

        return Xo, J_f_x, J_f_u, J_f_w

    # Expectation
    def h(self, X: State, V: OutputTangent) -> any:

        R  = X.get_R()
        p  = X.get_p()
        Rw = V.get_ΔRm()
        pw = V.get_Δpm()

        # Equation & jacobians of Re
        J_Re_R  = np.ndarray([SO3Tangent.DoF, SO3Tangent.DoF]) # Jacobian of Re wrt R
        J_Re_Rw = np.ndarray([SO3Tangent.DoF, SO3Tangent.DoF]) # Jacobian of Re wrt Rw
        Re = R.rplus(Rw, J_Re_R, J_Re_Rw)

        # Equation & jacobians of pe
        J_pe_p  = np.ndarray([R3Tangent.DoF, R3Tangent.DoF]) # Jacobian of pe wrt p
        J_pe_pw = np.ndarray([R3Tangent.DoF, R3Tangent.DoF]) # Jacobian of pe wrt pw
        pe = p.rplus(pw, J_pe_p, J_pe_pw)

        # J_h_x = np.array([[J_Re_R, _ZERO,  _ZERO, _ZERO, _ZERO],
        #                   [ _ZERO, _ZERO, J_pe_p, _ZERO, _ZERO]])

        J_h_x = np.zeros([6, 15])
        J_h_x[0:3, 0:3] = J_Re_R
        J_h_x[3:6, 6:9] = J_pe_p

        # J_h_v = np.array([[J_Re_Rw,   _ZERO],
        #                   [  _ZERO, J_pe_pw]])

        J_h_v = np.zeros([6, 6])
        J_h_v[0:3, 0:3] = J_Re_Rw
        J_h_v[3:6, 3:6] = J_pe_pw

        Ye = Output.Bundle(Re, pe)

        return Ye, J_h_x, J_h_v

    # Correction
    def z(self, Y: Output) -> any:

        Rm = Y.get_Rm()
        pm = Y.get_pm()

        V = OutputTangent.Identity()
        e, H_x, H_v = self.h(self.X, V)

        Re = e.get_Rm()
        pe = e.get_pm()

        # Equation & jacobians of Rz
        J_Rz_Rm = np.ndarray([SO3Tangent.DoF, SO3Tangent.DoF]) # Jacobian of Rz wrt Rm
        J_Rz_Re = np.ndarray([SO3Tangent.DoF, SO3Tangent.DoF]) # Jacobian of Rz wrt Re
        Rz = Rm.rminus(Re, J_Rz_Rm, J_Rz_Re)

        J_Re_R  = H_x[0:3, 0:3]     # Jacobian of Re wrt R
        J_Rz_R  = J_Rz_Re @ J_Re_R  # Jacobian of Rz wrt R
        J_Re_Rw = H_v[0:3, 0:3]     # Jacobian of Re wrt Rw
        J_Rz_Rw = J_Rz_Re @ J_Re_Rw # Jacobian of Rz wrt Rw

        # Equation & jacobians of pz
        J_pz_pm = np.ndarray([R3Tangent.DoF, R3Tangent.DoF]) # Jacobian of pz wrt pm
        J_pz_pe = np.ndarray([R3Tangent.DoF, R3Tangent.DoF]) # Jacobian of pz wrt pe
        pz = pm.rminus(pe, J_pz_pm, J_pz_pe)

        J_pe_p  = H_x[3:6, 6:9]     # Jacobian of pe wrt p
        J_pz_p  = J_pz_pe @ J_pe_p  # Jacobian of pz wrt p
        J_pe_pw = H_v[3:6, 3:6]     # Jacobian of pe wrt pw
        J_pz_pw = J_pz_pe @ J_pe_pw # Jacobian of pz wrt pw

        # J_z_x = np.array([[J_Rz_R, _ZERO,  _ZERO, _ZERO, _ZERO],
        #                   [ _ZERO, _ZERO, J_pz_p, _ZERO, _ZERO]])

        J_z_x = np.zeros([6, 15])
        J_z_x[0:3, 0:3] = J_Rz_R
        J_z_x[3:6, 6:9] = J_pz_p

        # J_z_v = np.array([[J_Rz_Rwn,    _ZERO],
        #                   [   _ZERO, J_pz_pwn]])

        J_z_v = np.zeros([6, 6])
        J_z_v[0:3, 0:3] = J_Rz_Rw
        J_z_v[3:6, 3:6] = J_pz_pw

        z = OutputTangent.Bundle(Rz, pz)

        return z, J_z_x, J_z_v