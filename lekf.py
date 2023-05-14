
import numpy as np
from manifpy import *

from measurement import *
from state import *


class LEKF:
    # Constructor
    def __init__(self) -> None:
        pass

    # User function
    def predict(self,U,dt):
        
        x = State (np.array([0.5,0.5,0.5,0.5]),
                   np.array([0,0,0]),
                   np.array([1/3,1/3,1/3]),
                   np.array([0,0,0]),
                   np.array([0,0,0]))
        
        P = np.identity(x.R.DoF + x.v.size + x.p.size + x.a_b.size + x.ω_b.size)

        W_predict = ImuNoise(np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3))    

        x_plus, F_x, F_u, F_w = self.f(x,U,W_predict)

        P_plus = F_x @ P @ F_x.transpose() + F_u @ Q @ F_u.transpose() + F_w @ W @ F_w.transpose()

        return x_plus, P_plus
   
    def correct(self, Y):
        pass

    # Method 

    def a(X,U):
        g = np.array([0,0,9.81])
        # compute the true a 
        ## remove bias from measurement
        ### equation
        Δa = U.a_m-X.a_b
        ### jacobian
        J_Δa_am = np.identity(len(Δa)) # Jacobian of the differnce wrt a_m
        J_Δa_ab = -np.identity(len(Δa)) # Jacobian of the differnce wrt a_b
        ## compute true value
        ### equation
        J_RΔa_R = np.ndarray([X.R.DoF,X.R.DoF]) # Defining the Jacobians for manif to compute. 
        J_RΔa_Δa = np.ndarray([X.R.DoF,Δa.size]) # Defining the Jacobians for manif to compute.
        a = X.R.act(Δa, J_RΔa_R, J_RΔa_Δa) + g # a = R * ( Δa ) + g. The result is a 3d vector.
        ### jacobian
        J_a_R = J_RΔa_R # From the paper: J_R*v_R = ... = -R[v]x. In this case J_a_R = -R[ da ]x = -R*[a_m -a_b]x
        J_a_am = J_RΔa_Δa @ J_Δa_am # J_a_am = J_R(am-ab)+g_(am-ab) dot J_(am-ab)_am
        J_a_ab = J_RΔa_Δa @ J_Δa_ab # J_a_ab = J_R(am-ab)+g_(am-ab) dot J_(am-ab)_ab
        return a, J_a_R, J_a_am, J_a_ab 

    def omega(X,U):
        
        ## compute true value
        ### equation
        ω = ω_m - X.wb # w = w_m - w_b defining wmeasured and wbias as a 3D vector. 
        ### jacobian
        J_ω_ωm = np.identity((3,3))
        J_ω_ωb = -np.identity((3,3))

        return ω, J_ω_ωm, J_ω_ωb 

    # Dynamics of system

    def f(self, X, U, W, dt):
        X_o = State(X.R, X,v, X.p, X.a_b, X.ω_b)
        ## compute real values of U
        a, J_a_R, J_a_am, J_a_ab = self.a(X,U) # real a
        ω, J_ω_ωm, J_ω_ωb = self.omega(X,U) # real ω
        ## compute value
        ### equation
        J_RExp_R = np.ndarray ([SO3.DoF(),X.R.DoF()]) # Defining the Jacobians for manif to compute.
        J_RExp_ωdt = np.ndarray ([SO3.DoF(),ω.size]) # Defining the Jacobians for manif to compute.
        X_o.R = X.R.rplus(SO3Tangent(ω*dt),J_RExp_R,J_RExp_ωdt) # R (+) w*dt = RExp(wdt)
        ### jacobian  
        J_ωdt_ω = dt.np.identity(3) # We computed the jacobian of the plus operation wrt to wdt, not w. So lets compute jacobian of wdt wrt w and then apply chain rule.
        J_RExp_ω = J_RExp_ωdt @ J_ωdt_ω # Chain rule
        ### equation
        X_o.v = X.v + a*dt # v =  v + a*dt
        ### jacobian  
        J_v_v = np.identity(3)
        J_v_a = dt*np.identity(3)
        ### equation
        X_o.p = X.p + X.v*dt + 0.5*a*dt**2  # p =  p + v*dt + 0.5*a*dt²
        ### jacobian  
        J_p_p = np.identity(3)
        J_p_v = dt*np.identity(3)
        J_p_a = 0.5*np.identity(3)*dt**2
        ### equation
        X_o.a_b = X.a_b + W.a_rw # a_b =  a_b + a_r 
        ### jacobian  
        J_ab_ab = np.identity(3)
        J_ab_ar = np.identity(3)
        ### equation
        X_o.ω_b = X.ω_b + W.ω_rw # ω_b =  ω_b + ω_r 
        ### jacobian  
        J_ωb_ωb = np.identity(3)
        J_ωb_ωr = np.identity(3)

        ### Assemble big jacobians: J_f_x, J_f_u, J_f_r

        Id = np.identity(3)

        #Jacobians of R wrt state vars
        J_R_wb = J_RExp_ω @ J_ω_ωb

        #Jacobians of v wrt state vars
        J_v_R = J_v_a @ J_a_R  #J_v+adt_adt @ J_adt_a @ J_R(am-ab)+g_R = J_v+adt_a @ J_R(am-ab)+g_R 
        J_v_ab = J_v_a @ J_a_ab

        #Jacobians of p wrt state vars
        J_p_R = J_p_a @ J_p_R
        J_p_ab = J_p_a @ J_a_ab

        #J_f_x

        J_f_x = np.array ([J_RExp_R,  0,    0,      0,  J_R_wb],
                          [J_v_R,     Id,   0,    J_v_ab,    0],
                          [J_p_R,   J_p_v, Id,    J_p_ab,    0],
                          [0,         0,    0,      Id,      0],
                          [0,         0,    0,       0,     Id]) 
        
        
        #Jacobians of v wrt IMU measurments
        J_v_am = J_v_a @ J_a_am

        #Jacobians of p wrt IMU measurments
        J_p_am = J_p_a @ J_a_am

        #J_f_u

        J_f_u = np.array ([0     , 0],
                          [J_v_am, 0],
                          [J_p_am, 0],
                          [0     , 0],
                          [0     , 0])
        #J_f_w

        J_f_w = np.array ([0    , 0],
                          [0    , 0],
                          [0    , 0],
                          [Id   , 0],
                          [Id   , Id])
        
        return X_o, J_f_x, J_f_u, J_f_w 
    
    def h(self, )

        
        



# Observation system