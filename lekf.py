
import numpy as np
from manifpy import *

from measurement import *
from state import *


class LEKF:
    # Constructor
    def __init__(self) -> None:
        pass

    # User function
    def predict(self, U):
        pass
    
    def correct(self, Y):
        pass

    # Dynamics of system

    def f(self, X, U, R):
        a, J_a_R, J_a_am, J_a_ab = a(X, U)


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
        J_RΔa_R = np.ndarray
        J_RΔa_Δa = np.ndarray
        a = X.R.act(Δa, J_RΔa_R, J_RΔa_Δa) + g # a = R * ( Δa ) + g. The result is a 3d vector.
        ### jacobian
        J_a_R = J_RΔa_R # From the paper: J_R*v_R = ... = -R[v]x. In this case J_a_R = -R[ da ]x = -R*[a_m -a_b]x
        J_a_am = J_RΔa_Δa*J_Δa_am # J_a_am = J_R(am-ab)+g_(am-ab) dot J_(am-ab)_am
        J_a_ab = J_RΔa_Δa*J_Δa_ab # J_a_ab = J_R(am-ab)+g_(am-ab) dot J_(am-ab)_ab
        return a, J_a_R, J_a_am, J_a_ab 

def True_omega(X,w_m):
    
    w = w_m - X.wb # w = w_m - w_b defining wmeasured and wbias as R3, if we use -, we get a SO3Tangent().
    J_w_wm = np.identity((3,3))
    J_w_wb = -np.identity((3,3))

    # Observation system