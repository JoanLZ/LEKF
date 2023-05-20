#!/usr/bin/python3

import numpy as np
from state import State
from manifpy import SO3, R3

# RATE & TIME
IMU_RATE = 1000  # [Hz]
OPTITRACK_RATE = 100  # [Hz]
TIME = 150  # [s]

# NOISE
## IMU
IMU_NOISE = True
σ_sim_aw = 6.3e-5 # m.s-2
σ_sim_ωw = 8.7e-5 # rad.s-1
σ_sim_ar = 4.0e-4 # m.s-2
σ_sim_ωr = 3.9e-5 # rad.s-1
## OPTITRACK
OPTITRACK_NOISE = True
σ_sim_Rw = (1.8e-2)/3 # 95% rand < 1º
σ_sim_pw = (1e-3)/3   # 95% rand < 1mm

# INIT
## STATE
v0 = np.array([0,1,0])
p0 = np.zeros(3)
X = State.Bundle(SO3.Identity(),
                 R3(v0),
                 R3(p0),
                 R3.Identity(),
                 R3.Identity())
## SIGMAS
σ_P_R = 1e-4
σ_P_v = 1e-4
σ_P_p = 1e-4
σ_P_ab = 0 # No random walk
σ_P_ωb = 0 # No random walk
σ_Q_am = 0 # deterministic
σ_Q_ωm = 0 # deterministic
σ_W_aw = 1e-4
σ_W_ωw = 1e-4
σ_W_ar = 0 # No random walk
σ_W_ωr = 0 # No random walk
σ_V_Rw = 1e-4
σ_V_pw = 1e-5
## ESTIMATE
X0 = State.Bundle(SO3.Random(),
                  R3.Identity(),
                  R3(p0+np.random.normal(0,σ_P_p,3)),
                  R3.Identity(),
                  R3.Identity())

# LEKF
DO_PREDICTION = True
DO_CORRECTION = True


# MISCAELLOUS
G = np.array([0, 0, -9.81])
