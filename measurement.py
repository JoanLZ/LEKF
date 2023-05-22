import numpy as np
from manifpy import SO3, SO3Tangent


class ImuMeasurement:

    def __init__(self, a_m=np.zeros(3), ω_m=np.zeros(3)):
        self.a_m = a_m  # linear acceleration measurement
        self.ω_m = ω_m  # angular velocity measurement
        # self.a_w = a_m  # linear acceleration measurement
        # self.ω_w = ω_m  # angular velocity measurement

class ImuBiasRandomWalk:

    def __init__(self, a_rw=np.zeros(3), ω_rw=np.zeros(3)):#, a_rw=np.zeros(3), ω_rw=np.zeros(3)):
        self.a_rw = a_rw  # linear acceleration random walk
        self.ω_rw = ω_rw  # angular velocity random walk

class ImuNoise:

    def __init__(self, a_wn=np.zeros(3), ω_wn=np.zeros(3)):#, a_wn=np.zeros(3), ω_wn=np.zeros(3)):
        self.a_wn = a_wn  # linear acceleration random walk
        self.ω_wn = ω_wn  # angular velocity random walk
        # self.a_rw = a_rw  # linear acceleration random walk
        # self.ω_rw = ω_rw  # angular velocity random walk
        
class OptitrackMeasurement:

    def __init__(self, R=SO3.Identity(), p=np.zeros(3)):
        self.R = SO3(R.coeffs_copy())  # rotation measurement
        self.p = p  # position measurement
    
class OptitrackNoise:

    def __init__(self, R=SO3Tangent.Zero(), p=np.zeros(3)):
        self.R = SO3Tangent(R.coeffs_copy())  # rotation white noise
        self.p = p  # position white noise