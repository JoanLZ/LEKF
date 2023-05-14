import numpy as np
from manifpy import SO3, SO3Tangent


class ImuMeasurement:

    def __init__(self, a_m, ω_m):
        self.a_m = a_m  # linear acceleration measurement
        self.ω_m = ω_m  # angular velocity measurement


class ImuNoise:

    def __init__(self, a_wn, ω_wn, a_rw, ω_rw):
        self.a_wn = a_wn  # linear acceleration white noise
        self.ω_wn = ω_wn  # angular velocity white noise
        self.a_rw = a_rw  # linear acceleration random walk
        self.ω_rw = ω_rw  # angular velocity random walk


class OptitrackMeasurement:

    def __init__(self, R_m, p_m):
        self.R_m = SO3(R_m.coeffs_copy())  # rotation measurement
        self.p_m = p_m  # position measurement
    

class OptitrackNoise:

    def __init__(self, R_wn, p_wn):
        self.R_wn = SO3Tangent(R_wn.copy())  # rotation white noise
        self.p_wn = p_wn  # position white noise