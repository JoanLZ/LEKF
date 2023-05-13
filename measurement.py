from manifpy import R3, SO3, R3Tangent, SO3Tangent


class ImuMeasurement:

    def __init__(self, a_m, ω_m):
        self.a_m = R3(a_m)  # linear acceleration measurement
        self.ω_m = R3(ω_m)  # angular velocity measurement


class ImuNoise:

    def __init__(self, a_wn, ω_wn, a_rw, ω_rw):
        self.a_wn = R3(a_wn)  # linear acceleration white noise
        self.ω_wn = R3(ω_wn)  # angular velocity white noise
        self.a_rw = R3(a_rw)  # linear acceleration random walk
        self.ω_rw = R3(ω_rw)  # angular velocity random walk


class OptitrackMeasurement:

    def __init__(self, R_m, p_m):
        self.R_m = SO3(R_m)  # rotation measurement
        self.p_m = R3(p_m)  # position measurement


class OptitrackNoise:

    def __init__(self, R_wn, p_wn):
        self.R_wn = SO3(R_wn)  # rotation white noise
        self.p_wn = R3(p_wn)  # position white noise