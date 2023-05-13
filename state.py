from manifpy import R3, SO3


class State:

    def __init__(self, R, v, p, a_b, ω_b):
        self.R = SO3(R) # rotation
        self.v = R3(v) # velocity
        self.p = R3(p) # position
        self.a_b = R3(a_b) # linear acceleration bias
        self.ω_b = R3(ω_b) # angular velocity bias