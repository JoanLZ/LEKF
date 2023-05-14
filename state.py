from manifpy import R3, SO3


class State:

    def __init__(self, R, v, p, a_b, ω_b):
        self.R = SO3(R.coeffs_copy()) # rotation
        self.v = v # velocity
        self.p = p # position
        self.a_b = a_b # linear acceleration bias
        self.ω_b = ω_b # angular velocity bias