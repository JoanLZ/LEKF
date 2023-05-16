from manifpy import SO3
import numpy as np


class State:

    def __init__(self, R=SO3.Identity(), v=np.zeros(3), p=np.zeros(3), a_b=np.zeros(3), ω_b=np.zeros(3)):
        self.R = SO3(R.coeffs_copy()) # rotation
        self.v = v # velocity
        self.p = p # position
        self.a_b = a_b # linear acceleration bias
        self.ω_b = ω_b # angular velocity bias