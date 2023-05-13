import numpy as np
from manifpy import SO3, Rn, SO3Tangent
from matplotlib import pyplot as plt

import lekf


class Simulator:

    def __init__(self) -> None:
        x0
        P0
        # init LEKF(x0,P0, + all the covariance matrix)
    def step(self, dt):
        # Compute the command u
        # Compute the noise r
        # Update your system x = f(x, u, r)
        # Do prediction with the LEKF.predict(u)
        if time_to_optitrack:
            # Compute the noise v
            # Compute the measurement y = h(x, v)
            # Do the correction with the LEKF.correct(y)
            pass


if __name__ == "__main__":
    sim = Simulator()
    while True:
        sim.step()
        # make som plot
        # sleep (dt)