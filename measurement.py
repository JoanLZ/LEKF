from __future__ import annotations
from manifpy import SO3, SO3Tangent, R3, R3Tangent
import numpy as np

class Command: # U = [am, ωm]

    # Constructor
    def __init__(self, coeffs: np.array) -> None:
        if coeffs.size != 6:
            raise ValueError(f'Size of coeffs should be 6, not {coeffs.size}.')
        self._coeffs = coeffs.copy()

    def Identity() -> Command:
        coeffs = np.zeros(6)
        return Command(coeffs)

    def Bundle(am: R3, ωm: R3) -> Command:
        coeffs = np.array([am.coeffs_copy(),
                           ωm.coeffs_copy()]).flatten()
        return Command(coeffs)
        
    # Get & Set
    def get_am(self) -> R3:
        return R3(self._coeffs[0:3].copy())
    
    def set_am(self, am: R3) -> None:
        self._coeffs[0:3] = am.coeffs_copy()

    def get_ωm(self) -> R3:
        return R3(self._coeffs[3:6].copy())
    
    def set_ωm(self, ωm: R3) -> None:
        self._coeffs[3:6] = ωm.coeffs_copy()

    # Operators

    def rplus(self, ΔU: CommandTangent) -> Command:
        am = self.get_am().rplus(ΔU.get_Δam())
        ωm = self.get_ωm().rplus(ΔU.get_Δωm())
        return Command.Bundle(am, ωm)

    def __add__(self, ΔU: CommandTangent) -> Command: # U+ΔU
        return self.rplus(ΔU)

class CommandTangent: # ΔU = [Δam, Δωm]

    # Constructor
    def __init__(self, coeffs: np.array) -> None:
        if coeffs.size != 6:
            raise ValueError(f'Size of coeffs should be 6, not {coeffs.size}.')
        self._coeffs = coeffs.copy()

    def Identity() -> CommandTangent:
        coeffs = np.zeros(6)
        return CommandTangent(coeffs)

    def Bundle(Δam: R3Tangent, Δωm: R3Tangent) -> CommandTangent:
        coeffs = np.array([Δam.coeffs_copy(),
                           Δωm.coeffs_copy()]).flatten()
        return CommandTangent(coeffs)

    # Get & Set
    def get_Δam(self) -> R3Tangent:
        return R3Tangent(self._coeffs[0:3].copy())
    
    def set_Δam(self, Δam: R3Tangent) -> None:
        self._coeffs[0:3] = Δam.coeffs_copy()

    def get_Δωm(self) -> R3Tangent:
        return R3Tangent(self._coeffs[3:6].copy())
    
    def set_Δωm(self, Δωm: R3Tangent) -> None:
        self._coeffs[3:6] = Δωm.coeffs_copy()

    def __rmatmul__(self, A: np.ndarray) -> np.array: # A@ΔU
        if A.shape[1] != 6:
            raise ValueError(f'Nb. of colums should be 6, not {A.shape[1]}.')
        return A@self._coeffs


class ImuMeasurement:

    def __init__(self, a_m=np.zeros(3), ω_m=np.zeros(3)):
        self.a_m = a_m  # linear acceleration measurement
        self.ω_m = ω_m  # angular velocity measurement


class ImuNoise:

    def __init__(self, a_wn=np.zeros(3), ω_wn=np.zeros(3), a_rw=np.zeros(3), ω_rw=np.zeros(3)):
        self.a_wn = a_wn  # linear acceleration white noise
        self.ω_wn = ω_wn  # angular velocity white noise
        self.a_rw = a_rw  # linear acceleration random walk
        self.ω_rw = ω_rw  # angular velocity random walk


class OptitrackMeasurement:

    def __init__(self, R_m=SO3.Identity(), p_m=np.zeros(3)):
        self.R_m = SO3(R_m.coeffs_copy())  # rotation measurement
        self.p_m = p_m  # position measurement
    

class OptitrackNoise:

    def __init__(self, R_wn=SO3Tangent.Zero(), p_wn=np.zeros(3)):
        self.R_wn = SO3Tangent(R_wn.coeffs_copy())  # rotation white noise
        self.p_wn = p_wn  # position white noise