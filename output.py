from __future__ import annotations
from manifpy import SO3, SO3Tangent, R3, R3Tangent
import numpy as np

class Output:  # Y = [Rm, pm]

    # Constructor
    def __init__(self, coeffs: np.array) -> None:
        if coeffs.size != 7:
            raise ValueError(f'Size of coeffs should be 7, not {coeffs.size}.')
        self._coeffs = coeffs.flatten().copy()

    def Identity() -> Output:
        coeffs = np.zeros(7)
        coeffs[3] = 1
        return Output(coeffs)

    def Bundle(Rm: SO3, pm: R3) -> Output:
        coeffs = []
        coeffs.extend(Rm.coeffs_copy())
        coeffs.extend(pm.coeffs_copy())
        coeffs = np.array(coeffs).flatten()
        return Output(coeffs)

    # Get & Set
    def get_Rm(self) -> SO3:
        return SO3(self._coeffs[0:4].copy())

    def set_Rm(self, Rm: SO3) -> None:
        self._coeffs[0:4] = Rm.coeffs_copy()

    def get_pm(self) -> R3:
        return R3(self._coeffs[4:7].copy())

    def set_pm(self, pm: R3) -> None:
        self._coeffs[4:7] = pm.coeffs_copy()

    # Operators

    def rplus(self, ΔY: OutputTangent) -> Output:
        Rm = self.get_Rm().rplus(ΔY.get_ΔRm())
        pm = self.get_pm().rplus(ΔY.get_Δpm())
        return Output.Bundle(Rm, pm)

    def __add__(self, ΔY: OutputTangent) -> Output:  # Y+ΔY
        return self.rplus(ΔY)
    
    def rminus(self, Yb: Output) -> OutputTangent:
        ΔRm = self.get_Rm().rminus(Yb.get_Rm())
        Δpm = self.get_pm().rplus(Yb.get_pm())
        return OutputTangent.Bundle(ΔRm, Δpm)
    
    def __sub__(self, Yb: Output) -> OutputTangent: # Y-Yb
        return self.rplus(Yb)


class OutputTangent:  # ΔY = [ΔRm, Δpm]

    # Constructor
    def __init__(self, coeffs: np.array) -> None:
        if coeffs.size != 6:
            raise ValueError(f'Size of coeffs should be 6, not {coeffs.size}.')
        self._coeffs = coeffs.flatten().copy()

    def Identity() -> OutputTangent:
        coeffs = np.zeros(6)
        return OutputTangent(coeffs)

    def Bundle(ΔRm: SO3Tangent, Δpm: R3Tangent) -> OutputTangent:
        coeffs = []
        coeffs.extend(ΔRm.coeffs_copy())
        coeffs.extend(Δpm.coeffs_copy())
        coeffs = np.array(coeffs).flatten()
        return OutputTangent(coeffs)

    # Get & Set
    def get_ΔRm(self) -> SO3Tangent:
        return SO3Tangent(self._coeffs[0:3].copy())

    def set_ΔRm(self, ΔRm: SO3Tangent) -> None:
        self._coeffs[0:3] = ΔRm.coeffs_copy()

    def get_Δpm(self) -> R3Tangent:
        return R3Tangent(self._coeffs[3:6].copy())

    def set_Δpm(self, Δpm: R3Tangent) -> None:
        self._coeffs[3:6] = Δpm.coeffs_copy()

    def __rmatmul__(self, A: np.ndarray) -> np.array:  # A@ΔY
        if A.shape[1] != 6:
            raise ValueError(f'Nb. of colums should be 6, not {A.shape[1]}.')
        return A@self._coeffs