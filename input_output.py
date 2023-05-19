from __future__ import annotations
from manifpy import SO3, SO3Tangent, R3, R3Tangent
import numpy as np


class Input:  # U = [am, ωm]

    # Constructor
    def __init__(self, coeffs: np.array) -> None:
        if coeffs.size != 6:
            raise ValueError(f'Size of coeffs should be 6, not {coeffs.size}.')
        self._coeffs = coeffs.copy()

    def Identity() -> Input:
        coeffs = np.zeros(6)
        return Input(coeffs)

    def Bundle(am: R3, ωm: R3) -> Input:
        coeffs = np.array([am.coeffs_copy(),
                           ωm.coeffs_copy()]).flatten()
        return Input(coeffs)

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

    def rplus(self, ΔU: InputTangent) -> Input:
        am = self.get_am().rplus(ΔU.get_Δam())
        ωm = self.get_ωm().rplus(ΔU.get_Δωm())
        return Input.bundle(am, ωm)

    def __add__(self, ΔU: InputTangent) -> Input:  # U+ΔU
        return self.rplus(ΔU)


class InputTangent:  # ΔU = [Δam, Δωm]

    # Constructor
    def __init__(self, coeffs: np.array) -> None:
        if coeffs.size != 6:
            raise ValueError(f'Size of coeffs should be 6, not {coeffs.size}.')
        self._coeffs = coeffs.copy()

    def Identity() -> InputTangent:
        coeffs = np.zeros(6)
        return InputTangent(coeffs)

    def Bundle(Δam: R3Tangent, Δωm: R3Tangent) -> InputTangent:
        coeffs = np.array([Δam.coeffs_copy(),
                           Δωm.coeffs_copy()]).flatten()
        return InputTangent(coeffs)

    # Get & Set
    def get_Δam(self) -> R3Tangent:
        return R3Tangent(self._coeffs[0:3].copy())

    def set_Δam(self, Δam: R3Tangent) -> None:
        self._coeffs[0:3] = Δam.coeffs_copy()

    def get_Δωm(self) -> R3Tangent:
        return R3Tangent(self._coeffs[3:6].copy())

    def set_Δωm(self, Δωm: R3Tangent) -> None:
        self._coeffs[3:6] = Δωm.coeffs_copy()

    def __rmatmul__(self, A: np.ndarray) -> np.array:  # A@ΔU
        if A.shape[1] != 6:
            raise ValueError(f'Nb. of colums should be 6, not {A.shape[1]}.')
        return A@self._coeffs

class Output:  # Y = [Rm, pm]

    # Constructor
    def __init__(self, coeffs: np.array) -> None:
        if coeffs.size != 7:
            raise ValueError(f'Size of coeffs should be 7, not {coeffs.size}.')
        self._coeffs = coeffs.copy()

    def Identity() -> Output:
        coeffs = np.zeros(7)
        coeffs[3] = 1
        return Output(coeffs)

    def Bundle(Rm: SO3, pm: R3) -> Output:
        coeffs = np.array([Rm.coeffs_copy(),
                           pm.coeffs_copy()]).flatten()
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
        return Output.bundle(Rm, pm)

    def __add__(self, ΔY: OutputTangent) -> Output:  # Y+ΔY
        return self.rplus(ΔY)


class OutputTangent:  # ΔY = [ΔRm, Δpm]

    # Constructor
    def __init__(self, coeffs: np.array) -> None:
        if coeffs.size != 6:
            raise ValueError(f'Size of coeffs should be 6, not {coeffs.size}.')
        self._coeffs = coeffs.copy()

    def Identity() -> OutputTangent:
        coeffs = np.zeros(6)
        return OutputTangent(coeffs)

    def Bundle(ΔRm: SO3Tangent, Δpm: R3Tangent) -> OutputTangent:
        coeffs = np.array([ΔRm.coeffs_copy(),
                           Δpm.coeffs_copy()]).flatten()
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
