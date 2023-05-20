from __future__ import annotations
from manifpy import R3, R3Tangent
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
        coeffs = []
        coeffs.extend(am.coeffs_copy())
        coeffs.extend(ωm.coeffs_copy())
        coeffs = np.array(coeffs).flatten()
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
        return Input.Bundle(am, ωm)

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
        coeffs = []
        coeffs.extend(Δam.coeffs_copy())
        coeffs.extend(Δωm.coeffs_copy())
        coeffs = np.array(coeffs).flatten()
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

