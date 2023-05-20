from __future__ import annotations
from manifpy import SO3, SO3Tangent, R3, R3Tangent
import numpy as np


class State:  # X = [R, v, p, ab, ωb]

    # Constructor
    def __init__(self, coeffs: np.array) -> None:
        if coeffs.size != 16:
            raise ValueError(
                f'Size of coeffs should be 16, not {coeffs.size}.')
        self._coeffs = coeffs.flatten().copy()

    def Identity() -> State:
        coeffs = np.zeros(16)
        coeffs[3] = 1
        return State(coeffs)

    def Bundle(R: SO3, v: R3, p: R3, ab: R3, ωb: R3) -> State:
        coeffs = []
        coeffs.extend(R.coeffs_copy())
        coeffs.extend(v.coeffs_copy())
        coeffs.extend(p.coeffs_copy())
        coeffs.extend(ab.coeffs_copy())
        coeffs.extend(ωb.coeffs_copy())
        coeffs = np.array(coeffs).flatten()
        return State(coeffs)

    # Get & Set
    def get_R(self) -> SO3:
        return SO3(self._coeffs[0:4].copy())

    def set_R(self, R: SO3) -> None:
        self._coeffs[0:4] = R.coeffs_copy()

    def get_v(self) -> R3:
        return R3(self._coeffs[4:7].copy())

    def set_v(self, v: R3) -> None:
        self._coeffs[4:7] = v.coeffs_copy()

    def get_p(self) -> R3:
        return R3(self._coeffs[7:10].copy())

    def set_p(self, p: R3) -> None:
        self._coeffs[7:10] = p.coeffs_copy()

    def get_ab(self) -> R3:
        return R3(self._coeffs[10:13].copy())

    def set_ab(self, ab: R3) -> None:
        self._coeffs[10:13] = ab.coeffs_copy()

    def get_ωb(self) -> R3:
        return R3(self._coeffs[13:16].copy())

    def set_ωb(self, ωb: R3) -> None:
        self._coeffs[13:16] = ωb.coeffs_copy()

    # Operators

    def rplus(self, ΔX: StateTangent) -> State:
        R = self.get_R().rplus(ΔX.get_ΔR())
        v = self.get_v().rplus(ΔX.get_Δv())
        p = self.get_p().rplus(ΔX.get_Δp())
        ab = self.get_ab().rplus(ΔX.get_Δab())
        ωb = self.get_ωb().rplus(ΔX.get_Δωb())
        return State.Bundle(R, v, p, ab, ωb)

    def __add__(self, ΔX: StateTangent) -> State:  # X+ΔX
        return self.rplus(ΔX)


class StateTangent:  # ΔX = [ΔR, Δv, Δp, Δab, Δωb]

    # Constructor
    def __init__(self, coeffs: np.array) -> None:
        if coeffs.size != 15:
            raise ValueError(
                f'Size of coeffs should be 15, not {coeffs.size}.')
        self._coeffs = coeffs.flatten().copy()

    def Identity() -> StateTangent:
        coeffs = np.zeros(15)
        return StateTangent(coeffs)

    def Bundle(ΔR: SO3Tangent, Δv: R3Tangent, Δp: R3Tangent, Δab: R3Tangent, Δωb: R3Tangent) -> StateTangent:
        coeffs = []
        coeffs.extend(ΔR.coeffs_copy())
        coeffs.extend(Δv.coeffs_copy())
        coeffs.extend(Δp.coeffs_copy())
        coeffs.extend(Δab.coeffs_copy())
        coeffs.extend(Δωb.coeffs_copy())
        coeffs = np.array(coeffs).flatten()
        return StateTangent(coeffs)

    # Get & Set
    def get_ΔR(self) -> SO3Tangent:
        return SO3Tangent(self._coeffs[0:3].copy())

    def set_ΔR(self, ΔR: SO3Tangent) -> None:
        self._coeffs[0:3] = ΔR.coeffs_copy()

    def get_Δv(self) -> R3Tangent:
        return R3Tangent(self._coeffs[3:6].copy())

    def set_Δv(self, Δv: R3Tangent) -> None:
        self._coeffs[3:6] = Δv.coeffs_copy()

    def get_Δp(self) -> R3Tangent:
        return R3Tangent(self._coeffs[6:9].copy())

    def set_Δp(self, Δp: R3Tangent) -> None:
        self._coeffs[6:9] = Δp.coeffs_copy()

    def get_Δab(self) -> R3Tangent:
        return R3Tangent(self._coeffs[9:12].copy())

    def set_Δab(self, Δab: R3Tangent) -> None:
        self._coeffs[9:12] = Δab.coeffs_copy()

    def get_Δωb(self) -> R3Tangent:
        return R3Tangent(self._coeffs[12:15].copy())

    def set_Δωb(self, Δωb: R3Tangent) -> None:
        self._coeffs[12:15] = Δωb.coeffs_copy()

    # Operators

    # def __matmul__(self, A: np.ndarray) -> np.array:
    #     if A.shape[0] != 15:
    #         raise ValueError(f'Nb. of row should be 15, not {A.shape[0]}.')
    #     return self._coeffs@A

    def __rmatmul__(self, A: np.ndarray) -> np.array:  # A@ΔX
        if A.shape[1] != 15:
            raise ValueError(f'Nb. of colums should be 15, not {A.shape[1]}.')
        return A@self._coeffs
