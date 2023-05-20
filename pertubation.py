from __future__ import annotations
from manifpy import R3Tangent
import numpy as np

class Pertubation:  # W = [aw, ωw, ar, ωr]

    # Constructor
    def __init__(self, coeffs: np.array) -> None:
        if coeffs.size != 12:
            raise ValueError(f'Size of coeffs should be 12, not {coeffs.size}.')
        self._coeffs = coeffs.flatten().copy()

    def Identity() -> Pertubation:
        coeffs = np.zeros(12)
        return Pertubation(coeffs)
    
    def Random(sigmas: np.array) -> Pertubation:
        if sigmas.size != 4:
            raise ValueError(f'Size of coeffs should be 4, not {sigmas.size}.')
        coeffs = np.array([np.random.normal(0,sigmas[0],3),
                           np.random.normal(0,sigmas[1],3),
                           np.random.normal(0,sigmas[2],3),
                           np.random.normal(0,sigmas[3],3)]).flatten()
        return Pertubation(coeffs)

    def Bundle(aw: R3Tangent, ωw: R3Tangent, ar: R3Tangent, ωr: R3Tangent) -> Pertubation:
        coeffs = []
        coeffs.extend(aw.coeffs_copy())
        coeffs.extend(ωw.coeffs_copy())
        coeffs.extend(ar.coeffs_copy())
        coeffs.extend(ωr.coeffs_copy())
        coeffs = np.array(coeffs).flatten()
        return Pertubation(coeffs)

    # Get & Set
    def get_aw(self) -> R3Tangent:
        return R3Tangent(self._coeffs[0:3].copy())

    def set_aw(self, aw: R3Tangent) -> None:
        self._coeffs[0:3] = aw.coeffs_copy()

    def get_ωw(self) -> R3Tangent:
        return R3Tangent(self._coeffs[3:6].copy())

    def set_ωw(self, ωw: R3Tangent) -> None:
        self._coeffs[3:6] = ωw.coeffs_copy()

    def get_ar(self) -> R3Tangent:
        return R3Tangent(self._coeffs[6:9].copy())

    def set_ar(self, ar: R3Tangent) -> None:
        self._coeffs[6:9] = ar.coeffs_copy()

    def get_ωr(self) -> R3Tangent:
        return R3Tangent(self._coeffs[9:12].copy())

    def set_ωr(self, ωr: R3Tangent) -> None:
        self._coeffs[9:12] = ωr.coeffs_copy()