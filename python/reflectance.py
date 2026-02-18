"""Utility functions for reflectance models.

Author: Travis Driver
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional

import numpy as np


class ReflectanceFactorType(Enum):
    """Enum for reflectance models.
    
    Factors names that end with 'Factor2' indicate that the factor uses scale and bias correction.
    """
    LunarLambert_AffinePoly3Factor2 = "LunarLambert_AffinePoly3Factor2"
    LunarLambert_AffinePoly4Factor2 = "LunarLambert_AffinePoly4Factor2"
    McEwenFactor2 = "McEwenFactor2"


@dataclass(frozen=True)
class ReflectanceParams:
    """Stores reflectance function parameters.

    Attributes:
        factor_type: Reflectance model type.
        phase_weighting_params: Parameters for the phase weighting function.
        phase_correction_params: Parameters for the phase correction function.
    """
    factor_type: ReflectanceFactorType
    phase_weighting_params: Optional[List[float]] = None
    phase_correction_params: Optional[List[float]] = None


def reflectance_angles(
        R_CB: np.ndarray, r_BC_C: np.ndarray, s_C: np.ndarray, l_B: np.ndarray, n_B: np.ndarray
    ) -> Tuple[float, float, float]:
    """Computes reflectance geometry.
    
    Args:
        R_CB: Rotation matrix from world to camera.
        r_BC_C: Translation vector from camera to world, expressed in camera coordinates.
        s_C: Sun vector in camera coordinates.
        l_B: Location of the landmark in world coordinates. Either shape (3,) or (N, 3).
        n_B: Surface normal vector in world coordinates. Either shape (3,) or (N, 3).
    
    Returns:
        Cosine of the incidence and emission angle and the phase angle in radians.
    """
    s_B = (R_CB.T @ s_C.reshape((3, -1))).flatten()
    r_B = (-R_CB.T @ r_BC_C.reshape((3, -1))).flatten()

    # Compute incidence, emission, and phase angles.
    if n_B.ndim == 2:
        e_B = r_B[None, ...] - l_B
        e_B /= np.linalg.norm(e_B, axis=1)[..., None]
        ci = np.sum(s_B[None, ...] * n_B, axis=1)
        cb = np.sum(e_B * n_B, axis=1)
        phi = np.arccos(np.sum(s_B[None, ...] * e_B, axis=1))
    else:
        e_B = r_B - l_B
        e_B /= np.linalg.norm(e_B)
        ci = np.sum(s_B * n_B)
        cb = np.sum(e_B * n_B)
        phi = np.arccos(np.sum(s_B * e_B))

    return ci, cb, phi


def lunarlambert_affine_poly3(
    R_CB: np.ndarray, 
    r_BC_C: np.ndarray, 
    s_C: np.ndarray, 
    l_B: np.ndarray, 
    n_B: np.ndarray,
    phase_weighting_params: List[float],
    phase_correction_params: List[float],
) -> Tuple[float, float, float, float]:
    """Computes reflectance using the Lunar-Lambert model with affine phase weighting and 3rd-order polynomial phase 
    correction.

    Refs:
    - S. Schröder et al. “Resolved spectrophotometric properties of the Ceres surface from Dawn Framing Camera images,” 
        Icarus, 2017. https://doi.org/10.1016/j.icarus.2017.01.026
    
    Args:
        R_CB: Rotation matrix from world to camera.
        r_BC_C: Translation vector from camera to world, expressed in camera coordinates.
        s_C: Sun vector in camera coordinates.
        l_B: Location of the landmark in world coordinates. Either shape (3,) or (N, 3).
        n_B: Surface normal vector in world coordinates. Either shape (3,) or (N, 3).
        phase_weighting_params: Parameters for the phase weighting function.
        phase_correction_params: Parameters for the phase correction function.

    Returns:
        Reflectance value.
    """
    # Compute incidence, emission, and phase angles.
    ci, cb, phi = reflectance_angles(R_CB, r_BC_C, s_C, l_B, n_B)
        
    # Compute phase weighting.
    w0, w1 = phase_weighting_params
    L = w0 + w1 * phi

    # Compute phase correction.
    c0, c1, c2, c3 = phase_correction_params
    aeq = c0 + c1 * phi + c2 * phi ** 2 + c3 * phi ** 3

    # Compute reflectance.
    reflect = aeq * ((1 - L) * ci + L * 2.0 * ci / (ci + cb))

    return reflect, np.arccos(ci), np.arccos(cb), phi


def lunarlambert_affine_poly4(
    R_CB: np.ndarray, 
    r_BC_C: np.ndarray, 
    s_C: np.ndarray, 
    l_B: np.ndarray, 
    n_B: np.ndarray,
    phase_weighting_params: List[float],
    phase_correction_params: List[float],
) -> Tuple[float, float, float, float]:
    """Computes reflectance using the Lunar-Lambert model with affine phase weighting and 4th-order polynomial phase 
    correction.

    Refs: 
    - S. Schröder et al. “Resolved photometry of Vesta reveals physical properties of crater regolith,” Planetary and 
        Space Science, 2013. https://doi.org/10.1016/j.pss.2013.06.009
    
    Args:
        R_CB: Rotation matrix from world to camera.
        r_BC_C: Translation vector from camera to world, expressed in camera coordinates.
        s_C: Sun vector in camera coordinates.
        l_B: Location of the landmark in world coordinates. Either shape (3,) or (N, 3).
        n_B: Surface normal vector in world coordinates. Either shape (3,) or (N, 3).
        phase_weighting_params: Parameters for the phase weighting function.
        phase_correction_params: Parameters for the phase correction function.

    Returns:
        Reflectance value.
    """
    # Compute incidence, emission, and phase angles.
    ci, cb, phi = reflectance_angles(R_CB, r_BC_C, s_C, l_B, n_B)
        
    # Compute phase weighting.
    w0, w1 = phase_weighting_params
    L = w0 + w1 * phi

    # Compute phase correction.
    c0, c1, c2, c3, c4 = phase_correction_params
    aeq = c0 + c1 * phi + c2 * phi ** 2 + c3 * phi ** 3 + c4 * phi **4

    # Compute reflectance.
    reflect = aeq * ((1 - L) * ci + L * 2.0 * ci / (ci + cb))

    return reflect, np.arccos(ci), np.arccos(cb), phi


def mcewen(
    R_CB: np.ndarray, r_BC_C: np.ndarray, s_C: np.ndarray, l_B: np.ndarray, n_B: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Computes reflectance using the McEwen model with the exponential phase weighting proposed by Gaskell.
    
    Args:
        R_CB: Rotation matrix from world to camera.
        r_BC_C: Translation vector from camera to world, expressed in camera coordinates.
        s_C: Sun vector in camera coordinates.
        l_B: Location of the landmark in world coordinates. Either shape (3,) or (N, 3).
        n_B: Surface normal vector in world coordinates. Either shape (3,) or (N, 3).

    Returns:
        Reflectance value.
    """
    # Compute incidence, emission, and phase angles.
    ci, cb, phi = reflectance_angles(R_CB, r_BC_C, s_C, l_B, n_B)
        
    # Compute phase weighting.
    L = np.exp(-3.0 * phi / np.pi)

    # Compute reflectance.
    reflect = (1 - L) * ci + L * 2.0 * ci / (ci + cb)

    return reflect, np.arccos(ci), np.arccos(cb), phi


def compute_reflectance(
    R_CB: np.ndarray, 
    r_BC_C: np.ndarray, 
    s_C: np.ndarray, 
    l_B: np.ndarray, 
    n_B: np.ndarray,
    reflectance_params: ReflectanceParams,
) -> Tuple[float, float, float, float]:
    """Computes reflectance based on the specified reflectance model.
    
    Args:
        R_CB: Rotation matrix from world to camera.
        r_BC_C: Translation vector from camera to world, expressed in camera coordinates.
        s_C: Sun vector in camera coordinates.
        l_B: Location of the landmark in world coordinates. Either shape (3,) or (N, 3).
        n_B: Surface normal vector in world coordinates. Either shape (3,) or (N, 3).
        reflectance_params: ReflectanceParams object containing model type and parameters.

    Returns:
        Reflectance value.
    """
    if reflectance_params.factor_type == ReflectanceFactorType.LunarLambert_AffinePoly3Factor2:
        return lunarlambert_affine_poly3(
            R_CB,
            r_BC_C,
            s_C,
            l_B,
            n_B,
            reflectance_params.phase_weighting_params,
            reflectance_params.phase_correction_params,
        )
    if reflectance_params.factor_type == ReflectanceFactorType.LunarLambert_AffinePoly4Factor2:
        return lunarlambert_affine_poly4(
            R_CB,
            r_BC_C,
            s_C,
            l_B,
            n_B,
            reflectance_params.phase_weighting_params,
            reflectance_params.phase_correction_params,
        )
    if reflectance_params.factor_type == ReflectanceFactorType.McEwenFactor2:
        return mcewen(
            R_CB,
            r_BC_C,
            s_C,
            l_B,
            n_B,
        )
    
    raise NotImplementedError(f"Reflectance model {reflectance_params.factor_type} not implemented.")
