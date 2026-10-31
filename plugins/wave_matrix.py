"""Recover S from independent port excitations, including incident cross-waves.

At each frequency B = S A. Columns are experiments, rows are ports. A
non-excited transmission-line port need not be matched to the requested
reference impedance; its incident wave must not silently be discarded.
This module has no solver or KiCad dependency.
"""
import numpy as np


def scattering_from_waves(incident, reflected, condition_limit=1e10):
    """Return (S, condition numbers) for a complete square wave matrix.

    Reject missing, nonfinite or ill-conditioned experiments instead of
    hiding their loss of information with a pseudoinverse.
    """
    a = np.asarray(incident, dtype=complex)
    b = np.asarray(reflected, dtype=complex)
    if a.ndim != 3 or a.shape != b.shape or a.shape[1] != a.shape[2] or not a.shape[1]:
        raise ValueError('Wave matrices must have matching (frequency, port, excitation) square shapes')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Wave matrices contain nonfinite values')
    # Scaling each experiment preserves B = S A and avoids rejecting an
    # otherwise independent set simply because source amplitudes differ.
    scale = np.max(np.abs(a), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError('An excitation has no incident wave')
    a, b = a / scale, b / scale
    condition = np.linalg.cond(a)
    bad = np.flatnonzero(~np.isfinite(condition) | (condition > condition_limit))
    if bad.size:
        raise ValueError('Incident-wave matrix is ill-conditioned at frequency index %d' % bad[0])
    # Solve S A = B without explicitly constructing inv(A).
    s = np.linalg.solve(a.swapaxes(-1, -2), b.swapaxes(-1, -2)).swapaxes(-1, -2)
    return s, condition


def quality_metrics(s):
    """Necessary passive-network checks for equal, real reference impedances.

    Column power tests only one incident port at a time. The largest
    singular value tests every coherent combination of incident waves.
    Neither test establishes measurement accuracy or causality.
    """
    s = np.asarray(s)
    if s.ndim != 3 or s.shape[1] != s.shape[2] or not np.isfinite(s).all():
        raise ValueError('S must be a finite square frequency sweep')
    sigma = np.linalg.svd(s, compute_uv=False)[:, 0]
    power = np.sum(np.abs(s) ** 2, axis=1)
    return {
        'max_singular_value': float(sigma.max()),
        'worst_singular_frequency_index': int(np.argmax(sigma)),
        'max_single_excitation_power': float(power.max()),
        'max_reciprocity_error': float(np.max(np.abs(s - s.swapaxes(-1, -2)))),
    }
