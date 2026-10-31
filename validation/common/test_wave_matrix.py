"""Network-physics regressions independent of any field solver."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'plugins'))
from wave_matrix import scattering_from_waves, quality_metrics


class WaveMatrixTests(unittest.TestCase):
    def test_cross_incident_waves_do_not_create_gain(self):
        # A matched reciprocal attenuator with reflected waves returning
        # from the external terminations in both independent experiments.
        expected = np.array([[[0, .9j], [.9j, 0]]])
        a = np.array([[[1, .3], [.3, 1]]])
        b = expected @ a
        naive = b / np.diagonal(a, axis1=1, axis2=2)[:, None, :]
        self.assertGreater(quality_metrics(naive)['max_singular_value'], 1)
        actual, _ = scattering_from_waves(a, b)
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        self.assertAlmostEqual(quality_metrics(actual)['max_singular_value'], .9)

    def test_diagonal_excitation_recovers_standard_ratios(self):
        a = np.array([np.diag([2+3j, 4-2j])])
        expected = np.array([[[.1j, .7], [.7, .1j]]])
        actual, _ = scattering_from_waves(a, expected @ a)
        np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_independent_excitation_scales_do_not_change_result(self):
        a = np.array([[[1e-12, .2e5j], [.1e-12j, 1e5]]])
        expected = np.array([[[.1, .8j], [.8j, .1]]])
        actual, condition = scattering_from_waves(a, expected @ a)
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        self.assertLess(condition.max(), 2)

    def test_single_port(self):
        actual, _ = scattering_from_waves([[[2j]]], [[[1j]]])
        np.testing.assert_allclose(actual, [[[.5]]])

    def test_missing_or_dependent_excitations_fail(self):
        for a in (np.zeros((1,2,2)), np.ones((1,2,2)), np.ones((1,2,1))):
            with self.assertRaises(ValueError):
                scattering_from_waves(a, a)

    def test_nonfinite_samples_fail(self):
        with self.assertRaises(ValueError):
            scattering_from_waves([[[1]]], [[[np.nan]]])

    def test_column_power_is_not_sufficient_for_passivity(self):
        s = np.full((1,2,2), .6)
        q = quality_metrics(s)
        self.assertLess(q['max_single_excitation_power'], 1)
        self.assertGreater(q['max_singular_value'], 1)


if __name__ == '__main__':
    unittest.main()
