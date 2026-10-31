"""Compare extraction methods on saved solver waves, without another FDTD run.

python check_port_waves.py path/to/port_waves.npz
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'plugins'))
from wave_matrix import scattering_from_waves, quality_metrics


def main(path):
    with np.load(path, allow_pickle=False) as waves:
        a, b = waves['incident'], waves['reflected']
        freq = waves['frequency_hz']
    corrected, condition = scattering_from_waves(a, b)
    old = b / np.diagonal(a, axis1=1, axis2=2)[:, None, :]
    result = {'points': len(freq), 'start_hz': float(freq[0]), 'stop_hz': float(freq[-1]),
              'old_ratio': quality_metrics(old), 'complete_wave_matrix': quality_metrics(corrected),
              'max_condition': float(condition.max())}
    print(json.dumps(result, indent=2))
    if result['complete_wave_matrix']['max_singular_value'] > 1.01:
        raise SystemExit('Full-matrix passivity failed at the 1% diagnostic tolerance')


if __name__ == '__main__':
    main(sys.argv[1])
