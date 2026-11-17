import importlib.util
from pathlib import Path
import unittest
import numpy as np


class ContractTests(unittest.TestCase):
    def load(self):
        path = Path(__file__).with_name('feature_contract.py')
        self.assertTrue(path.exists(), 'Time-aligned feature contract adapter is missing')
        spec = importlib.util.spec_from_file_location('feature_contract', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_f0_uses_frame_centers_and_keeps_unvoiced_gap(self):
        m = self.load()
        sr, hop = 22050, 128
        t = np.arange(sr * 2) / sr
        voiced = (t >= .3) & (t < 1.7) & ~((t >= .95) & (t < 1.15))
        audio = .2 * np.sin(2 * np.pi * 240 * t) * voiced
        frames = len(audio) // hop + 1
        f0, info = m.aligned_f0(audio, frames, sr, hop)
        grid = np.arange(frames) * hop / sr
        stable = ((grid > .4) & (grid < .85)) | ((grid > 1.3) & (grid < 1.6))
        self.assertLess(float(np.median(abs(f0[stable] - 240))), 1)
        self.assertTrue(np.all(f0[(grid > 1.01) & (grid < 1.09)] == 0))
        self.assertLessEqual(info['max_nearest_time_error_seconds'], hop / sr / 2 + 1e-8)
        self.assertFalse(info['pitch_correction_applied'])

    def test_exact_external_duration_rejects_missing_audio(self):
        m = self.load()
        signal = .1 * np.sin(2 * np.pi * 220 * np.arange(1024) / 22050)
        result = m.to_external(signal, 2037)
        self.assertEqual(len(result), 2037)
        self.assertTrue(np.isfinite(result).all())
        with self.assertRaises(ValueError):
            m.to_external(signal, 3000)

    def test_moving_pitch_is_assigned_to_its_physical_timestamp(self):
        m = self.load()
        sr, hop = 22050, 128
        t = np.arange(sr * 2) / sr
        audio = .2 * np.sin(2 * np.pi * (200 * t + 50 * t * t))
        count = len(audio) // hop + 1
        f0, _ = m.aligned_f0(audio, count, sr, hop)
        centers = np.arange(count) * hop / sr
        stable = (centers > .3) & (centers < 1.7)
        self.assertLess(float(np.median(abs(f0[stable] - (200 + 100 * centers[stable])))), 1.5)


if __name__ == '__main__':
    unittest.main()
