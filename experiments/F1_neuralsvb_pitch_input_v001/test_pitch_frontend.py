import importlib.util
from pathlib import Path
import unittest
import numpy as np


class PitchFrontendTests(unittest.TestCase):
    def load(self):
        path = Path(__file__).with_name('pitch_frontend.py')
        self.assertTrue(path.exists(), 'source-pitch adapter has not been implemented')
        spec = importlib.util.spec_from_file_location('pitch_frontend', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_real_octave_and_silence_are_preserved(self):
        module = self.load()
        sr = 22050
        t = np.arange(sr * 3) / sr
        y = np.zeros_like(t)
        y[t < 1] = .2 * np.sin(2*np.pi*220*t[t < 1])
        y[t >= 2] = .2 * np.sin(2*np.pi*440*t[t >= 2])
        for method in ['praat', 'pyin']:
            f0, meta = module.extract(y, len(y)//128+1, method=method)
            times = np.arange(len(f0))*128/sr
            for lo, hi, expected in [(.2,.8,220),(2.2,2.8,440)]:
                v = f0[(times>lo)&(times<hi)]
                self.assertGreater(np.mean(v>0), .95)
                self.assertLess(abs(1200*np.log2(np.median(v)/expected)), 15)
            self.assertEqual(float(np.max(f0[(times>1.2)&(times<1.8)])), 0.)
            self.assertFalse(meta['melody_correction_applied'])

    def test_praat_frame_grid_tracks_original_event_time(self):
        module = self.load()
        sr=22050
        t=np.arange(sr*2)/sr
        frequency=np.where(t<1,200.,350.)
        y=.2*np.sin(2*np.pi*np.cumsum(frequency)/sr)
        f0, meta=module.extract(y,len(y)//128+1,method='praat')
        crossing=np.where(f0>300)[0][0]*128/sr
        self.assertLess(abs(crossing-1),.03)
        self.assertLessEqual(meta['max_grid_error_seconds'],128/sr/2+1e-8)

    def test_invalid_rate_and_short_support_are_rejected(self):
        module=self.load()
        with self.assertRaises(ValueError):
            module.extract(np.zeros(22050),173,sr=44100,method='pyin')
        with self.assertRaises(ValueError):
            module.extract(np.zeros(22050),500,method='pyin')


if __name__=='__main__':
    unittest.main()
