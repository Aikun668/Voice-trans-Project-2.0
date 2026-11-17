from pathlib import Path
import importlib.util
import unittest
import numpy as np


class AdapterTests(unittest.TestCase):
    def load(self):
        path = Path(__file__).with_name('residual_adapter.py')
        self.assertTrue(path.exists(), 'Neural residual synthesis adapter is missing')
        spec = importlib.util.spec_from_file_location('residual_adapter', path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_disabled_mapping_is_identity_even_when_reconstruction_is_damaged(self):
        m = self.load()
        sr=44100
        t=np.arange(sr)/sr
        source=.2*np.sin(2*np.pi*235*t)+.03*np.sin(2*np.pi*15000*t)
        frames=len(source)//256+1
        damaged=np.full((frames,80),-3.,dtype=np.float32)
        actual,meta=m.render(source,damaged,damaged,strength=1.)
        self.assertEqual(len(actual),len(source))
        self.assertLess(float(np.max(abs(actual-source))),1e-6)
        self.assertEqual(meta['model_delta_rms_db'],0.)

    def test_only_model_difference_is_applied_and_uncovered_highs_are_preserved(self):
        m=self.load()
        sr=44100
        t=np.arange(sr)/sr
        source=.15*np.sin(2*np.pi*440*t)+.02*np.sin(2*np.pi*15000*t)
        frames=len(source)//256+1
        reconstructed=np.full((frames,80),-3.,dtype=np.float32)
        mapped=reconstructed+.05
        actual,meta=m.render(source,mapped,reconstructed,strength=1.)
        stable=slice(2048,-2048)
        measure=lambda x,f:abs(np.sum(x[stable]*np.exp(-2j*np.pi*f*t[stable])))
        low_gain=20*np.log10(measure(actual,440)/measure(source,440))
        high_gain=20*np.log10(measure(actual,15000)/measure(source,15000))
        self.assertAlmostEqual(low_gain,1.,delta=.04)
        self.assertAlmostEqual(high_gain,0.,delta=.04)
        self.assertTrue(np.isfinite(actual).all())
        self.assertAlmostEqual(meta['model_delta_rms_db'],1.,delta=.001)


if __name__=='__main__':
    unittest.main()
