"""Behavioral risk tests for the frozen MusicBERT adapter (stdlib unittest)."""
import importlib
from pathlib import Path
import tempfile
import unittest
import numpy as np


class AdapterRisks(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('adapter')
        except ModuleNotFoundError as exc:
            self.fail('Required frozen-model adapter is not implemented: '+str(exc))

    def test_silence_produces_no_notes(self):
        a=self.api()
        self.assertEqual(a.segment_notes(np.arange(100)*.02,np.zeros(100),np.zeros(100,dtype=bool)),[])

    def test_short_voiced_burst_is_never_actionable(self):
        a=self.api()
        t=np.arange(30)*.02;f=np.zeros(30);f[8:12]=220
        notes=a.segment_notes(t,f,f>0)
        self.assertTrue(all(not n['eligible'] for n in notes))

    def test_stable_note_retains_original_contour(self):
        a=self.api()
        t=np.arange(100)*.02;f=220*2**(.12*np.sin(t*2*np.pi*5)/12)
        notes=a.segment_notes(t,f,np.ones(100,dtype=bool))
        self.assertEqual(len(notes),1)
        self.assertEqual(notes[0]['pitch'],57)
        self.assertTrue(notes[0]['eligible'])

    def test_octuple_roundtrip_and_bad_pitch_rejection(self):
        a=self.api();codec=a.Codec()
        note={'start':.5,'end':1.,'pitch':60,'center':60.,'eligible':True,'agreement':1.}
        ids=codec.encode([note])
        self.assertEqual(ids.shape,(24,))
        self.assertEqual(codec.symbols[int(ids[11])],'<3-60>')
        with self.assertRaises(ValueError):codec.encode([{**note,'pitch':-1}])

    def test_existing_output_is_rejected_without_touching_contents(self):
        a=self.api()
        with tempfile.TemporaryDirectory(dir=a.HERE) as root:
            p=Path(root);(p/'sentinel').write_text('keep')
            with self.assertRaises(FileExistsError):a.new_output(p)
            self.assertEqual((p/'sentinel').read_text(),'keep')

    def test_output_outside_experiment_is_rejected(self):
        a=self.api()
        with self.assertRaises(ValueError):a.new_output(a.PROJECT/'outside-prohibited-test')

    def test_weak_odds_and_short_note_do_not_move(self):
        a=self.api()
        note={'start':.2,'end':.6,'pitch':60,'center':60.,'eligible':True,'agreement':1.}
        p=np.zeros(128);p[60]=.4;p[61]=.5
        self.assertFalse(a.choose_target(note,p)['accepted'])
        p[60]=.01;p[61]=.9
        self.assertTrue(a.choose_target(note,p)['accepted'])
        self.assertFalse(a.choose_target({**note,'eligible':False},p)['accepted'])

    def test_unchanged_note_is_not_snapped_to_temperament(self):
        a=self.api()
        note={'start':.2,'end':.8,'pitch':60,'center':60.3,'eligible':True,'agreement':1.}
        p=np.zeros(128);p[60]=.9;p[61]=.01
        self.assertFalse(a.choose_target(note,p)['accepted'])

    def test_zero_command_bypasses_exact_samples(self):
        a=self.api();x=np.random.default_rng(41).normal(0,.05,4800)
        y,mask=a.synthesize(x,48000,np.zeros(5))
        self.assertTrue(np.array_equal(x,y))
        self.assertFalse(mask.any())

    def test_command_is_zero_outside_selected_note(self):
        a=self.api();grid=np.arange(100)*.02
        rows=[{'start':.4,'end':1.,'accepted':True,'shift_cents':100.}]
        cmd=a.make_command(grid,rows)
        self.assertTrue(np.all(cmd[(grid<.4)|(grid>=1.)]==0))
        self.assertAlmostEqual(float(cmd.max()),100.)

    def test_reference_guard_allows_torch_reference_code_but_blocks_dataset(self):
        a=self.api();entry=importlib.import_module('infer')
        source=a.PROJECT/'test_dataset/input.wav';out=a.HERE/'test-output-unused'
        self.assertFalse(entry.is_forbidden_read(a.PROJECT/'third_party_neuralsvb/venv/Lib/site-packages/torch/ao/nn/quantized/reference/__init__.py',source,out))
        self.assertTrue(entry.is_forbidden_read(a.PROJECT/'test_dataset/reference/vocals/example.wav',source,out))


if __name__=='__main__':unittest.main(verbosity=2)
