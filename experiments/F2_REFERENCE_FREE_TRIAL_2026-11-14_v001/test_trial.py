"""Protect label boundaries, reference-free inputs and real checkpoint participation."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('trial',Path(__file__).with_name('trial.py'))
trial=importlib.util.module_from_spec(spec)
if Path(spec.origin).exists():spec.loader.exec_module(trial)


class TrialTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(trial,'own_features'),'trial adapter not implemented')

    def test_missing_target_frames_are_not_interpolated_across_silence(self):
        result=trial.interpolate_voiced(np.array([0.,.02,.04,.06]),np.array([220.,0.,0.,440.]),np.array([.02,.03,.04]))
        np.testing.assert_array_equal(result,np.zeros(3))

    def test_identity_pitch_produces_finite_own_only_features(self):
        f=np.full(100,220.);x=trial.own_features(f)
        self.assertEqual(x.shape[0],100)
        self.assertTrue(np.isfinite(x).all())
        self.assertGreater(x.shape[1],2)

    def test_input_features_have_no_target_or_reference_argument(self):
        with self.assertRaises(TypeError):trial.own_features(np.full(10,220.),reference=np.full(10,440.))

    def test_unvoiced_frames_never_receive_correction(self):
        f=np.array([0.]*10+[220.]*30+[0.]*10)
        d=trial.render_command(f,np.full(50,100.))
        self.assertTrue(np.all(d[f==0]==0.))

    def test_zero_prediction_has_exact_zero_command(self):
        np.testing.assert_array_equal(trial.render_command(np.full(100,220.),np.zeros(100)),np.zeros(100))

    def test_existing_stage_output_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'output';p.mkdir();(p/'keep').write_text('unchanged')
            with self.assertRaises(FileExistsError):trial.new_output(p)
            self.assertEqual((p/'keep').read_text(),'unchanged')

    def test_checkpoint_reload_reproduces_prediction_without_training_data(self):
        self.assertTrue(hasattr(trial,'PitchModel'),'sequence model not implemented')
        import torch
        torch.manual_seed(7);model=trial.PitchModel()
        with torch.no_grad():model.head[-1].bias.fill_(.4)
        x=trial.own_features(np.full(40,220.))
        before=trial.predict_array(model,x)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'model.pt'
            torch.save({'state':model.state_dict(),'schema':'own-pitch-context-v1'},path)
            restored=trial.load_model(path)
            np.testing.assert_array_equal(trial.predict_array(restored,x),before)
            self.assertGreater(np.abs(before).max(),50.)

    def test_zeroed_model_cannot_silently_fall_back_to_reference(self):
        self.assertTrue(hasattr(trial,'PitchModel'),'sequence model not implemented')
        import torch
        model=trial.PitchModel()
        with torch.no_grad():
            for p in model.parameters():p.zero_()
        np.testing.assert_array_equal(trial.predict_array(model,trial.own_features(np.full(60,220.))),np.zeros(60))

    def test_zero_command_bypasses_resynthesis_exactly(self):
        self.assertTrue(hasattr(trial,'synthesize'),'renderer adapter not implemented')
        t=np.arange(24000)/24000;x=.1*np.sin(2*np.pi*220*t)
        y,mask=trial.synthesize(x,24000,np.zeros(50))
        np.testing.assert_array_equal(y,x);np.testing.assert_array_equal(mask,np.zeros(24000))

    def test_known_hundred_cent_command_is_reproduced(self):
        self.assertTrue(hasattr(trial,'synthesize'),'renderer adapter not implemented')
        import parselmouth
        t=np.arange(48000)/24000;x=.1*np.sin(2*np.pi*220*t)
        y,mask=trial.synthesize(x,24000,np.full(100,100.))
        f=parselmouth.Sound(y,24000).to_pitch(time_step=.02,pitch_floor=65,pitch_ceiling=850).selected_array['frequency']
        actual=1200*np.log2(np.median(f[f>0])/220)
        self.assertLess(abs(actual-100),5)
        self.assertEqual(len(x),len(y))

    def test_context_features_are_finite_for_a_silent_recording(self):
        self.assertTrue(hasattr(trial,'context_features'),'context baseline not implemented')
        x=trial.context_features(np.zeros(100))
        self.assertEqual(len(x),100);self.assertTrue(np.isfinite(x).all())

    def test_context_features_change_when_neighboring_notes_change(self):
        self.assertTrue(hasattr(trial,'context_features'),'context baseline not implemented')
        a=np.full(100,220.);b=a.copy();b[35:45]=330.
        x,y=trial.context_features(a),trial.context_features(b)
        self.assertGreater(float(np.max(np.abs(x[50]-y[50]))),.1)

    def test_controlled_note_practice_preserves_voicing_and_has_known_inverse(self):
        self.assertTrue(hasattr(trial,'make_practice'),'controlled practice not implemented')
        f=np.r_[np.zeros(10),np.full(20,220.),np.full(20,247.),np.full(20,262.),np.full(20,294.),np.zeros(10)]
        changed,label=trial.make_practice(f,17)
        np.testing.assert_array_equal(changed==0,f==0)
        voiced=f>0
        actual=1200*np.log2(changed[voiced]/f[voiced])
        np.testing.assert_allclose(actual,-label[voiced],atol=1e-8)
        self.assertTrue(np.any(np.abs(label)>60))
        self.assertTrue(np.any(label[voiced]==0))


if __name__=='__main__':unittest.main()
