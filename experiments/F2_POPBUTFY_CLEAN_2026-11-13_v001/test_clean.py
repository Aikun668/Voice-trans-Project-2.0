"""Guard against silently teaching the model the wrong sentence or leaking songs."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import soundfile as sf

spec = importlib.util.spec_from_file_location('clean', Path(__file__).with_name('clean.py'))
clean = importlib.util.module_from_spec(spec)
if Path(spec.origin).exists():
    spec.loader.exec_module(clean)


class CleanTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(clean, 'match_clips'), 'cleaning adapter has not been implemented')

    def test_equal_number_different_lyrics_must_not_pair(self):
        a = [{'index': 0, 'lyric_norm': 'hello', 'path': 'a0'}]
        b = [{'index': 0, 'lyric_norm': 'goodbye', 'path': 'p0'}]
        self.assertEqual(clean.match_clips(a, b), [])

    def test_shifted_number_is_recovered_by_ordered_lyrics(self):
        a = [{'index': i, 'lyric_norm': t, 'path': 'a'+str(i)}
             for i, t in enumerate(['first line', 'second line', 'third line'])]
        b = [{'index': i+1, 'lyric_norm': t, 'path': 'p'+str(i+1)}
             for i, t in enumerate(['first line', 'second line', 'third line'])]
        self.assertEqual(clean.match_clips(a, b), [(0, 0), (1, 1), (2, 2)])

    def test_missing_lyrics_do_not_create_false_matches(self):
        self.assertEqual(clean.match_clips([{'index': 0, 'lyric_norm': ''}],
                                           [{'index': 0, 'lyric_norm': ''}]), [])

    def test_repeat_match_is_monotonic_without_reusing_clips(self):
        a = [{'index': i, 'lyric_norm': t} for i, t in enumerate(['a', 'chorus', 'b', 'chorus'])]
        b = [{'index': i, 'lyric_norm': t} for i, t in enumerate(['a', 'b', 'chorus'])]
        result = clean.match_clips(a, b)
        self.assertEqual(len({i for i,j in result}), len(result))
        self.assertEqual(len({j for i,j in result}), len(result))
        self.assertTrue(all(i<k and j<l for (i,j),(k,l) in zip(result,result[1:])))
        self.assertTrue(all(a[i]['lyric_norm']==b[j]['lyric_norm'] for i,j in result))

    def test_punctuation_and_case_do_not_prevent_lyric_match(self):
        self.assertEqual(clean.normalize_lyric("  I'll   STAY, with you! "), 'ill stay with you')

    def test_duplicate_recording_connects_song_splits(self):
        rows = [{'song': 'A Song', 'sha256': 'x'}, {'song': 'a_song', 'sha256': 'y'},
                {'song': 'Different Title', 'sha256': 'x'}]
        splits = clean.song_splits(rows)
        self.assertEqual(len({v['split'] for v in splits.values()}), 1)
        self.assertEqual(len({v['split_group'] for v in splits.values()}), 1)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'result'; p.mkdir(); (p/'keep').write_text('original')
            with self.assertRaises(FileExistsError): clean.new_output(p)
            self.assertEqual((p/'keep').read_text(), 'original')

    def test_corrupt_audio_is_recorded_instead_of_dropping_input(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'bad.mp3'; p.write_bytes(b'not audio')
            result = clean.inspect_audio(p)
            self.assertEqual(result['file_status'], 'decode_error')
            self.assertTrue(result['sha256'])
            self.assertIn('error', result)

    def test_silence_is_excluded_but_source_is_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'silence.wav'; sf.write(p, np.zeros(16000), 16000)
            before = p.read_bytes(); result = clean.inspect_audio(p)
            self.assertEqual(result['file_status'], 'silent_or_very_quiet')
            self.assertEqual(result['frames'], 16000)
            self.assertEqual(p.read_bytes(), before)


if __name__ == '__main__': unittest.main()
