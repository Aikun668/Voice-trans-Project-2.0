"""Save one unprocessed contextual excerpt with exact PCM verification."""
from pathlib import Path
import hashlib
import json
import wave

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
SOURCE = ROOT / 'test_dataset/my_singing/shige_medium.wav'
CLIP = HERE / 'shige_medium_10s-12.5s_original.wav'
META = HERE / 'excerpt_v001.json'
START, END = 10.0, 12.5


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if CLIP.exists() or META.exists():
        raise FileExistsError('Refuse to overwrite existing excerpt')
    source_hash = digest(SOURCE)
    with wave.open(str(SOURCE), 'rb') as wav:
        params = wav.getparams()
        start, end = round(START * wav.getframerate()), round(END * wav.getframerate())
        wav.setpos(start)
        samples = wav.readframes(end - start)
    with CLIP.open('xb') as raw:
        with wave.open(raw, 'wb') as out:
            out.setparams((params.nchannels, params.sampwidth, params.framerate, 0, params.comptype, params.compname))
            out.writeframes(samples)
    with wave.open(str(CLIP), 'rb') as out:
        assert out.getnframes() == end - start and out.readframes(out.getnframes()) == samples
    if digest(SOURCE) != source_hash:
        raise RuntimeError('Original source changed')
    manifest = {'source': str(SOURCE), 'source_sha256': source_hash, 'excerpt': str(CLIP),
                'excerpt_sha256': digest(CLIP), 'source_start_seconds': START, 'source_end_seconds': END,
                'candidate_source_seconds': [11.03, 11.23], 'candidate_excerpt_seconds': [1.03, 1.23],
                'operation': 'one continuous PCM excerpt; no gain, processing, reordering or stitching',
                'pcm_samples_match_source': True, 'human_label': 'pending_jade'}
    META.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
