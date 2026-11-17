"""Acquire pinned upstream assets inside this experiment; never install into old envs."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import time
import requests

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', action='store_true')
    args = ap.parse_args()
    assets = HERE / 'assets'
    assets.mkdir(exist_ok=True)
    session = requests.Session()
    session.headers['User-Agent'] = 'MusicBERT-research-probe/1.0'
    if not (HERE / 'baseline.json').exists():
        roots = [PROJECT / 'docs', PROJECT / 'f2', PROJECT / 'experiments/F2_REFERENCE_FREE_TRIAL_2026-11-14_v001']
        paths = [PROJECT / 'AGENTS.md', PROJECT / 'README.md']
        for root in roots:
            paths.extend(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        source = PROJECT / 'test_dataset/my_singing/rerecord_2026-11-04_f2/wav/心墙01-轻度跑调.wav'
        paths.append(source)
        dump(HERE / 'baseline.json', {'files': {str(p.relative_to(PROJECT)): sha(p) for p in paths},
             'git_status': subprocess.run(['git','status','--porcelain=v1','-uall'],cwd=PROJECT,capture_output=True,text=True,encoding='utf-8').stdout})
    repo = 'microsoft/muzic'
    local_repo = assets / 'muzic_repo'
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=local_repo,text=True).strip()
    print('official commit', commit, flush=True)
    files = {'README.md':'musicbert/README.md', 'preprocess.py':'musicbert/preprocess.py',
             'musicbert_original.py':'musicbert/musicbert/__init__.py', 'LICENSE':'LICENSE'}
    saved = []
    for name, relative in files.items():
        dest = assets / name
        url = f'https://raw.githubusercontent.com/{repo}/{commit}/{relative}'
        content = (local_repo / relative).read_bytes()
        if not dest.exists():
            dest.write_bytes(content)
        elif dest.read_bytes() != content:
            raise ValueError(f'Existing upstream file differs: {dest}')
        saved.append({'path':name,'url':url,'sha256':sha(dest)})
    provenance = {'repo': f'https://github.com/{repo}', 'commit':commit, 'license':'MIT', 'files':saved,
                  'weight_source_folder':'https://drive.google.com/drive/folders/1aGcEB3qZoJ_ZE4qqmrrs6-JAp3z1vkE4',
                  'small_file_id':'1RQvThodh0hbfbZ9SaXm7Q8vXNOFTfYZu', 'expected_bytes':265448349}
    dump(assets / 'upstream.json', provenance)
    if not args.weights:
        return
    dest = assets / 'checkpoint_last_musicbert_small.pt'
    if dest.exists():
        if dest.stat().st_size != provenance['expected_bytes']:
            raise ValueError('Existing checkpoint has unexpected length')
        print('existing checkpoint', dest.stat().st_size, sha(dest), flush=True)
        return
    url = 'https://drive.usercontent.google.com/download'
    params = {'id':provenance['small_file_id'],'export':'download','confirm':'t'}
    response = session.get(url, params=params, stream=True, timeout=(30,90))
    response.raise_for_status()
    if 'text/html' in response.headers.get('Content-Type',''):
        body = response.text
        dump(assets / 'download_response.json', {'status':response.status_code,'html_prefix':body[:3000]})
        confirm = dict(re.findall(r'name="([^"]+)" value="([^"]*)"', body))
        if 'uuid' in confirm:
            params.update({k:html.unescape(v) for k,v in confirm.items()})
            response = session.get(url, params=params, stream=True, timeout=(30,90))
            response.raise_for_status()
        if 'text/html' in response.headers.get('Content-Type',''):
            raise RuntimeError('Google Drive did not return checkpoint bytes; preserved diagnostic response')
    partial = dest.with_suffix('.pt.partial')
    if partial.exists():
        raise FileExistsError(partial)
    size = 0
    tick = time.monotonic()
    with partial.open('xb') as f:
        for chunk in response.iter_content(1024*1024):
            f.write(chunk)
            size += len(chunk)
            if time.monotonic()-tick > 8:
                print('downloaded_bytes',size,flush=True)
                tick = time.monotonic()
    if size != provenance['expected_bytes']:
        raise ValueError(f'Unexpected checkpoint size: {size}')
    partial.rename(dest)
    provenance['checkpoint'] = {'path':dest.name,'bytes':size,'sha256':sha(dest),'download_url':url}
    dump(assets / 'upstream.json', provenance)
    print('checkpoint_ready',size,provenance['checkpoint']['sha256'],flush=True)


if __name__ == '__main__':
    main()
