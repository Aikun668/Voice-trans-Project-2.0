"""Verify this documentation-only reorganization against before snapshots."""
import csv
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

R = Path(__file__).resolve().parent
M = R.parents[1]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def broken_links(path, text):
    text = re.sub(r'```.*?```', '', text, flags=re.S)
    broken = []
    for url in re.findall(r'\[[^\]\n]+\]\(([^)\n]+)\)', text):
        url = unquote(url.strip().strip('<>'))
        if re.match(r'^(?:https?://|mailto:|thread:|codex:|#)', url):
            continue
        file = url.split('#', 1)[0]
        if not file:
            continue
        if re.match(r'^/[A-Za-z]:/', file):
            file = file[1:]
        target = Path(file)
        if not target.is_absolute():
            target = path.parent / target
        if not target.exists():
            broken.append(url)
    return sorted(set(broken))


def main():
    before = json.loads((R / 'before_manifest.json').read_text(encoding='utf-8'))
    protected = json.loads((R / 'protected_before.json').read_text(encoding='utf-8'))
    # The retained listening server continues recording browser requests.
    runtime_log = str(M / 'experiments/F2_MUSICBERT_FROZEN_2026-11-14_v001/server_error.log')
    runtime_changes = []
    if runtime_log in protected and Path(runtime_log).is_file() and sha(runtime_log) != protected[runtime_log]:
        runtime_changes.append({'path': runtime_log, 'sha256_before': protected[runtime_log],
                                'sha256_after': sha(runtime_log),
                                'reason': '既有试听服务的HTTP请求日志继续增长，未由整理脚本改写'})
    changed = [p for p, digest in protected.items() if p != runtime_log
               and (not Path(p).is_file() or sha(p) != digest)]
    assert not changed, changed
    new_broken, legacy_broken, modified = [], [], []
    preserved = []
    for item in before:
        file = Path(item['path'])
        original_bytes = (R / item['snapshot']).read_bytes()
        assert hashlib.sha256(original_bytes).hexdigest() == item['sha256']
        old = original_bytes.decode('utf-8-sig').replace('\r\n', '\n')
        new = file.read_text(encoding='utf-8-sig')
        assert new != old
        modified.append({'path': str(file), 'sha256_before': item['sha256'], 'sha256_after': sha(file)})
        old_broken = set(broken_links(file, old))
        for url in broken_links(file, new):
            (legacy_broken if url in old_broken else new_broken).append({'file': str(file), 'url': url})
        if item['checkout'] != 'main' and file.name != 'AGENTS.md':
            assert new.endswith(old), file
            preserved.append(str(file))
        if item['checkout'] == 'main' and file.name in {
            '新阶段_前三项任务初步计划.md', 'F2_第一轮有参考修音_训练计划书_给第三方Agent.md',
            'TEAM_COLLABORATION_DRAFT.md'}:
            assert new.endswith(old), file
            preserved.append(str(file))
    for file in (M / 'docs/项目分区说明_2026-11-14.md', M / 'docs/archive/README.md', R / 'REPORT.md'):
        for url in broken_links(file, file.read_text(encoding='utf-8')):
            if file == R / 'REPORT.md' and url == 'verification.json':
                continue  # Written only after all checks pass; checked again below.
            new_broken.append({'file': str(file), 'url': url})
    assert not new_broken, new_broken

    dirs = json.loads((R / 'inventory_before.json').read_text(encoding='utf-8'))
    all_old_dirs = [p for paths in dirs.values() for p in paths]
    assert all(Path(p).is_dir() for p in all_old_dirs)
    rows = list(csv.DictReader((R / 'experiment_roles.csv').open(encoding='utf-8-sig')))
    expected = {p for p in all_old_dirs if not Path(p).name.startswith('.')}
    assert {x['path'] for x in rows} == expected
    assert all(x['physical_action'] == 'retain_in_place' for x in rows)
    assert set(x['group'] for x in rows) == {'主线', '次线', '共用资产', '历史记录'}

    bindings = list(csv.DictReader((R / 'source_bindings.csv').open(encoding='utf-8-sig')))
    assert len(bindings) == 34 and len({x['pair_id'] for x in bindings}) == 17
    assert all(sha(x['path']) == x['sha256'] for x in bindings)
    assets = json.loads((R / 'asset_roles.json').read_text(encoding='utf-8'))
    assert assets['global_default_model'] is None
    assert all(sha(x['path']) == x['sha256'] for x in assets['assets'])
    roles = {x['role']: x for x in assets['assets']}
    assert roles['mainline_failed_weights']['sha256'] != roles['sideline_frozen_weights']['sha256']

    roadmap = (M / 'docs/ROADMAP.md').read_text(encoding='utf-8')
    assert '最新实验在次线，不改变主次关系' in roadmap
    assert '主线新训练未启动' in (M / 'docs/开发日志_有参考修音.md').read_text(encoding='utf-8')
    assert all(term in roadmap for term in ('jade_rejected', 'pending_jade', '0微调步', '复制反馈'))
    report = {'documentation_snapshots': len(before), 'modified_documents': modified,
              'protected_files_verified': len(protected) - (runtime_log in protected),
              'protected_files_changed': changed, 'runtime_log_changes': runtime_changes,
              'historical_bodies_preserved': preserved, 'new_broken_links': new_broken,
              'preexisting_historical_broken_links': legacy_broken,
              'experiment_instances_classified': len(rows), 'existing_directories_still_in_place': True,
              'source_binding_rows': len(bindings), 'unique_confirmed_source_files': len({x['path'] for x in bindings}),
              'registered_assets_hashes_verified': True, 'global_default_model': None,
              'mainline_and_sideline_separate': True, 'documentation_verification_pass': True,
              'scope': '文档/来源/保留核验，不是新增训练或修音效果验证'}
    (R / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    assert not broken_links(R / 'REPORT.md', (R / 'REPORT.md').read_text(encoding='utf-8'))
    print(json.dumps({k: v for k, v in report.items() if k not in {
        'modified_documents', 'historical_bodies_preserved', 'preexisting_historical_broken_links'}}, ensure_ascii=False))
    print('Preexisting historical link issues:', len(legacy_broken))


if __name__ == '__main__':
    main()
