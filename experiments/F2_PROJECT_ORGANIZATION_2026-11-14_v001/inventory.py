"""Record roles and source identities without moving or altering experiments."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

M = Path(__file__).resolve().parents[2]
R = Path(__file__).resolve().parent
H = Path('C:/Users/杨玉璋/.codex/worktrees/f2-collaboration-handoff/Voice-trans-Project-2.0')
I = Path('C:/Users/杨玉璋/.codex/worktrees/f2-initial-plan/Voice-trans-Project-2.0')
ROOTS = {'main': M, 'handoff': H, 'initial': I}
AUDIT = M / 'experiments/F2_VERSION_AUDIT_2026-11-14_v001'
CONFIRMED = H / 'experiments/F2_CONFIRMED_RETRAIN_2026-11-09_v001'
TRIAL = M / 'experiments/F2_REFERENCE_FREE_TRIAL_2026-11-14_v001'
SIDE = M / 'experiments/F2_MUSICBERT_FROZEN_2026-11-14_v001'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(name, data):
    with (R / name).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


def write_csv(name, rows):
    with (R / name).open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    names = ['retained_dependencies.json', 'source_bindings.csv', 'asset_roles.json',
             'experiment_roles.csv', 'inventory_summary.json']
    assert not any((R / n).exists() for n in names), 'Inventory already exists; use a new run.'
    edges = [
        (H / 'f2/render_reference.py', H / 'experiments/F2_MODEL_LISTEN_2026-11-09_v002/run.py',
         'F2_MODEL_LISTEN_2026-11-09_v002/run.py', '动态导入工程检查代码'),
        (H / 'f2/model_sequence.py', H / 'experiments/F2_REFERENCE_SEQUENCE_2026-11-09_v001/render_v002',
         'F2_REFERENCE_SEQUENCE_2026-11-09_v001/render_v002', '读取直接参考对照数据'),
        (H / 'f2/reference_sequence.py', I / 'experiments/F2_TASK3_PAIR_AUDIT_2026-11-05_v001/逐字候选对照.csv',
         '逐字候选对照.csv', '读取辅助逐字上下文，不替代最终切口'),
        (M / 'experiments/F1_demo_chain_2026-11-02_v003/run.py',
         M / 'experiments/F1_demo_chain_2026-11-02_v001/run.py',
         'F1_demo_chain_2026-11-02_v001/run.py', '复用母带等函数'),
    ]
    dependencies = []
    for consumer, target, literal, purpose in edges:
        assert consumer.is_file() and target.exists()
        assert literal in consumer.read_text(encoding='utf-8-sig')
        dependencies.append({'consumer': str(consumer), 'target': str(target),
                             'purpose': purpose, 'literal_verified': True,
                             'physical_action': 'retain_in_place'})
    write_json('retained_dependencies.json', {'scope': '四条已逐条核对的调用关系，不是全项目依赖闭合证明',
                                            'edges': dependencies,
                                            'other_retained': ['f2公共函数', '旧权重与训练表', '原始素材和环境']})

    pairs = json.loads((CONFIRMED / 'confirmed_pairs.json').read_text(encoding='utf-8-sig'))
    assert len(pairs) == 17
    bindings = []
    for pair in pairs:
        for role in ('self', 'reference'):
            source = pair[role]
            actual = sha(source['path'])
            assert actual == source['sha256'], source['path']
            assert all(abs(t * source['rate'] - frame) <= 1
                       for t, frame in zip(source['seconds'], source['frames']))
            bindings.append({'pair_id': pair['id'], 'song': pair['song'], 'phrase': pair['phrase'],
                             'role': role, 'path': source['path'], 'sha256': actual,
                             'seconds_start': source['seconds'][0], 'seconds_end': source['seconds'][1],
                             'sample_start': source['frames'][0], 'sample_end': source['frames'][1],
                             'sample_rate': source['rate'], 'verified': True,
                             'authority': 'confirmed_pairs.json；不是逐帧目标音真值'})
    write_csv('source_bindings.csv', bindings)

    assets = []
    for role, path, expected, state in [
        ('confirmed_cut_authority', CONFIRMED / 'confirmed_pairs.json', None, '17对已确认'),
        ('legacy_reference_pointer', H / 'f2/active_pitch_model.json', None, '仅旧有参考实验'),
        ('legacy_reference_weights', CONFIRMED / 'results/model.pkl',
         '08e8d7e9f9d58c93784bd8ee85dbd429a0f1f879db21a41b2702e217fb16dd1f', '历史基线'),
        ('mainline_failed_weights', TRIAL / 'practice_training_v001/model.pt',
         'd63eea5b0dea900aa4fb100fbb0159c0e4cba3177821d24ac6e20ceebdca955d', 'jade_rejected'),
        ('sideline_frozen_weights', SIDE / 'assets/checkpoint_last_musicbert_small.pt',
         'f5bb685afe128bc9e0e445be51b5333b98be6a5456df850bd9c9f2cc80bb7275', 'effectiveness_gate_failed; pending_jade'),
        ('mainline_label_manifest', TRIAL / 'data_v001/label_manifest.json', None, '机器监督与本轮划分'),
    ]:
        actual = sha(path)
        assert expected is None or actual == expected
        assets.append({'role': role, 'path': str(path), 'sha256': actual, 'state': state,
                       'formal_default_model': False})
    write_json('asset_roles.json', {'assets': assets, 'global_default_model': None})

    previous = {row['path']: row for row in csv.DictReader((AUDIT / 'experiment_status.csv').open(encoding='utf-8-sig'))}
    rows = []
    excluded = []
    for checkout, root in ROOTS.items():
        for directory in sorted(p for p in (root / 'experiments').iterdir() if p.is_dir()):
            if directory.name.startswith('.'):
                excluded.append({'path': str(directory), 'reason': '工具缓存，不作为实验；原位保留'})
                continue
            old = previous.get(str(directory), {})
            group, state = '历史记录', '原位归档'
            reason = old.get('reason', '历史阶段，保留原位；未据名称推定可安全搬移')
            if directory == TRIAL:
                group, state, reason = '主线', '首次试训失败证据', '自动效果失败，Jade无有用改善；不作默认模型'
            elif directory == SIDE:
                group, state, reason = '次线', '0微调实验，待听感', 'MusicBERT冻结权重；工程通过但效果门槛未过'
            elif directory == M / 'experiments/F2_POPBUTFY_CLEAN_2026-11-13_v001':
                group, state, reason = '共用资产', '公开数据准备', '全量映射仍被主线复核使用；pilot仅历史探索'
            elif directory in (AUDIT, R):
                group, state, reason = '共用资产', '治理记录', '来源核查、入口整理与逐项证据'
            elif directory == CONFIRMED:
                group, state, reason = '共用资产', '17对确认权威；旧模型历史', '确认表持续有效；prepared_v003和旧权重不代表新主线'
            elif directory == H / 'experiments/F2_MANUAL_MARKS_REVIEW_2026-11-09_v002':
                group, state, reason = '共用资产', '人工编辑来源', '最终确认表的编辑来源，保留原位'
            retained = [edge['target'] for edge in dependencies
                        if Path(edge['target']).is_relative_to(directory)]
            rows.append({'checkout': checkout, 'path': str(directory), 'group': group, 'state': state,
                         'reason': reason, 'retained_runtime_targets': json.dumps(retained, ensure_ascii=False),
                         'prior_literal_reference_count': old.get('incoming_code_literal_refs', ''),
                         'physical_action': 'retain_in_place', 'formal_default_model': False})
    write_csv('experiment_roles.csv', rows)
    write_json('inventory_summary.json', {'experiment_instances': len(rows),
               'by_group': dict(Counter(x['group'] for x in rows)), 'source_bindings': len(bindings),
               'unique_source_files': len({x['path'] for x in bindings}),
               'retained_runtime_edges': len(dependencies), 'moved_experiments': 0,
               'excluded_non_experiment_dirs': excluded})
    print(json.dumps({'experiments': len(rows), 'groups': dict(Counter(x['group'] for x in rows)),
                      'source_bindings': len(bindings), 'runtime_edges': len(dependencies)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
