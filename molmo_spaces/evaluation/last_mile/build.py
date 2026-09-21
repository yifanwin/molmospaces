"""只读源数据，确定性构建 P0 子集。python -m ...last_mile.build --help。"""
import argparse
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess

SEED = 20260921
CATEGORIES = {'cup', 'mug', 'bottle', 'winebottle', 'saltshaker', 'peppershaker'}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stable(*parts):
    return hashlib.sha256(json.dumps([SEED, *parts], ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows))
    os.replace(temp, path)


def build(source, assets, output):
    source, assets, output = map(Path, (source, assets, output))
    # 已冻结目录不得因元数据变化或算法变化被静默覆盖。
    old_provenance_path = output / 'provenance.json'
    if old_provenance_path.exists():
        verify_asset_metadata(json.loads(old_provenance_path.read_text()))
    source_sha = digest(source)
    episodes = json.loads(source.read_text())
    annotation_path = assets / 'objects/objathor_metadata/objects_metadata.json.gz'
    with gzip.open(annotation_path, 'rt') as f:
        annotations = json.load(f)
    scene_cache, scene_hashes = {}, {}
    eligible, rejected = [], []
    for index, episode in enumerate(episodes):
        assert (episode['scene_dataset'], episode['data_split'], episode['robot']['robot_name']) == ('procthor-10k', 'val', 'rby1m')
        house = episode['house_index']
        target = episode['task']['pickup_obj_name']
        if house not in scene_cache:
            path = assets / f'scenes/procthor-10k-val/val_{house}_metadata.json'
            scene_cache[house] = json.loads(path.read_text())['objects'] if path.exists() else {}
            scene_hashes[str(house)] = digest(path) if path.exists() else None
        obj = scene_cache[house].get(target, {})
        uid = obj.get('asset_id')
        annotation = annotations.get(uid, {})
        # Scene metadata distinguishes WineBottle/SaltShaker even where aggregate labels merge them.
        raw = obj.get('category')
        normalized = str(raw or '').lower().replace(' ', '').replace('_', '')
        reason = None
        if not uid or not annotation or not raw:
            reason = 'missing_category_metadata'
        elif normalized not in CATEGORIES:
            reason = 'category_excluded'
        row = dict(source_index=index, house=house, target=target,
                   container=episode['task'].get('place_receptacle_name'),
                   asset_id=uid, category=normalized, raw_category=raw,
                   asset_category=annotation.get('category'), source=episode.get('source'),
                   scene_metadata_sha256=scene_hashes[str(house)],
                   source_sha256=source_sha, robot='rby1m', split='val',
                   episode_id=f'{source_sha[:16]}:{index}',
                   seed=episode.get('seed') if episode.get('seed') is not None else int(stable('seed', source_sha, index)[:8], 16))
        if reason:
            rejected.append(dict(row, reason=reason))
        else:
            eligible.append(row)
    unique = {}
    for row in sorted(eligible, key=lambda r: stable('episode', r['episode_id'])):
        key = (row['house'], row['target'])
        if key in unique:
            rejected.append(dict(row, reason='duplicate_target', representative=unique[key]['source_index']))
        else:
            unique[key] = row
    groups = collections.defaultdict(list)
    for row in unique.values():
        groups[row['house']].append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: stable('target', r['house'], r['target']))
    houses = sorted(groups, key=lambda h: stable('house', h))
    pilot_houses = [h for h in houses if len(groups[h]) >= 2][:5]
    if len(pilot_houses) != 5:
        raise ValueError('不足 5 个可提供两个独立目标的 houses')
    pilot = [r for h in pilot_houses for r in groups[h][:2]]
    formal = []
    for level in range(5):
        for house in houses:
            if house not in pilot_houses and len(groups[house]) > level and len(formal) < 100:
                formal.append(groups[house][level])
    assert len(formal) == 100 and len({r['house'] for r in formal}) >= 20
    config = dict(schema_version=1, master_seed=SEED, source_sha256=source_sha,
                  categories=sorted(CATEGORIES), pilot_count=10, formal_count=100,
                  max_per_house=5, repair_robot_base_pose_if_colliding=False,
                  state_atol=1e-10, geometry_atol=1e-8, trajectory_atol=1e-8,
                  restore_repeats=10, control_steps=20, policy='disabled',
                  policy_dt_ms=100, ctrl_dt_ms=20, sim_dt_ms=4)
    if (output / 'config.json').exists() and json.loads((output / 'config.json').read_text()) != config:
        raise ValueError('已有输出配置不匹配；拒绝覆盖')
    expected_manifest = [dict(row, subset=name, subset_index=i)
                         for name, rows in [('pilot', pilot), ('formal', formal)]
                         for i, row in enumerate(rows)]
    existing_manifest = output / 'manifest.jsonl'
    if existing_manifest.exists():
        existing = [json.loads(line) for line in existing_manifest.read_text().splitlines()]
        if existing != expected_manifest:
            raise ValueError('冻结样本发生变化；请使用新输出目录，不得覆盖旧清单')
    for name, rows in [('pilot', pilot), ('formal', formal)]:
        for subset_index, row in enumerate(rows):
            row.update(subset=name, subset_index=subset_index)
        selected = [episodes[r['source_index']] for r in rows]
        atomic_json(output / name / 'benchmark.json', selected)
        lengths = [e.get('source', {}).get('episode_length') for e in selected]
        lengths = [x for x in lengths if x is not None]
        import statistics
        metadata = dict(description=f'P0 {name}: val 内诊断子集，非官方新评测集',
                        num_episodes=len(rows), num_houses=len({r['house'] for r in rows}),
                        object_category_counts=dict(collections.Counter(r['category'] for r in rows)),
                        house_counts=dict(collections.Counter(str(r['house']) for r in rows)),
                        robot_counts={'rby1m': len(rows)}, source_sha256=source_sha,
                        task_cls_counts=dict(collections.Counter(e['task']['task_cls'] for e in selected)),
                        episode_length_stats=dict(min=min(lengths), max=max(lengths), mean=statistics.mean(lengths), median=statistics.median(lengths)) if lengths else None)
        atomic_json(output / name / 'benchmark_metadata.json', metadata)
        jsonl(output / name / 'manifest.jsonl', rows)
    selected_ids = {r['source_index'] for r in pilot + formal}
    reserves = [dict(r, reserve_group='pilot' if r['house'] in pilot_houses else 'formal')
                for h in houses for r in groups[h] if r['source_index'] not in selected_ids]
    jsonl(output / 'manifest.jsonl', pilot + formal)
    jsonl(output / 'reserves.jsonl', reserves)
    jsonl(output / 'excluded.jsonl', rejected)
    atomic_json(output / 'config.json', config)
    repo = Path(__file__).resolve().parents[3]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()
    provenance = dict(schema_version=1, source=str(source.resolve()), source_sha256=source_sha,
                      assets=str(assets.resolve()), annotation_sha256=digest(annotation_path),
                      scene_metadata_sha256=scene_hashes, base_commit=git('rev-parse', 'HEAD'),
                      branch=git('branch', '--show-current'), dirty_summary=git('status', '--short'),
                      selection_sha256=digest(output / 'manifest.jsonl'))
    atomic_json(output / 'provenance.json', provenance)
    assert digest(source) == source_sha
    print(json.dumps({'pilot_houses': pilot_houses, 'formal_houses': len({r['house'] for r in formal}), 'eligible_unique': len(unique), 'reserves': len(reserves)}, ensure_ascii=False))
    return pilot, formal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.assets, args.output)



def verify_asset_metadata(provenance):
    assets = Path(provenance['assets'])
    if digest(assets / 'objects/objathor_metadata/objects_metadata.json.gz') != provenance['annotation_sha256']:
        raise ValueError('资产注释已改变，拒绝混用冻结清单')
    for house, expected in provenance['scene_metadata_sha256'].items():
        path = assets / f'scenes/procthor-10k-val/val_{house}_metadata.json'
        actual = digest(path) if path.exists() else None
        if actual != expected:
            raise ValueError(f'house {house} 场景元数据已改变')


def audit(output):
    """验证冻结清单、源内容和现有 EpisodeSpec schema（需配置资产环境变量）。"""
    from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec, load_all_episodes, BenchmarkMetadata
    output = Path(output)
    provenance = json.loads((output / 'provenance.json').read_text())
    verify_asset_metadata(provenance)
    assert digest(provenance['source']) == provenance['source_sha256']
    assert digest(output / 'manifest.jsonl') == provenance['selection_sha256']
    source = json.loads(Path(provenance['source']).read_text())
    all_rows = []
    houses = {}
    for name, count in [('pilot', 10), ('formal', 100)]:
        raw = json.loads((output / name / 'benchmark.json').read_text())
        assert len(load_all_episodes(output / name)) == count
        BenchmarkMetadata.from_json_file(output / name / 'benchmark_metadata.json')
        rows = [json.loads(s) for s in (output / name / 'manifest.jsonl').read_text().splitlines()]
        assert len(raw) == len(rows) == count
        for e, row in zip(raw, rows):
            EpisodeSpec.model_validate(e)
            assert e == source[row['source_index']]
            assert e['house_index'] == row['house'] and e['task']['pickup_obj_name'] == row['target']
            assert e['data_split'] == 'val' and e['robot']['robot_name'] == 'rby1m'
        metadata = json.loads((output / name / 'benchmark_metadata.json').read_text())
        houses[name] = {r['house'] for r in rows}
        assert metadata['num_episodes'] == count and metadata['num_houses'] == len(houses[name])
        assert metadata['house_counts'] == dict(collections.Counter(str(r['house']) for r in rows))
        assert metadata['object_category_counts'] == dict(collections.Counter(r['category'] for r in rows))
        assert max(collections.Counter(r['house'] for r in rows).values()) <= 5
        all_rows.extend(rows)
    assert len(houses['pilot']) == 5 and len(houses['formal']) >= 20
    assert not houses['pilot'] & houses['formal']
    assert len({(r['house'], r['target']) for r in all_rows}) == 110
    assert all_rows == [json.loads(s) for s in (output / 'manifest.jsonl').read_text().splitlines()]
    return dict(status='passed', source_unchanged=True, pilot_count=10, formal_count=100,
                pilot_houses=len(houses['pilot']), formal_houses=len(houses['formal']))


if __name__ == '__main__':
    main()
