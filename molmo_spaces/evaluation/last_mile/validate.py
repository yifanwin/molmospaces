"""运行 P0 真实场景恢复验证；失败逐条保留，不自动补样。"""
import argparse
from copy import deepcopy
import gc
import functools
import json
import random
import sys
import time
import traceback
from pathlib import Path

import mujoco
import numpy as np
from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec
from molmo_spaces.evaluation.eval_main import EvalRuntimeParams
from .sampler import P0JsonEvalTaskSampler
from .build import atomic_json, digest, jsonl, audit
from .config import P0Config
from .snapshot import EpisodeSnapshot, integration_state, override_base_pose, rng_state, sensors


def config_value(value):
    if isinstance(value, functools.partial):
        return {'partial': config_value(value.func), 'args': value.args, 'kwargs': value.keywords}
    if callable(value):
        return f"{getattr(value, '__module__', type(value).__module__)}.{getattr(value, '__qualname__', type(value).__qualname__)}"
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'未适配配置类型：{type(value)}')


def equal(a, b, atol=1e-10):
    if isinstance(a, np.ndarray):
        if a.dtype.kind in 'biu':
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_allclose(a, b, rtol=0, atol=atol)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), (a.keys(), b.keys())
        for key in a:
            equal(a[key], b[key], atol)
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y, atol)
    else:
        assert a == b, (a, b)


def observation(task):
    data = task.env.current_data
    # 所有 body（包含目标、容器、机器人），旋转使用矩阵避免四元数正负二义性。
    return dict(xpos=data.xpos.copy(), xmat=data.xmat.copy(), cvel=data.cvel.copy(),
                qpos=data.qpos.copy(), qvel=data.qvel.copy(), info=deepcopy(task.get_info()))


def hold_trajectory(task):
    states = []
    for _ in range(20):
        task.env.current_robot.compute_control()
        for _ in range(5):  # 20 ms control / 4 ms integration
            mujoco.mj_step(task.env.current_model, task.env.current_data)
        states.append(integration_state(task.env.current_model, task.env.current_data))
    return np.stack(states)


def verify(task, provenance, path, mode="minimal"):
    env = task.env
    snapshot = EpisodeSnapshot.capture(task, provenance)
    snapshot.save(path)
    snapshot = EpisodeSnapshot.load(path, provenance)
    snapshot.restore(task, provenance)
    direct = observation(task)
    # 观察也可能更新成功判定缓存，保存其结果以检验调用顺序。
    after_direct = EpisodeSnapshot.capture(task, provenance)
    max_error = 0.0
    repeats = 1 if mode == "minimal" else 10
    for repeat in range(repeats):
        env.current_data.qvel[:] += 0.01
        env.current_data.ctrl[:] += 0.01
        env.current_data.qfrc_applied[:] += 0.01
        random.random()
        np.random.random()
        for controller in env.current_robot.controllers.values():
            controller.set_to_stationary()
        for controller in env.current_robot.controllers.values():
            if hasattr(controller, '_target'):
                controller._target = controller._target + 0.1
            controller._stationary = not controller._stationary
        for name, saved in snapshot.runtime['sensors'].items():
            for key in saved['state']:
                setattr(sensors(task)[name], key, None)
        if 'torch' in sys.modules:
            sys.modules['torch'].rand(1)
        env.current_data.mocap_pos[:] += 0.1
        env.current_data.xfrc_applied[:] += 0.1
        env.current_data.eq_active[:] = 0
        task.episode_step_count += 1
        task._supported_rel_poses.clear()
        snapshot.restore(task, provenance)
        actual = EpisodeSnapshot.capture(task, provenance)
        equal(snapshot.state, actual.state)
        equal(snapshot.runtime, actual.runtime)
        equal(snapshot.runtime['rng'], rng_state(), atol=0)
        max_error = max(max_error, float(np.max(np.abs(snapshot.state - actual.state))))
        equal(direct, observation(task), atol=1e-8)
    snapshot.restore(task)
    pose = env.current_robot.robot_view.base.pose.copy()
    pose[0, 3] += 0.1
    override_base_pose(task, pose)
    observation(task)
    snapshot.restore(task)
    equal(direct, observation(task), atol=1e-8)
    equal(after_direct.runtime, EpisodeSnapshot.capture(task, provenance).runtime)
    if mode == 'minimal':
        assert not any(name.startswith('curobo') for name in sys.modules), '意外导入 CuRobo'
        snapshot.restore(task)
        return dict(protocol='minimal', restore_repeats=1, max_state_error=max_error,
                    a_restore_a=True, b_restore_a=True, order_independence=True,
                    snapshot_sha256=digest(path), model_sha256=snapshot.model_sha256,
                    policy='disabled', control_steps=0, trajectory_check='not_run')
    try:
        with snapshot.restored(task):
            env.current_data.qvel[:] += 0.2
            raise RuntimeError('P0 注入异常')
    except RuntimeError as exc:
        assert str(exc) == 'P0 注入异常'
    equal(snapshot.state, integration_state(env.current_model, env.current_data))
    snapshot.restore(task)
    first = hold_trajectory(task)
    snapshot.restore(task)
    second = hold_trajectory(task)
    equal(first, second, atol=1e-8)
    timestep = env.current_model.opt.timestep
    try:
        env.current_model.opt.timestep = timestep * 2
        try:
            snapshot.restore(task)
        except ValueError as exc:
            assert '模型' in str(exc)
        else:
            raise AssertionError('未拒绝错误模型')
    finally:
        env.current_model.opt.timestep = timestep
    snapshot.restore(task)
    try:
        snapshot.restore(task, {'wrong': 'provenance'})
    except ValueError:
        pass
    else:
        raise AssertionError('未拒绝错误输入')
    assert not any(name.startswith('curobo') for name in sys.modules), '意外导入 CuRobo'
    return dict(restore_repeats=10, max_state_error=max_error,
                trajectory_max_error=float(np.max(np.abs(first - second))),
                order_independence=True, exception_restore=True, wrong_model_rejected=True,
                snapshot_sha256=digest(path), model_sha256=snapshot.model_sha256,
                policy='disabled', control_steps=20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['minimal', 'full'], default='minimal', help='默认 Minimal P0；完整旧验收仅按需运行')
    parser.add_argument('--limit', type=int, default=10, help='工程调试上限；少于 10 不判定 P0 完成')
    args = parser.parse_args()
    root = args.output
    data_audit = audit(root)
    atomic_json(root / 'data_audit.json', data_audit)
    config = json.loads((root / 'config.json').read_text())
    origin = json.loads((root / 'provenance.json').read_text())
    assert digest(origin['source']) == config['source_sha256']
    episodes = json.loads((root / 'pilot/benchmark.json').read_text())
    rows = [json.loads(line) for line in (root / 'pilot/manifest.jsonl').read_text().splitlines()]
    from molmo_spaces.molmo_spaces_constants import DATA_TYPE_TO_SOURCE_TO_VERSION
    protocol = dict(name=args.mode, restore_repeats=1 if args.mode == 'minimal' else 10, control_steps=0 if args.mode == 'minimal' else 20, repair_robot_base_pose_if_colliding=False)
    common = dict(validation_protocol=protocol, asset_versions=DATA_TYPE_TO_SOURCE_TO_VERSION, mujoco_version=mujoco.__version__, asset_metadata_sha256=origin['annotation_sha256'], scene_metadata_sha256=origin['scene_metadata_sha256'], source_sha256=config['source_sha256'], config_sha256=digest(root / 'config.json'),
                  manifest_sha256=digest(root / 'manifest.jsonl'), pilot_sha256=digest(root / 'pilot/benchmark.json'),
                  implementation_sha256={str(p.relative_to(Path(__file__).parents[3])): digest(p) for p in [*Path(__file__).parent.glob('*.py'), Path(__file__).parents[2] / 'configs/policy_configs.py', Path(__file__).parents[3] / 'scripts/evaluation/run_last_mile_p0.sh']})
    result_dir = root / f'validation_{args.mode}'
    result_dir.mkdir(parents=True, exist_ok=True)
    run_path = result_dir / 'inputs.json'
    if run_path.exists() and json.loads(run_path.read_text()) != common:
        raise ValueError('验证输入或实现改变，请使用新的 validation 目录保留旧结果')
    atomic_json(run_path, common)
    results = []
    for index, (episode, row) in enumerate(zip(episodes, rows)):
        if index >= args.limit:
            break
        result_path = result_dir / f'{index:03d}.json'
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if previous['status'] == 'passed':
                snapshot_path = result_dir / f'{index:03d}_snapshot.npz'
                if digest(snapshot_path) != previous['snapshot_sha256']:
                    raise ValueError('已完成快照的哈希不匹配')
            results.append(previous)
            continue
        start = time.monotonic()
        task = sampler = None
        result = dict(subset_index=index, source_index=row['source_index'], house=row['house'], target=row['target'])
        try:
            seed = row['seed']
            random.seed(seed)
            np.random.seed(seed)
            if 'torch' in sys.modules:
                sys.modules['torch'].manual_seed(seed)
            exp = P0Config(seed=seed, output_dir=result_dir)
            exp.eval_runtime_params = EvalRuntimeParams()
            exp.eval_runtime_params.repair_robot_base_pose_if_colliding = False
            spec = EpisodeSpec.model_validate(deepcopy(episode))
            spec.seed = seed
            sampler = P0JsonEvalTaskSampler(exp, spec, seed)
            task = sampler.sample_task(house_index=spec.house_index)
            if task is None:
                raise RuntimeError('初始化未返回任务')
            atomic_json(result_dir / f'{index:03d}_config.json', exp.model_dump(mode='json', exclude={'eval_runtime_params'}, fallback=config_value, warnings=False))
            assert not exp.eval_runtime_params.repair_robot_base_pose_if_colliding
            result.update(verify(task, dict(common, episode_id=row['episode_id']), result_dir / f'{index:03d}_snapshot.npz', mode=args.mode))
            result['status'] = 'passed'
        except Exception as exc:
            result.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        finally:
            result['elapsed_sec'] = time.monotonic() - start
            atomic_json(result_path, result)
            results.append(result)
            print(json.dumps({k: result[k] for k in ('subset_index', 'house', 'status', 'elapsed_sec')}), flush=True)
            # 不关闭全局资源管理器；每条仅释放其环境。
            if task is not None:
                task.close()
            del task, sampler
            gc.collect()
    jsonl(result_dir / 'episodes.jsonl', results)
    assert digest(origin['source']) == config['source_sha256']
    summary = dict(protocol=args.mode, attempted=len(results), passed=sum(r['status'] == 'passed' for r in results),
                   complete=len(results) == 10 and all(r['status'] == 'passed' for r in results),
                   failures=[r for r in results if r['status'] != 'passed'], formal_simulation='not_run')
    atomic_json(result_dir / 'summary.json', summary)
    if summary['complete']:
        atomic_json(result_dir / 'COMPLETE.json', {'inputs_sha256': digest(run_path), 'episodes_sha256': digest(result_dir / 'episodes.jsonl')})
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
