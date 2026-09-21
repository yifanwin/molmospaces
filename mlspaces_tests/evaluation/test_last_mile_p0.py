"""无需资产下载的 P0 数据/快照回归测试。"""
import gzip
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'molmo_spaces/evaluation/last_mile/{name}.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


build = module('build')
snap = module('snapshot')


def dataset(tmp_path):
    assets = tmp_path / 'assets'
    scenes = assets / 'scenes/procthor-10k-val'
    scenes.mkdir(parents=True)
    meta = assets / 'objects/objathor_metadata/objects_metadata.json.gz'
    meta.parent.mkdir(parents=True)
    with gzip.open(meta, 'wt') as f:
        json.dump({'asset_mug': {'category': 'mug'}}, f)
    episodes = []
    for house in range(65):
        objects = {}
        for target in range(3):
            name = f'not_a_category_{target}'
            objects[name] = {'asset_id': 'asset_mug', 'category': 'Mug'}
            episodes.append(dict(house_index=house, scene_dataset='procthor-10k', data_split='val',
                                 seed=None, robot={'robot_name': 'rby1m'}, source={'episode_length': 10},
                                 task={'pickup_obj_name': name, 'place_receptacle_name': 'bowl',
                                       'task_cls': 'PickAndPlaceTask', 'succ_pos_threshold': float('inf')}))
        (scenes / f'val_{house}_metadata.json').write_text(json.dumps({'objects': objects}))
    episodes.append(episodes[0].copy())
    source = tmp_path / 'source.json'
    source.write_text(json.dumps(episodes))
    return source, assets


def test_reproducible_subsets(tmp_path):
    source, assets = dataset(tmp_path)
    before = build.digest(source)
    a, b = tmp_path / 'a', tmp_path / 'b'
    pilot, formal = build.build(source, assets, a)
    build.build(source, assets, b)
    for file in ['manifest.jsonl', 'reserves.jsonl', 'pilot/benchmark.json', 'formal/benchmark.json']:
        assert build.digest(a / file) == build.digest(b / file)
    assert len(pilot) == 10 and len(formal) == 100
    assert len({r['house'] for r in pilot}) == 5
    assert len({r['house'] for r in formal}) >= 20
    assert not {r['house'] for r in pilot} & {r['house'] for r in formal}
    assert len({(r['house'], r['target']) for r in pilot + formal}) == 110
    assert all(r['category'] == 'mug' for r in pilot + formal)
    assert all(e['task']['succ_pos_threshold'] == float('inf') for e in json.loads((a / 'pilot/benchmark.json').read_text()))
    assert before == build.digest(source)


def test_split_rejected(tmp_path):
    source, assets = dataset(tmp_path)
    episodes = json.loads(source.read_text())
    episodes[0]['data_split'] = 'train'
    source.write_text(json.dumps(episodes))
    with pytest.raises(AssertionError):
        build.build(source, assets, tmp_path / 'out')


def tiny_task():
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="box"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    controller = type('JointPosController', (), {})()
    controller._target = np.array([0.4])
    controller._stationary = False
    robot = SimpleNamespace(controllers={'arm': controller}, _last_unnoised_cmd_joint_pos=None)
    env = SimpleNamespace(current_model=model, current_data=data, mj_datas=[data], current_robot=robot)
    return SimpleNamespace(env=env, _registered_policy=None, _sensor_suite=None,
                           episode_step_count=0, _supported_rel_poses={1: [np.eye(4)]})


def test_snapshot_roundtrip_and_exception(tmp_path):
    task = tiny_task()
    original = snap.EpisodeSnapshot.capture(task, {'source': 'test'})
    path = tmp_path / 'state.npz'
    original.save(path)
    loaded = snap.EpisodeSnapshot.load(path, {'source': 'test'})
    for _ in range(10):
        task.env.current_data.qvel[:] = 5
        task.episode_step_count = 100
        task.env.current_robot.controllers['arm']._target[:] = 20
        random.random()
        np.random.random()
        loaded.restore(task)
        np.testing.assert_array_equal(original.state, snap.integration_state(task.env.current_model, task.env.current_data))
        assert task.episode_step_count == 0
        assert task.env.current_robot.controllers['arm']._target[0] == .4
    with pytest.raises(RuntimeError):
        with loaded.restored(task):
            task.env.current_data.qvel[:] = 2
            raise RuntimeError('injected')
    np.testing.assert_array_equal(original.state, snap.integration_state(task.env.current_model, task.env.current_data))
    with pytest.raises(ValueError):
        snap.EpisodeSnapshot.load(path, {'source': 'wrong'})
    task.env.current_model.opt.timestep *= 2
    with pytest.raises(ValueError, match='模型'):
        loaded.restore(task)


def test_unknown_policy_and_controller_fail_closed():
    task = tiny_task()
    task._registered_policy = object()
    with pytest.raises(ValueError, match='策略'):
        snap.EpisodeSnapshot.capture(task, {})
    task._registered_policy = None
    task.env.current_robot.controllers['arm'].new_history = []
    with pytest.raises(ValueError, match='控制器'):
        snap.EpisodeSnapshot.capture(task, {})


def test_sensor_rng_and_cache_restored(tmp_path):
    import gymnasium.spaces as spaces
    task = tiny_task()
    sensor = type('GraspStateSensor', (), {})()
    sensor._object_geoms = {1, 2}
    sensor._gripper_geoms = {'left': {3, 4}}
    sensor.observation_space = spaces.Dict({'value': spaces.Box(-1, 1, (2,))})
    sensor.observation_space.seed(42)
    task._sensor_suite = SimpleNamespace(sensors={'grasp': sensor})
    snapshot = snap.EpisodeSnapshot.capture(task, {})
    path = tmp_path / 'sensor.npz'
    snapshot.save(path)
    snapshot = snap.EpisodeSnapshot.load(path, {})
    expected = sensor.observation_space.sample()
    sensor._object_geoms = {99}
    sensor._gripper_geoms = None
    snapshot.restore(task)
    assert sensor._object_geoms == {1, 2}
    assert sensor._gripper_geoms == {'left': {3, 4}}
    np.testing.assert_array_equal(expected['value'], sensor.observation_space.sample()['value'])


def test_missing_metadata_is_excluded(tmp_path):
    source, assets = dataset(tmp_path)
    (assets / 'scenes/procthor-10k-val/val_0_metadata.json').unlink()
    output = tmp_path / 'out'
    pilot, formal = build.build(source, assets, output)
    assert not any(r['house'] == 0 for r in pilot + formal)
    excluded = [json.loads(s) for s in (output / 'excluded.jsonl').read_text().splitlines()]
    assert sum(r['reason'] == 'missing_category_metadata' for r in excluded) == 4


def test_integration_covers_controls_mocap_equality_and_force(tmp_path):
    task = tiny_task()
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <worldbody><body name="box"><joint name="j" type="slide"/><geom type="sphere" size=".1" mass="1"/></body>
      <body name="anchor" mocap="true"><geom type="sphere" size=".01" contype="0" conaffinity="0"/></body></worldbody>
      <equality><weld body1="box" body2="anchor" active="false"/></equality>
      <actuator><motor joint="j"/></actuator></mujoco>''')
    data = mujoco.MjData(model)
    task.env.current_model = model
    task.env.current_data = data
    task.env.mj_datas = [data]
    data.ctrl[:] = .2
    data.mocap_pos[:] = .1
    data.qfrc_applied[:] = .3
    data.xfrc_applied[:] = .4
    data.qacc_warmstart[:] = .5
    mujoco.mj_forward(model, data)
    original = snap.EpisodeSnapshot.capture(task, {})
    data.ctrl[:] = 8
    data.mocap_pos[:] = 9
    data.eq_active[:] = 1
    data.qfrc_applied[:] = 10
    data.xfrc_applied[:] = 11
    original.restore(task)
    np.testing.assert_array_equal(original.state, snap.integration_state(model, data))


def test_same_control_trajectory_is_repeatable():
    task = tiny_task()
    snapshot = snap.EpisodeSnapshot.capture(task, {})
    trajectories = []
    for _ in range(2):
        snapshot.restore(task)
        states = []
        for _ in range(20):
            mujoco.mj_step(task.env.current_model, task.env.current_data)
            states.append(snap.integration_state(task.env.current_model, task.env.current_data))
        trajectories.append(states)
    np.testing.assert_allclose(*trajectories, rtol=0, atol=1e-8)


def test_asset_metadata_drift_rejected(tmp_path):
    source, assets = dataset(tmp_path)
    output = tmp_path / 'out'
    build.build(source, assets, output)
    provenance = json.loads((output / 'provenance.json').read_text())
    build.verify_asset_metadata(provenance)
    (assets / 'scenes/procthor-10k-val/val_0_metadata.json').write_text('{}')
    with pytest.raises(ValueError, match='元数据'):
        build.verify_asset_metadata(provenance)


def test_restore_preflight_does_not_mutate_physics():
    task = tiny_task()
    snapshot = snap.EpisodeSnapshot.capture(task, {})
    task.env.current_data.qvel[:] = 9
    before = snap.integration_state(task.env.current_model, task.env.current_data)
    task.env.current_robot.controllers['arm'].unknown_cache = [1]
    with pytest.raises(ValueError, match='控制器'):
        snapshot.restore(task)
    np.testing.assert_array_equal(before, snap.integration_state(task.env.current_model, task.env.current_data))


def test_same_model_different_task_config_rejected():
    task = tiny_task()
    class Config:
        seed = 1
        def model_dump(self, **kwargs):
            return {'seed': self.seed, 'target': 'mug'}
    task.config = Config()
    snapshot = snap.EpisodeSnapshot.capture(task, {})
    task.config.seed = 2
    with pytest.raises(ValueError, match='任务配置'):
        snapshot.restore(task)


def test_existing_freeze_cannot_be_silently_reselected(tmp_path):
    source, assets = dataset(tmp_path)
    output = tmp_path / 'out'
    build.build(source, assets, output)
    before = build.digest(output / 'manifest.jsonl')
    build.build(source, assets, output)
    assert before == build.digest(output / 'manifest.jsonl')
    path = assets / 'scenes/procthor-10k-val/val_0_metadata.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='元数据'):
        build.build(source, assets, output)
    assert before == build.digest(output / 'manifest.jsonl')


def test_nonfinite_integration_is_not_a_successful_restore():
    task = tiny_task()
    task.env.current_data.qvel[:] = np.nan
    with pytest.raises(ValueError, match='NaN'):
        snap.EpisodeSnapshot.capture(task, {})
