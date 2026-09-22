"""P0 单环境状态适配器。仅接受已审计控制器；不 pickle 仿真对象。"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import functools
import json
import os
from pathlib import Path
import random
import sys

import mujoco
import numpy as np

STATE = mujoco.mjtState.mjSTATE_INTEGRATION
TASK_FIELDS = ('_cumulative_reward', '_num_steps_taken', 'episode_step_count',
               '_policy_done', '_done_action_received', '_supported_rel_poses',
               'last_action', 'action_cache', 'observation_cache', 'reward_cache',
               'terminal_cache', 'truncated_cache', 'success_cache')
CONTROLLERS = {'JointPosController', 'JointRelPosController', 'JointVelController',
               'DiffDriveBasePoseController', 'TorsoHeightJointPosController'}
CONTROL_FIELDS = ('_target', '_stationary', 'desired_joint_positions', '_target_height')
CONTROL_CONSTANTS = {'robot_move_group', 'ctrl_dim', 'ctrl_range', 'robot_config',
                     'wheel_base', 'wheel_radius', 'max_height'}


def task_config_fingerprint(task):
    config = getattr(task, 'config', None)
    if config is None:
        return None
    def fallback(value):
        if isinstance(value, functools.partial):
            return dict(partial=fallback(value.func), args=value.args, kwargs=value.keywords)
        if callable(value):
            return f"{getattr(value, '__module__', type(value).__module__)}.{getattr(value, '__qualname__', type(value).__qualname__)}"
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f'未适配配置类型：{type(value)}')
    value = config.model_dump(mode='json', exclude={'eval_runtime_params'}, fallback=fallback, warnings=False)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def model_fingerprint(model):
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return hashlib.sha256(buffer).hexdigest()


def integration_state(model, data):
    state = np.empty(mujoco.mj_stateSize(model, STATE))
    mujoco.mj_getState(model, data, state, STATE)
    if not np.isfinite(state).all():
        raise ValueError('MuJoCo 积分状态包含 NaN/Inf')
    return state


def encode(value, arrays):
    if isinstance(value, np.ndarray):
        key = f'a{len(arrays)}'
        arrays[key] = value.copy()
        return {'array': key}
    if isinstance(value, np.generic):
        return encode(value.item(), arrays)
    if isinstance(value, dict):
        return {'dict': [[encode(k, arrays), encode(v, arrays)] for k, v in value.items()]}
    if isinstance(value, set):
        return {'set': [encode(v, arrays) for v in sorted(value)]}
    if isinstance(value, (tuple, list)):
        return {type(value).__name__: [encode(v, arrays) for v in value]}
    if isinstance(value, bytes):
        return {'bytes': value.hex()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'未适配状态类型：{type(value)}')


def decode(value, arrays):
    if not isinstance(value, dict):
        return value
    if 'array' in value:
        return arrays[value['array']].copy()
    if 'dict' in value:
        return {decode(k, arrays): decode(v, arrays) for k, v in value['dict']}
    if 'set' in value:
        return {decode(v, arrays) for v in value['set']}
    if 'tuple' in value:
        return tuple(decode(v, arrays) for v in value['tuple'])
    if 'list' in value:
        return [decode(v, arrays) for v in value['list']]
    return bytes.fromhex(value['bytes'])


def fields(obj, names):
    return {n: deepcopy(getattr(obj, n)) for n in names if hasattr(obj, n)}


def sensors(task):
    suite = getattr(task, '_sensor_suite', None)
    return suite.sensors if suite is not None else {}


SENSOR_FIELDS = {
    'CameraParameterSensor': (), 'CameraSensor': (), 'RobotJointPositionSensor': (),
    'RobotJointVelocitySensor': (), 'RobotBasePoseSensor': (), 'EnvStateSensor': (),
    'TaskInfoSensor': (), 'LastActionSensor': (), 'LastCommandedJointPosSensor': (),
    'LastCommandedRelativeJointPosSensor': ('_prev_jp',),
    'LastCommandedEETwistSensor': ('_prev_poses', '_tracked_keys'),
    'LastCommandedEEPoseSensor': (), 'ObjectImagePointsSensor': (),
    'ObjectStartPoseSensor': ('_initial_pose',),
    'GraspStateSensor': ('_object_geoms', '_gripper_geoms'),
    'RBY1GraspStateSensor': ('obj_name',),
}


def space_rng(space):
    children = getattr(space, 'spaces', {})
    if isinstance(children, tuple):
        children = dict(enumerate(children))
    return dict(state=deepcopy(space._np_random.bit_generator.state) if getattr(space, '_np_random', None) is not None else None,
                children={k: space_rng(v) for k, v in children.items()})


def set_space_rng(space, state):
    if state['state'] is None:
        space._np_random = None
    else:
        space.np_random.bit_generator.state = deepcopy(state['state'])
    for k, value in state['children'].items():
        set_space_rng(space.spaces[k], value)


def sensor_state(sensor):
    cls = type(sensor).__name__
    if cls not in SENSOR_FIELDS:
        raise ValueError(f'未适配传感器：{cls}')
    return dict(state=fields(sensor, SENSOR_FIELDS[cls]), space_rng=space_rng(sensor.observation_space), class_name=cls)


def rng_state():
    state = {'python': random.getstate(), 'numpy': np.random.get_state()}
    torch = sys.modules.get('torch')
    if torch is not None:
        state['torch_cpu'] = torch.get_rng_state().numpy().copy()
        if torch.cuda.is_initialized():
            state['torch_cuda'] = [s.cpu().numpy().copy() for s in torch.cuda.get_rng_state_all()]
    return state


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    if 'torch_cpu' in state:
        torch = sys.modules['torch']
        torch.set_rng_state(torch.from_numpy(state['torch_cpu'].copy()))
        if 'torch_cuda' in state:
            torch.cuda.set_rng_state_all([torch.from_numpy(s.copy()) for s in state['torch_cuda']])


@dataclass
class EpisodeSnapshot:
    model_sha256: str
    state: np.ndarray
    runtime: dict
    provenance: dict

    @classmethod
    def capture(cls, task, provenance):
        env = task.env
        if len(env.mj_datas) != 1:
            raise ValueError('P0 仅支持单环境')
        if getattr(task, '_registered_policy', None) is not None:
            raise ValueError('策略已启用但没有状态适配器')
        controls = {}
        for name, controller in env.current_robot.controllers.items():
            unknown = set(vars(controller)) - set(CONTROL_FIELDS) - CONTROL_CONSTANTS
            if type(controller).__name__ not in CONTROLLERS or unknown:
                raise ValueError(f'未适配控制器：{type(controller).__name__} {unknown}')
            controls[name] = {'class': type(controller).__name__, 'state': fields(controller, CONTROL_FIELDS)}
        runtime = dict(task_config_sha256=task_config_fingerprint(task), task=fields(task, TASK_FIELDS), controllers=controls,
                       robot=fields(env.current_robot, ('_last_unnoised_cmd_joint_pos',)),
                       sensors={name: sensor_state(s) for name, s in sensors(task).items()},
                       rng=rng_state(), policy='disabled')
        encode(runtime, {})  # Early rejection, rather than fail when saving.
        return cls(model_fingerprint(env.current_model), integration_state(env.current_model, env.current_data), runtime, deepcopy(provenance))

    def restore(self, task, provenance=None):
        env = task.env
        if provenance is not None and provenance != self.provenance:
            raise ValueError('快照输入/配置指纹不匹配')
        if task_config_fingerprint(task) != self.runtime['task_config_sha256']:
            raise ValueError('任务配置与快照不匹配')
        if model_fingerprint(env.current_model) != self.model_sha256:
            raise ValueError('快照模型指纹不匹配')
        if getattr(task, '_registered_policy', None) is not None:
            raise ValueError('策略状态未适配')
        if set(sensors(task)) != set(self.runtime['sensors']):
            raise ValueError('传感器集合不匹配')
        if set(env.current_robot.controllers) != set(self.runtime['controllers']):
            raise ValueError('控制器集合不匹配')
        for name, item in self.runtime['controllers'].items():
            controller = env.current_robot.controllers[name]
            unknown = set(vars(controller)) - set(CONTROL_FIELDS) - CONTROL_CONSTANTS
            if type(controller).__name__ != item['class'] or unknown:
                raise ValueError('控制器类型/状态未适配')
        for name, item in self.runtime['sensors'].items():
            if type(sensors(task)[name]).__name__ != item['class_name']:
                raise ValueError('传感器类型不匹配')
        torch = sys.modules.get('torch')
        if (torch is not None) != ('torch_cpu' in self.runtime['rng']):
            raise ValueError('RNG 集合改变，需重新捕获快照')
        if torch is not None and torch.cuda.is_initialized() != ('torch_cuda' in self.runtime['rng']):
            raise ValueError('CUDA RNG 集合改变，需重新捕获快照')
        mujoco.mj_setState(env.current_model, env.current_data, self.state, STATE)
        mujoco.mj_forward(env.current_model, env.current_data)
        # mj_forward may update the warmstart; preserve the captured integration fields.
        mujoco.mj_setState(env.current_model, env.current_data, self.state, STATE)
        for name in TASK_FIELDS:
            if name in self.runtime['task']:
                setattr(task, name, deepcopy(self.runtime['task'][name]))
            elif name in vars(task):
                delattr(task, name)
        for name, item in self.runtime['controllers'].items():
            controller = env.current_robot.controllers[name]
            if type(controller).__name__ != item['class']:
                raise ValueError('控制器类型不匹配')
            for key, value in item['state'].items():
                setattr(controller, key, deepcopy(value))
        for key, value in self.runtime['robot'].items():
            setattr(env.current_robot, key, deepcopy(value))
        for name, item in self.runtime['sensors'].items():
            sensor = sensors(task)[name]
            if type(sensor).__name__ != item['class_name']:
                raise ValueError('传感器类型不匹配')
            for key, value in item['state'].items():
                setattr(sensor, key, deepcopy(value))
            set_space_rng(sensor.observation_space, item['space_rng'])
        restore_rng(self.runtime['rng'])

    @contextmanager
    def restored(self, task):
        self.restore(task)
        try:
            yield
        finally:
            self.restore(task)

    def save(self, path):
        arrays = {'integration': self.state}
        meta = dict(schema_version=1, mujoco_version=mujoco.__version__, state_spec=int(STATE),
                    model_sha256=self.model_sha256, provenance=self.provenance,
                    runtime=encode(self.runtime, arrays))
        arrays['metadata'] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path) + '.tmp', 'wb') as f:
            np.savez_compressed(f, **arrays)
        os.replace(str(path) + '.tmp', path)

    @classmethod
    def load(cls, path, provenance):
        with np.load(path, allow_pickle=False) as arrays:
            meta = json.loads(arrays['metadata'].tobytes())
            if (meta['schema_version'] != 1 or meta['mujoco_version'] != mujoco.__version__
                    or meta['state_spec'] != int(STATE) or meta['provenance'] != provenance):
                raise ValueError('快照版本或输入指纹不匹配')
            return cls(meta['model_sha256'], arrays['integration'].copy(), decode(meta['runtime'], arrays), meta['provenance'])


def override_base_pose(task, pose):
    """显式底盘变换，不重置其他控制器、不执行稳定步。"""
    robot = task.env.current_robot
    robot.robot_view.base.pose = np.asarray(pose).copy()
    # Holo base ``noop_ctrl`` is read from a site pose.  Refresh kinematics before
    # asking the controller to hold the teleported pose, otherwise it targets the
    # stale pre-teleport site and drives back there on the first real step.
    mujoco.mj_forward(task.env.current_model, task.env.current_data)
    controller = robot.controllers['base']
    controller.set_to_stationary()
    controller.robot_move_group.ctrl = controller.compute_ctrl_inputs()
    cache = robot._last_unnoised_cmd_joint_pos
    if cache is not None and 'base' in cache:
        cache['base'] = controller.target_pos.copy()
    state = integration_state(task.env.current_model, task.env.current_data)
    mujoco.mj_forward(task.env.current_model, task.env.current_data)
    mujoco.mj_setState(task.env.current_model, task.env.current_data, state, STATE)
