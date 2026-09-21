"""P0 严格加载适配器；默认 JSON sampler 不变。"""
import random
import sys

import mujoco
import numpy as np

from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler


class P0JsonEvalTaskSampler(JsonEvalTaskSampler):
    def __init__(self, exp_config, episode_spec, seed):
        runtime = exp_config.eval_runtime_params
        if runtime is None or runtime.repair_robot_base_pose_if_colliding:
            raise ValueError('P0 必须显式关闭底盘自动 repair')
        if runtime.robot_override_fn:
            raise ValueError('P0 不支持跨机器人替换')
        spec = episode_spec.model_copy(deep=True)
        if (spec.scene_dataset, spec.data_split, spec.robot.robot_name) != ('procthor-10k', 'val', 'rby1m'):
            raise ValueError('P0 仅支持 RBY1 val 数据')
        spec.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        if 'torch' in sys.modules:
            sys.modules['torch'].manual_seed(seed)
        super().__init__(exp_config, spec)
        self.seed_task_sampling(seed)

    def randomize_scene(self, env, robot_view):
        super().randomize_scene(env, robot_view)
        # 通用 sampler 允许略过缺失 body，且 1 mm 内不重写；P0 不能默默接受。
        for name, pose in self.episode_spec.scene_modifications.object_poses.items():
            body = create_mlspaces_body(env.current_data, name)
            body.position = pose[:3]
            body.quat = pose[3:7]
        mujoco.mj_forward(env.current_model, env.current_data)
