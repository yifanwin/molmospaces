"""P0 无策略配置：不继承会初始化 CuRobo 的数据生成配置。"""
from pathlib import Path
from pydantic import Field
from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.robot_configs import RBY1MConfig
from molmo_spaces.configs.policy_configs import DummyPolicyConfig
from molmo_spaces.configs.task_configs import PickAndPlaceTaskConfig
from molmo_spaces.configs.task_sampler_configs import BaseMujocoTaskSamplerConfig


class P0Config(MlSpacesExpConfig):
    num_envs: int = 1
    task_type: str = 'pick_and_place'
    use_passive_viewer: bool = False
    viewer_cam_dict: dict = Field(default_factory=dict)
    policy_dt_ms: float = 100.0
    ctrl_dt_ms: float = 20.0
    sim_dt_ms: float = 4.0
    task_horizon: int = 600
    scene_dataset: str = 'procthor-10k'
    data_split: str = 'val'
    output_dir: Path = Path('eval_output/last_mile/p0_20260921')
    robot_config: RBY1MConfig = Field(default_factory=RBY1MConfig)
    policy_config: DummyPolicyConfig = Field(default_factory=DummyPolicyConfig)
    task_config: PickAndPlaceTaskConfig = Field(default_factory=PickAndPlaceTaskConfig)
    task_sampler_config: BaseMujocoTaskSamplerConfig = Field(default_factory=lambda: BaseMujocoTaskSamplerConfig(house_inds=[0], samples_per_house=1, task_batch_size=1, max_tasks=1))
    use_wandb: bool = False
    datagen_profiler: bool = False
    filter_for_successful_trajectories: bool = False

    @property
    def tag(self):
        return 'last_mile_p0'
