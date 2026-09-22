"""Train the staged RBY1 high-level policy with Stable-Baselines3 SAC."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from molmo_spaces.rl.action_adapter import RBY1ActionAdapter
from molmo_spaces.rl.env import RBY1RLEnv
from molmo_spaces.rl.observation import OBSERVATION_VERSION
from molmo_spaces.rl.reward import DEFAULT_REWARD_CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--house-indices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--total-timesteps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--random-baseline-episodes", type=int, default=10)
    parser.add_argument("--skip-env-check", action="store_true")
    return parser.parse_args()


def evaluate_random_policy(env: RBY1RLEnv, episodes: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for episode in range(episodes):
        _obs, _info = env.reset(seed=seed + episode)
        done = False
        episode_return = 0.0
        macro_steps = 0
        success = False
        while not done:
            action = rng.uniform(-1.0, 1.0, size=RBY1ActionAdapter.action_dim).astype(np.float32)
            _obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            macro_steps += 1
            success = bool(info.get("benchmark_success", False))
            done = terminated or truncated
        rows.append(
            {
                "episode": episode,
                "return": episode_return,
                "macro_steps": macro_steps,
                "success": int(success),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.env_checker import check_env
    except ImportError as exc:
        raise SystemExit("Install the RL extra first: uv sync --extra rl --extra curobo") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = RBY1RLEnv(house_indices=args.house_indices)
    try:
        if not args.skip_env_check:
            check_env(env, warn=True)
        baseline = evaluate_random_policy(env, args.random_baseline_episodes, args.seed + 10_000)
        write_csv(args.output_dir / "random_baseline.csv", baseline)

        class EpisodeCSVCallback(BaseCallback):
            def __init__(self, path: Path):
                super().__init__()
                self.path = path
                self.rows = []

            def _on_step(self) -> bool:
                for info in self.locals.get("infos", []):
                    episode = info.get("episode")
                    if episode is not None:
                        self.rows.append(
                            {
                                "timesteps": self.num_timesteps,
                                "return": episode["r"],
                                "length": episode["l"],
                                "success": int(info.get("benchmark_success", False)),
                            }
                        )
                        write_csv(self.path, self.rows)
                return True

        model = SAC(
            "MultiInputPolicy",
            env,
            seed=args.seed,
            device=args.device,
            gamma=0.95,
            learning_rate=3e-4,
            buffer_size=max(10_000, args.total_timesteps),
            learning_starts=min(20, max(1, args.total_timesteps // 5)),
            batch_size=min(64, max(2, args.total_timesteps)),
            tensorboard_log=str(args.output_dir / "tensorboard"),
            verbose=1,
        )
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=EpisodeCSVCallback(args.output_dir / "training_episodes.csv"),
        )
        model.save(args.output_dir / "model")
        metadata = {
            "observation_version": OBSERVATION_VERSION,
            "action_dim": RBY1ActionAdapter.action_dim,
            "max_candidates": env.policy_config.rl_max_grasp_candidates,
            "position_scale_m": env.policy_config.rl_position_scale_m,
            "max_base_translation_m": env.policy_config.rl_max_base_translation_m,
            "max_base_yaw_delta_rad": env.policy_config.rl_max_base_yaw_delta_rad,
            "reward_version": 1,
            "reward_config": DEFAULT_REWARD_CONFIG,
            "scene_dataset": "procthor-10k",
            "data_split": "train",
            "house_indices": args.house_indices,
            "seed": args.seed,
            "total_timesteps": args.total_timesteps,
        }
        (args.output_dir / "rl_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True)
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
