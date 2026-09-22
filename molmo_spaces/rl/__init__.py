"""High-level reinforcement-learning support for MolmoSpaces.

Modules are intentionally imported lazily: policy configuration imports the
learned policy, while the training environment imports evaluation config.
"""

__all__ = ["RBY1RLEnv"]


def __getattr__(name):
    if name == "RBY1RLEnv":
        from molmo_spaces.rl.env import RBY1RLEnv

        return RBY1RLEnv
    raise AttributeError(name)
