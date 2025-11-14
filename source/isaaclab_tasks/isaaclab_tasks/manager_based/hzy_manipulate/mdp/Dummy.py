import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
import torch

def dummy_reward(env: ManagerBasedRLEnv):
    return torch.zeros(env.num_envs, device=env.device)

def dummy_obs(env: ManagerBasedRLEnv):
    return torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)

def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    return env.episode_length_buf >= env.max_episode_length