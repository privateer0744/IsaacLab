import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObject
import torch

def dummy_reward(env: ManagerBasedRLEnv):
    





    return torch.zeros(env.num_envs, device=env.device)

def dummy_obs(env: ManagerBasedRLEnv):
    return torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)

def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    return env.episode_length_buf >= env.max_episode_length

def get_pos(objects: RigidObject) -> tuple[torch.Tensor,torch.Tensor]:
    pos = objects.data.root_pos_w
    # 旋转 (四元数) [num_envs, 36, 4]
    quat = objects.data.root_quat_w
    #print("bottles' pos are:",pos,quat)
    return pos, quat