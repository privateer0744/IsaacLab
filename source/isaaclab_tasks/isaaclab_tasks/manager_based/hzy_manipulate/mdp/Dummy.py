import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObject
import torch
from isaaclab.utils.math import quat_apply
import math

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

def quat_conjugate(q):
    # q shape (...,4)
    w = q[..., 0:1]
    xyz = -q[..., 1:]
    return torch.cat([w, xyz], dim=-1)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return torch.stack((w, x, y, z), dim=-1)


# push,grasp,pick,load off,withdraw
def update_phase(env: ManagerBasedRLEnv):
    return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)


def get_min_inscribed_radius(env: ManagerBasedRLEnv):
    # --- 1. 基础数据与参数 ---
    bottle_h = 9 #单位是厘米
    bottle_r = 2.5
    p_b = env.bottles_pos_w    # [N, M, 3]
    #print("bottles pos shape: ",p_b.shape)
    q_b = env.bottles_quat_w   # [N, M, 4]
    # 物体局部Z轴
    v = torch.tensor([[0.,0.,1.]], device="cuda")


    #print("p_b: ",p_b)
    #print("q_b: ",q_b)
    target_xy = env.target_pos_w[:,:,:2] # [N, 1, 2]
    #print("target pos shape: ",target_xy.shape)
    num_envs = env.num_envs
    num_bottles = p_b.shape[1]

    # --- 2. 瓶子轴向与基向量计算 ---
    z_unit = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(num_envs, num_bottles, 1)
    v_a = quat_apply(q_b, z_unit)
    print("v_a : ",v_a) 

    ref_vec = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(num_envs, num_bottles, 1)
    is_x_parallel = torch.abs(v_a[:, :, 0]) > 0.99
    ref_vec[is_x_parallel] = torch.tensor([0.0, 1.0, 0.0], device=env.device)
    print("ref_vec : ",ref_vec) 

    u = torch.cross(ref_vec, v_a, dim=-1)
    print("u : ",u)
    u = u / torch.norm(u, dim=-1, keepdim=True)
    v = torch.cross(v_a, u, dim=-1)
    #v = v / torch.norm(v, dim=-1, keepdim=True)
    print("u : ",u)
    print("v : ",v)

    # --- 3. 轴线段最短距离 (情况一) ---
    dx_base = p_b[:, :, 0] - target_xy[:, :, 0]
    dy_base = p_b[:, :, 1] - target_xy[:, :, 1]
    denom = v_a[:, :, 0]**2 + v_a[:, :, 1]**2
    #print("denom shape: ",denom.shape)
    
    t1 = torch.where(denom > 1e-6, 
                     -(dx_base * v_a[:, :, 0] + dy_base * v_a[:, :, 1]) / denom, 
                     torch.zeros_like(denom))
    #print("t1: ",t1)


    is_at_bottom = t1 <= 0.0
    is_at_top = t1 >= bottle_h
    is_at_caps = is_at_bottom | is_at_top

    # --- 4. 处理情况二：投影椭圆的最短距离 (参数化采样) ---
    num_samples = 16
    theta = torch.linspace(0, 2 * math.pi, num_samples, device=env.device)
    
    # 维度修正：view成 [1, 1, 16]，确保与 [N, M, 1] 广播出 [N, M, 16]
    cos_theta = torch.cos(theta).view(1, 1, num_samples) 
    sin_theta = torch.sin(theta).view(1, 1, num_samples)

    t_end = torch.where(is_at_top, torch.tensor(bottle_h, device=env.device), torch.tensor(0.0, device=env.device))
    p_cap_center = p_b + t_end.unsqueeze(-1) * v_a 
    #print("t_end shape: ",t_end.shape)
    print("p_cap_center : ",p_cap_center)

    # 这里的维度匹配逻辑：[N, M, 1] + [1, 1, 16] * [N, M, 1] -> [N, M, 16]
    samples_x = p_cap_center[:, :, 0:1] + bottle_r * (cos_theta * u[:, :, 0:1] + sin_theta * v[:, :, 0:1])
    samples_y = p_cap_center[:, :, 1:2] + bottle_r * (cos_theta * u[:, :, 1:2] + sin_theta * v[:, :, 1:2])
    print("samples_x :", samples_x)
    print("samples_y :", samples_y)

    # stack 之后维度变成 [N, M, 16, 2]
    samples_xy = torch.stack([samples_x, samples_y], dim=-1)

    # 这里的 target_xy.unsqueeze(2) 维度是 [N, 1, 1, 2]，完美匹配 [N, M, 16, 2]
    dist_to_samples = torch.norm(samples_xy - target_xy.unsqueeze(2), dim=-1)
    min_dist_cap, _ = torch.min(dist_to_samples, dim=-1) 

    # --- 5. 整合结果 ---
    p_closest_side = p_b + torch.clamp(t1, 0, bottle_h).unsqueeze(-1) * v_a
    dist_side = torch.norm(p_closest_side[:, :, :2] - target_xy, dim=-1) - bottle_r

    final_dist = torch.where(is_at_caps, min_dist_cap, dist_side)



    #print(f"bottle pos: {p_b[0,0]}, target pos: {target_xy[0,0]}")
    #print(f"v_a: {v_a[0,0]}, denom: {denom[0,0]}, t1: {t1[0,0]}")
    #print(f"is_at_caps: {is_at_caps[0,0]}, dist_side: {dist_side[0,0]}, min_dist_cap: {min_dist_cap[0,0]}")
    print(f"final_dist: ",final_dist)
    
    return torch.clamp(final_dist, min=0.0)


def get_obj_max_momentum(env: ManagerBasedRLEnv):
    v_b = env.bottles_vel_w
    pre_v_b = env.pre_bottles_vel_w
    omeg_b = env.bottles_omeg_w
    pre_omeg_b = env.pre_bottles_omeg_w
    grad_v = v_b - pre_v_b
        

    return torch.tensor(0.0),torch.tensor(0.0)

def get_contactforce(env: ManagerBasedRLEnv):
    return torch.tensor(0.0),torch.tensor(0.0)


def get_stuck_duration(env: ManagerBasedRLEnv):
    return torch.tensor(0.0)


def get_wander_duration(env: ManagerBasedRLEnv):
    return torch.tensor(0.0)