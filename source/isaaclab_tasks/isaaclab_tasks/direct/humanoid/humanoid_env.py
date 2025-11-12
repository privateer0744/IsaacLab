# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_assets import HUMANOID_CFG, current_file_path, current_dir

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

import torch
@configclass
class HumanoidEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    action_space = 21
    observation_space = 75
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = HUMANOID_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    joint_gears: list = [
        67.5000,  # lower_waist
        67.5000,  # lower_waist
        67.5000,  # right_upper_arm
        67.5000,  # right_upper_arm
        67.5000,  # left_upper_arm
        67.5000,  # left_upper_arm
        67.5000,  # pelvis
        45.0000,  # right_lower_arm
        45.0000,  # left_lower_arm
        45.0000,  # right_thigh: x
        135.0000,  # right_thigh: y
        45.0000,  # right_thigh: z
        45.0000,  # left_thigh: x
        135.0000,  # left_thigh: y
        45.0000,  # left_thigh: z
        90.0000,  # right_knee
        90.0000,  # left_knee
        22.5,  # right_foot
        22.5,  # right_foot
        22.5,  # left_foot
        22.5,  # left_foot
    ]

    heading_weight: float = 1.0
    up_weight: float = 0.1

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0
    dof_vel_scale: float = 0.5

    death_cost: float = -2.0
    termination_height: float = 0.6

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01






class HumanoidEnv(LocomotionEnv):
    cfg: HumanoidEnvCfg

    def __init__(self, cfg: HumanoidEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

# acquire properties
        print("======================= usd file current_file_path:", current_file_path)
        print("======================= usd file current_dir:", current_dir)
        torch.set_printoptions(precision=4)
        self._body_ids, self._body_names = self.robot.find_bodies(self.robot.body_names)
        print("***********************body_ids is", self._body_ids)
        print("***********************body_names is", self._body_names)
        print("******************joint_dof_idx is:", self._joint_dof_idx)
        _ , self.dof_names = self.robot.find_joints(".*")
        self.ee_names = ['left_foot','right_foot','left_hand','right_hand']
        #self.shoulder_pitch_ids = [self._joint_dof_idx[self.dof_names.index(name)] for name in ['left_upper_arm','right_upper_arm']]
        self.knee_ids = [self._joint_dof_idx[self.dof_names.index(name)] for name in ['left_shin', 'right_shin']]
        self.ee_body_ids = [self._body_ids[self._body_names.index(name)] for name in self.ee_names]
        self.pelvis_body_ids = self._body_ids[self._body_names.index('pelvis')]
        #print("*******************shoulder_joint_ids", self.shoulder_pitch_ids)
        print("*******************knee_joint_ids", self.knee_ids)
        print("*******************ee_ids", self.ee_body_ids)
        print("*******************base_ids", self.pelvis_body_ids)


    def _get_rewards(self):
        rwd_plus = super()._get_rewards()
        
        #rwd_plus += torch.sum(self.dof_pos_scaled[:,torch.tensor(self.shoulder_pitch_ids, device=self.device)]**2, dim = -1) * 0.5
        mask = (self.dof_pos[:, torch.tensor(self.knee_ids, device=self.device)] > 0).all(dim=-1) #where cannot sum up bool matrix directly
        rwd_plus -= torch.where(mask > 0, torch.tensor(1), torch.tensor(-1))*1.0
        
        return rwd_plus
