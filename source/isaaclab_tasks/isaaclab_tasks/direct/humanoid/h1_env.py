# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#  start up: isaaclab.bat -p scripts/reinforcement_learning/rl_games/train.py --task Isaac-H1-Direct-v0
# isaaclab.bat -p scripts/reinforcement_learning/rl_games/play.py --task Isaac-H1-Direct-v0 --num_envs 1
# ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task Isaac-H1-Direct-v0 --headless --num_envs 256

from __future__ import annotations
import torch

from isaaclab_assets import H1_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

from isaaclab_tasks.direct.humanoid.H1_remote_procss import Intermediate_Motion, quaternion_to_rotation_matrix, quaternion_conjugate, quaternion_multiply,Judge_contact
from isaaclab_tasks.direct.humanoid.exoskltn_intef import Communicating
from threading import Thread
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

@configclass
class H1EnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    action_space = 19
    observation_space = 69#97
    state_space = 0

    IntervalSim:float = 1 / 120
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=IntervalSim, render_interval=decimation)
    # for flat terrain
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
    # for rough terrains
    #terrain = TerrainImporterCfg(
    #    prim_path="/World/ground",
    #    terrain_type="generator",
    #    terrain_generator=ROUGH_TERRAINS_CFG,
    #    max_init_terrain_level=9,
    #    collision_group=-1,
    #    physics_material=sim_utils.RigidBodyMaterialCfg(
    #        friction_combine_mode="multiply",
    #        restitution_combine_mode="multiply",
    #        static_friction=1.0,
    #        dynamic_friction=1.0,
    #    ),
    #    visual_material=sim_utils.MdlFileCfg(
    #        mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
    #        project_uvw=True,
    #    ),
    #    debug_vis=False,
    #)
    # scene only one environment
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=4.0, replicate_physics=True)
    #scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1)
    # robot
    robot: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    joint_gears: list = [
    50.0,  # left_hip_yaw
    50.0,  # right_hip_yaw
    50.0,  # torso
    50.0,  # left_hip_roll
    50.0,  # right_hip_roll
    50.0,  # left_shoulder_pitch
    50.0,  # right_shoulder_pitch
    50.0,  # left_hip_pitch
    50.0,  # right_hip_pitch
    50.0,  # left_shoulder_roll
    50.0,  # right_shoulder_roll
    50.0,  # left_knee
    50.0,  # right_knee
    50.0,  # left_shoulder_yaw
    50.0,  # right_shoulder_yaw
    50.0,  # left_ankle
    50.0,  # right_ankle
    50.0,  # left_elbow
    50.0,  # right_elbow
]

    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0
    dof_vel_scale: float = 0.1

    death_cost: float = -1.0
    termination_height: float = 0.8

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01

    # for teleoperation,end effector track, and hip position track
    ends_track_scale: float = 0.1
    ends_vel_track_scale: float = 0.01  
    contact_scale: float = 1.0
    vertical_scale: float = 0.5
    vertical_vel_scale: float = 0.2
    heading_vel_scale: float = 0.0
    arm_pitch_scale: float = 2.0
    arm_roll_scale: float = 2.0
    arm_yaw_scale: float = 2.0
    elbow_bend_scale: float = 1.0
    knee_reverse_punish: float = 2.0

class H1Env(LocomotionEnv): # use listened data here, process data_2_skeleton here
    cfg: H1EnvCfg 

    def __init__(self, cfg: H1EnvCfg , render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)



    # acquire properties
        torch.set_printoptions(precision=4)
        self._body_ids, self._body_names = self.robot.find_bodies(self.robot.body_names)
        print("***********************body_ids is", self._body_ids)
        print("***********************body_names is", self._body_names)
        print("******************joint_dof_idx is:", self._joint_dof_idx)
        
        _ , self.dof_names = self.robot.find_joints(".*")
        self.ee_names = ['left_ankle_link','right_ankle_link','left_elbow_link','right_elbow_link']

        self.shoulder_pitch_ids = [self._joint_dof_idx[self.dof_names.index(name)] for name in ['left_shoulder_pitch','right_shoulder_pitch']]
        self.shoulder_roll_ids = [self._joint_dof_idx[self.dof_names.index(name)] for name in ['left_shoulder_roll','right_shoulder_roll']]
        self.shoulder_yaw_ids = [self._joint_dof_idx[self.dof_names.index(name)] for name in ['left_shoulder_yaw','right_shoulder_yaw']]
        self.elbow_ids = [self._joint_dof_idx[self.dof_names.index(name)] for name in ['left_elbow','right_elbow']]
        self.knee_ids = [self._joint_dof_idx[self.dof_names.index(name)] for name in ['left_knee', 'right_knee']]
        self.ee_body_ids = [self._body_ids[self._body_names.index(name)] for name in self.ee_names]
        self.pelvis_body_ids = self._body_ids[self._body_names.index('pelvis')]
        print("*******************shoulder_joint_ids", self.shoulder_pitch_ids)
        print("*******************knee_joint_ids", self.knee_ids)
        print("*******************ee_ids", self.ee_body_ids)
        print("*******************base_ids", self.pelvis_body_ids)

        self.lf_jacobian = self.robot.root_physx_view.get_jacobians()[0, self.ee_body_ids[0], :, :]

        self.rf_jacobian = self.robot.root_physx_view.get_jacobians()[0, self.ee_body_ids[1], :, :]

        self.pelvis_jacobian = self.robot.root_physx_view.get_jacobians()[0, self.pelvis_body_ids, :, :]

        print("******************base jacobian:", self.pelvis_jacobian.shape)

        


    # modify some configurations of locomotionenv
        self.targets = torch.tensor([1000, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1) # move only at the current place
        )
        self.targets += self.scene.env_origins


    # start send and listen threads here
        self.ThreadManager = Communicating() #run on cpu!!!!

    # add motion processing instance
        self.MasterMotion = Intermediate_Motion(self.ThreadManager.master_data, device = self.sim.device)



        #self.sendthread = Thread(target=self.ThreadManager.send)

        self.listenthread = Thread(target = self.ThreadManager.listen)

        #self.sendthread.start()

        self.listenthread.start()





    #def _reset_idx(self, *args, **kwargs): #rendomize parameter and reset walking targets here
    #    #self.targets[:, :2] = torch.rand((self.num_envs, 2), device=self.sim.device) * 2000 - 1000 # range -1000~1000 in x and y 
    #    rand_target = torch.rand(2, device=self.sim.device) * 2000 - 1000
    #    self.targets[:, :2] = rand_target.unsqueeze(0)
    #    self.targets += self.scene.env_origins
    #    print("====================== target reset", self.targets[:, :2])
    #    super()._reset_idx(*args, **kwargs)
    


    def _update_ee_vel_pos(self):

        lf_state_w = self.robot.data.body_state_w[:,self.ee_body_ids[0],:]

        rf_state_w = self.robot.data.body_state_w[:,self.ee_body_ids[1],:]
        
        self.lf_pos_w, self.lf_quat_w, self.lf_vel_w, self.lf_omg_w = lf_state_w[:,:3], lf_state_w[:,3:7], lf_state_w[:,7:10], lf_state_w[:,10:13]

        self.rf_pos_w, self.rf_quat_w, self.rf_vel_w, self.rf_omg_w = rf_state_w[:,:3], rf_state_w[:,3:7], rf_state_w[:,7:10], rf_state_w[:,10:13]

        Rot_w_root = quaternion_to_rotation_matrix(self.robot.data.root_quat_w).transpose(-2, -1)  # inverse (transpose)

        # calculate states in root frame
        self.lf_vel_root = torch.einsum('bij,bj->bi', Rot_w_root, self.lf_vel_w - self.robot.data.root_lin_vel_w)
        self.lf_omg_root = torch.einsum('bij,bj->bi', Rot_w_root, self.lf_omg_w - self.robot.data.root_ang_vel_w)
        self.rf_vel_root = torch.einsum('bij,bj->bi', Rot_w_root, self.rf_vel_w - self.robot.data.root_lin_vel_w)
        self.rf_omg_root = torch.einsum('bij,bj->bi', Rot_w_root, self.rf_omg_w - self.robot.data.root_ang_vel_w)
        self.lf_pos_root = torch.einsum('bij,bj->bi', Rot_w_root, self.lf_pos_w - self.robot.data.root_pos_w)
        self.rf_pos_root = torch.einsum('bij,bj->bi', Rot_w_root, self.rf_pos_w - self.robot.data.root_pos_w)

        root_quat_w_conj = quaternion_conjugate(self.robot.data.root_quat_w)

        self.lf_quat_root = quaternion_multiply(root_quat_w_conj, self.lf_quat_w)
        self.rf_quat_root = quaternion_multiply(root_quat_w_conj, self.rf_quat_w)

        #print("*********lf_pos_root size ", self.lf_pos_root.shape)
        #print("*********lf_vel_root size ", self.lf_vel_root.shape)
        #print("*********rf_omg_root size ", self.rf_omg_root.shape)
        #print("*********rf_quat_root size ", self.rf_quat_root.shape)



   

    #modify reward definition based on father methods
    def _get_rewards(self):
    #    
        self._update_ee_vel_pos()

        self.MasterMotion.MasterData_Process(self.ThreadManager.master_data)

        self.lf_contact, self.rf_contact = Judge_contact(self.robot.data.body_acc_w[...,self.pelvis_body_ids,2].unsqueeze(-1),
                                                           self.lf_pos_root[..., 2].unsqueeze(-1),
                                                             self.rf_pos_root[...,2].unsqueeze(-1))
        
        #print("************recvdata: ",self.ThreadManager.master_data)


        master_print = [*self.MasterMotion.master_root_pos_lf.tolist(),      
                *self.MasterMotion.master_root_pos_rf.tolist(),
                *self.MasterMotion.master_root_vel_lf.tolist(),
                *self.MasterMotion.master_root_vel_rf.tolist(),
                *self.MasterMotion.master_root_quat_lf.tolist(), 
                *self.MasterMotion.master_root_quat_rf.tolist(),       
                *self.MasterMotion.master_root_omeg_lf.tolist(),       
                *self.MasterMotion.master_root_omeg_rf.tolist(),       
                float(self.MasterMotion.master_contact_flgl.item()),      
                float(self.MasterMotion.master_contact_flgr.item())]
        
        slave_print = [*self.lf_pos_root[0,...].tolist(),
                *self.rf_pos_root[0,...].tolist(),
                *self.lf_vel_root[0,...].tolist(),
                *self.rf_vel_root[0,...].tolist(),
                *self.lf_quat_root[0,...].tolist(),
                *self.rf_quat_root[0,...].tolist(),
                *self.lf_omg_root[0,...].tolist(),
                *self.rf_omg_root[0,...].tolist(),
                float(self.lf_contact[0,...].item()),
                float(self.rf_contact[0,...].item())]
        
        formatted_masterdata = [round(x, 4) for x in master_print]
        formatted_slavedata = [round(x, 4) for x in slave_print]
        
        #print("************masterdata: ", formatted_masterdata)
        #print("************slavedata: ", formatted_slavedata)

        #========================================================================#
        # add reward
        rwd_plus = super()._get_rewards()
        batchsize = rwd_plus.size(0)

        #print("******************slave_root_pos_lf is on:", self.lf_pos_root.shape)
        #print("******************slave_root_vel_lf is on:", self.rf_pos_root.shape)
        #print("******************slave_root_pos_rf is on:", self.lf_vel_root.shape)
        #print("******************slave_root_vel_rf is on:", self.rf_vel_root.shape)
        #print("******************slave_root_quat_lf is on:", self.lf_quat_root.shape)
        #print("******************slave_root_omeg_lf is on:", self.rf_quat_root.shape)
        #print("******************slave_root_quat_rf is on:", self.lf_omg_root.shape)
        #print("******************slave_root_omeg_rf is on:", self.rf_omg_root.shape)
        #print("******************slave_contact_flgl is on:", self.lf_contact.shape)
        #print("******************slave_contact_flgr is on:", self.rf_contact.shape)
        #print("************rwd_plus: ", rwd_plus.shape)

        #print("================== master_root_pos_lf[2]:",self.MasterMotion.master_root_pos_lf[0,2])
        #print("================== lf_pos_root[2]:",self.lf_pos_root[0,2])
        #print("================== master_root_pos_rf[2]:",self.MasterMotion.master_root_pos_rf.shape)
        #print("================== rf_pos_root[2]:",self.rf_pos_root.shape)


        rwd_plus -= torch.sum((self.MasterMotion.master_root_pos_lf[2] -    self.lf_pos_root[0,2])**2, dim = -1) * self.cfg.ends_track_scale
        rwd_plus -= torch.sum((self.MasterMotion.master_root_pos_rf[2] -    self.rf_pos_root[0,2])**2, dim = -1) * self.cfg.ends_track_scale
        #rwd_plus += torch.sum((self.MasterMotion.master_root_vel_lf -    self.lf_vel_root)**2, dim = -1) * self.cfg.ends_vel_track_scale
        #rwd_plus += torch.sum((self.MasterMotion.master_root_vel_rf -    self.rf_vel_root)**2, dim = -1) * self.cfg.ends_vel_track_scale
        #rwd_plus += torch.sum((self.MasterMotion.master_root_quat_lf -   self.lf_quat_root)**2, dim = -1) * self.cfg.ends_track_scale
        #rwd_plus += torch.sum((self.MasterMotion.master_root_quat_rf -   self.rf_quat_root)**2, dim = -1) * self.cfg.ends_track_scale
        #rwd_plus += torch.sum((self.MasterMotion.master_root_omeg_lf -   self.lf_omg_root)**2, dim = -1) * self.cfg.ends_vel_track_scale
        #rwd_plus += torch.sum((self.MasterMotion.master_root_omeg_rf -   self.rf_omg_root)**2, dim = -1) * self.cfg.ends_vel_track_scale
        #rwd_plus += torch.where(self.MasterMotion.master_contact_flgl == self. lf_contact.squeeze(-1), 
        #                        torch.ones(batchsize, device=self.device), torch.zeros(batchsize, device=self.device)) * self.cfg.contact_scale
        #rwd_plus += torch.where(self.MasterMotion.master_contact_flgr == self. rf_contact.squeeze(-1),
        #                        torch.ones(batchsize, device=self.device), torch.zeros(batchsize, device=self.device)) * self.cfg.contact_scale


        # arm angle track, #knee must not reverse
        #print("*************self.dof_pos_scaled", self.dof_pos_scaled.shape)
        #print("*************torch.tensor(self.shoulder_pitch_ids, device=self.device)", torch.tensor(self.shoulder_pitch_ids, device=self.device).shape)
        #print("=============== arm pitch:", self.dof_pos[:,torch.tensor(self.shoulder_pitch_ids, device=self.device)])
        #print("=============== arm roll:", self.dof_pos[:,torch.tensor(self.shoulder_roll_ids, device=self.device)])
        #print("=============== arm yaw:", self.dof_pos[:,torch.tensor(self.shoulder_yaw_ids, device=self.device)])
        #print("=============== elbow yaw:", self.dof_pos[:,torch.tensor(self.elbow_ids, device=self.device)])
        #print("=============== knee angle:", self.dof_pos[:, torch.tensor(self.knee_ids, device=self.device)])
        rwd_plus -= torch.sum(self.dof_pos[:,torch.tensor(self.shoulder_pitch_ids, device=self.device)]**2, dim = -1) * self.cfg.arm_pitch_scale
        rwd_plus -= torch.sum(self.dof_pos[:,torch.tensor(self.shoulder_roll_ids, device=self.device)]**2, dim = -1) * self.cfg.arm_roll_scale
        rwd_plus -= torch.sum(self.dof_pos[:,torch.tensor(self.shoulder_yaw_ids, device=self.device)]**2, dim = -1) * self.cfg.arm_yaw_scale
        rwd_plus -= torch.sum(self.dof_pos[:,torch.tensor(self.elbow_ids, device=self.device)]**2, dim = -1) * self.cfg.elbow_bend_scale
        mask = (self.dof_pos[:, torch.tensor(self.knee_ids, device=self.device)] > 0).all(dim=-1) #where cannot sum up bool matrix directly
        rwd_plus += torch.where(mask > 0, torch.tensor(1), torch.tensor(-1))*self.cfg.knee_reverse_punish
    #    
        return rwd_plus

    # modify observation based on father method
    def _get_observations(self):
        obs = super()._get_observations()
        batchsize = obs["policy"].size(0)
        

        master_root_pos_lf =self.MasterMotion.master_root_pos_lf.unsqueeze(0).expand(batchsize,-1)
        master_root_vel_lf =self.MasterMotion.master_root_vel_lf.unsqueeze(0).expand(batchsize,-1)
        master_root_pos_rf =self.MasterMotion.master_root_pos_rf.unsqueeze(0).expand(batchsize,-1)
        master_root_vel_rf =self.MasterMotion.master_root_vel_rf.unsqueeze(0).expand(batchsize,-1)
        master_root_quat_lf= self.MasterMotion.master_root_quat_lf.unsqueeze(0).expand(batchsize,-1)
        master_root_omeg_lf= self.MasterMotion.master_root_omeg_lf.unsqueeze(0).expand(batchsize,-1)
        master_root_quat_rf= self.MasterMotion.master_root_quat_rf.unsqueeze(0).expand(batchsize,-1)
        master_root_omeg_rf= self.MasterMotion.master_root_omeg_rf.unsqueeze(0).expand(batchsize,-1)
        master_contact_flgl= self.MasterMotion.master_contact_flgl.unsqueeze(0).expand(batchsize,-1)
        master_contact_flgr= self.MasterMotion.master_contact_flgr.unsqueeze(0).expand(batchsize,-1)
        
    

        #print("******************obs is on:", obs["policy"].shape)
#   #
        #print("******************master_root_pos_lf is on:", master_root_pos_lf.shape)
        #print("******************master_root_vel_lf is on:", master_root_vel_lf.shape)
        #print("******************master_root_pos_rf is on:", master_root_pos_rf.shape)
        #print("******************master_root_vel_rf is on:", master_root_vel_rf.shape)
        #print("******************master_root_quat_lf is on:", master_root_quat_lf.shape)
        #print("******************master_root_omeg_lf is on:", master_root_omeg_lf.shape)
        #print("******************master_root_quat_rf is on:", master_root_quat_rf.shape)
        #print("******************master_root_omeg_rf is on:", master_root_omeg_rf.shape)
        #print("******************master_contact_flgl is on:", master_contact_flgl.shape)
        #print("******************master_contact_flgr is on:", master_contact_flgr.shape)






        new_obs = torch.cat((
            obs["policy"],
            master_root_pos_lf,
            master_root_vel_lf,
            master_root_pos_rf,
            master_root_vel_rf,
            master_root_quat_lf,
            master_root_omeg_lf,
            master_root_quat_rf,
            master_root_omeg_rf,
            master_contact_flgl,
            master_contact_flgr
            ),
            dim = -1
            )
        observations = {"policy": new_obs}
        return obs