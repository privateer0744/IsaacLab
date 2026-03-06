from dataclasses import MISSING
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg,ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg 
from isaaclab.managers import EventTermCfg 
from isaaclab.managers import ObservationGroupCfg 
from isaaclab.managers import ObservationTermCfg 
from isaaclab.managers import RewardTermCfg 
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import DIGITALTWIN_NUCLEUS_DIR
from isaaclab_assets import FRANKA_PANDA_CFG
import isaaclab_tasks.manager_based.hzy_manipulate.mdp as mdp
from isaaclab.sensors import CameraCfg
from isaaclab.actuators import ImplicitActuatorCfg
import math

class NonprehensileSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )


    MyScene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/MyScene",
        spawn=sim_utils.UsdFileCfg(
            usd_path="D:/projects/Non-Prehensile/Basket2.usd",                           
            # rigid_props or other props as needed
            scale=(1.0,1.0,1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=0.1,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(math.cos(math.pi / 2/2), math.sin(math.pi / 2/2), 0, 0),
        ),
    )
    #robot = ArticulationCfg(
    #    prim_path="{ENV_REGEX_NS}/MyScene/panda_instanceable",
    #    spawn=None,
    #    init_state=ArticulationCfg.InitialStateCfg(
    #    joint_pos={
    #        "panda_joint1": 0.0,
    #        "panda_joint2": -0.0,
    #        "panda_joint3": 0.0,
    #        "panda_joint4": -1.57,
    #        "panda_joint5": 0.0,
    #        "panda_joint6": 0.0,
    #        "panda_joint7": 0.0,
    #        "panda_finger_joint.*": 0.0,
    #        },
    #    ),
    #    actuators={
    #    "panda_shoulder": ImplicitActuatorCfg(
    #        joint_names_expr=["panda_joint[1-4]"],
    #        effort_limit=1e9,
    #        velocity_limit=2.175,
    #        stiffness=1e7,
    #        damping=1e6,
    #    ),
    #    "panda_forearm": ImplicitActuatorCfg(
    #        joint_names_expr=["panda_joint[5-7]"],
    #        effort_limit=1e9,
    #        velocity_limit=2.61,
    #        stiffness=1e7,
    #        damping=1e6,
    #    ),
    #    "panda_hand": ImplicitActuatorCfg(
    #        joint_names_expr=["panda_finger_joint.*"],
    #        effort_limit=1e9,
    #        velocity_limit=0.2,
    #        stiffness=1e7,
    #        damping=1e6,
    #    ),
    #},
    #)
    
    bottles = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/MyScene/bottle1_.*",
        spawn=None,
    )
    targetbottle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/MyScene/targetbottle",
        spawn=None,
    )

     #dual franka
    #robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
    #robot.init_state.pos=(-0.5,0.5,0.9)
    
    
    # #lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1000.0),
    )
        
    #camera
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 2.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )




@configclass
class ObservationsCfg:
    class Dummy(ObservationGroupCfg):
        dummy_obs = ObservationTermCfg(func=mdp.dummy_obs)
    policy: Dummy = Dummy()


#@configclass
#class ObservationsCfg:
#    class PolicyCfg(ObservationGroupCfg):
#        # observation terms (order preserved)
#        joint_pos_rel = ObservationTermCfg(func=mdp.joint_pos_rel)
#        joint_vel_rel = ObservationTermCfg(func=mdp.joint_vel_rel)
#    policy: PolicyCfg = PolicyCfg()

@configclass
class ActionsCfg:
    pass

@configclass
class RewardsCfg:
    dummy = RewardTermCfg(func=mdp.dummy_reward, weight=0.0)

@configclass
class TerminationsCfg:
    # (1) Time out
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

@configclass
class EventCfg:
    pass




@configclass
class NonprehensileEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the MuJoCo-style Humanoid walking environment."""

    # Scene settings
    scene: NonprehensileSceneCfg = NonprehensileSceneCfg(num_envs=16, env_spacing=5.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 16.0
        # simulation settings
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0

class NonprehensileEnv(ManagerBasedRLEnv):

    def __init__(self, cfg: NonprehensileEnvCfg , render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.task_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.obstacle_handle = self.scene["bottles"]
        self.target_obj_handle = self.scene["targetbottle"]
        #self.robot = self.scene["robot"]
        self.ee_link_name = "panda_hand"
        #self._ee_link_idx, _ = self.robot.find_bodies(self.ee_link_name)
        self.num_bottles_per_env = self.obstacle_handle.data.body_state_w.shape[0] // self.num_envs
        self.father_rot=torch.tensor([math.cos(math.pi / 2/2), math.sin(math.pi / 2/2), 0, 0], dtype=torch.float32, device=self.device).view(1,1,4).repeat(self.num_envs, self.num_bottles_per_env, 1)
        self.min_inscribed_radius = torch.zeros((self.num_envs,self.num_bottles_per_env), dtype=torch.float32, device=self.device)
        bottles_state_w = self.obstacle_handle.data.body_state_w.view(self.num_envs, self.num_bottles_per_env, 13)

        #get obstacle pos
        self.pre_bottles_vel_w = bottles_state_w[:,:,7:10]
        self.pre_bottles_omeg_w = bottles_state_w[:,:,10:13]



    def _reset_idx(self, env_ids: torch.Tensor, *args, **kwargs):
        # 必须在重置环境时，将对应环境的阶段归零
        super()._reset_idx(env_ids, *args, **kwargs)
        self.task_phase[env_ids] = 1


    def step(self, actions):
        obs, reward, terminated, time_out, extras = super().step(actions)
        
        #print(f"瓶子在 Tensor 中的真实顺序: {self.obstacle_handle.cfg.prim_path}")
        #mdp.get_pos(obstacle_handle)
        target_obj_state_w = self.target_obj_handle.data.body_state_w

        
        bottles_state_w = self.obstacle_handle.data.body_state_w.view(self.num_envs, self.num_bottles_per_env, 13)
        #print('bottles_state_w:',bottles_state_w)
        #get obstacle pos vel
        self.bottles_pos_w = bottles_state_w[:,:,:3]

        son_bottles_quat_w = bottles_state_w[:,:,3:7].clone()#把usd composer的y-up坐标系转换成真实的坐标系
        #print("son_bottles_quat_w ",son_bottles_quat_w)
        son_bottles_quat_w[:,:,2:3]=-1*bottles_state_w[:,:,6:7]
        son_bottles_quat_w[:,:,3:4]=bottles_state_w[:,:,5:6]
        #print("son_bottles_quat_w ",son_bottles_quat_w)
        self.bottles_quat_w = mdp.quat_mul(mdp.quat_conjugate(self.father_rot), son_bottles_quat_w)


        self.bottles_vel_w = bottles_state_w[:,:,7:10]
        self.bottles_omeg_w = bottles_state_w[:,:,10:13]
        #get target object pos
        self.target_pos_w = target_obj_state_w[:,:,:3]
        self.target_quat_w = target_obj_state_w[:,:,3:7]

        #get obstacle pos vel
        #self.bottles_pos_w = bottles_state_w[:,:,:3]
        #self.bottles_vel_w = bottles_state_w[:,:,3:6]
        #self.bottles_quat_w = bottles_state_w[:,:,6:10]
        #self.bottles_omeg_w = bottles_state_w[:,:,10:13]
        ##get target object pos
        #self.target_pos_w = target_obj_state_w[:,:,:3]
        #self.target_quat_w = target_obj_state_w[:,:,6:10]



        #get end effector pos
        #ee_state_w = self.robot.data.body_state_w[:,self._ee_link_idx,:].squeeze(1)
       
        #self.ee_pos_w, self.ee_quat_w, self.ee_vel_w, self.ee_omg_w = ee_state_w[:,:3], ee_state_w[:,3:7], ee_state_w[:,7:10], ee_state_w[:,10:13]

        #self.bottle_pos_w, self.bottle_quat_w, self.bottle_vel_w, self.bottle_omeg_w  = bottles_state_w[:,:,:3], bottles_state_w[:,:,3:7], bottles_state_w[:,:,7:10], bottles_state_w[:,:,10:13]
        #print("末端执行器状态: ",ee_state_w)
        #print("末端执行器位置: ",self.ee_pos_w)
        #print("末端执行器姿态: ",self.ee_quat_w)
        #print("末端执行器速度: ",self.ee_vel_w)
        #print("末端执行器角速度: ",self.ee_omg_w)
        #print("目标对象位置: ",self.target_pos_w)
        self.task_phase = mdp.update_phase(self)


        self.min_inscribed_radius = mdp.get_min_inscribed_radius(self) #获取目标对象的抓取通道开放半径

        #self.max_momentum,self.max_moment = mdp.get_obj_max_momentum(self) #用于惩罚将任意物体打飞
#
        #self.contactforce, self.contactforce_gradient  = mdp.get_contactforce(self)# 用于惩罚过大的接触力
#
        #self.max_stuck_duration = mdp.get_stuck_duration(self) #用于惩罚堵塞时间过长
#
        #self.wander_duration = mdp.get_wander_duration(self)#用于惩罚末端装置无效闲逛挂机

        #print("当前状态: ",self.task_phase)
        #print("size of self.bottles_pos_w: ",self.bottles_pos_w.shape)
        #print("size of self.target_pos_w: ",self.target_pos_w.shape)

        #print(f"检测到瓶子数量: {self.scene['bottles'].num_instances}")
#        ## 获取摄像头 sensor 对象
#        cam = self.scene.sensors.get("camera")
##
#        if cam is None:
#            print("Camera sensor not found in scene.sensors!")
#        else:
#            #print(cam.data.output.keys())
#            rgb = cam.data.output['rgb']
#            if rgb is None:
#                print("Camera found, but RGB is None (sensor not updated?)")
#            else:
#                print("Camera OK:", rgb.shape)

        self.pre_bottles_vel_w = self.bottles_vel_w
        self.pre_bottles_omeg_w = self.bottles_omeg_w

        return obs, reward, terminated, time_out, extras