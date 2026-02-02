from dataclasses import MISSING

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


class MySceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )


    # base for dual arm
    robot_base = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/robot_base",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{DIGITALTWIN_NUCLEUS_DIR}/Assets/Warehouse/Furnishing/Workbenches/LabWorkbench_A/LabWorkbench_A02_01.usd",                           
            # rigid_props or other props as needed
            scale=(0.005,0.02,0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.5, 0.0, 0.0),
            #rot=(0.0, 0, 0, 0.0),
        ),
    )


     #dual franka
    arml: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/arml")
    arml.init_state.pos=(-0.5,0.5,0.9)
    
    
    armr: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/armr")
    armr.init_state.pos=(-0.5,-0.5,0.9)
    #robot_2.init_state.rot=(0,0,0,1)

    # #lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1000.0),
    )
        

    # simulate conveyor
    worktable = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/worktable",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{DIGITALTWIN_NUCLEUS_DIR}/Assets/Warehouse/Furnishing/Workbenches/LabWorkbench_A/LabWorkbench_A02_01.usd",                           
            # rigid_props or other props as needed
            scale=(0.005,0.03,0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            #rot=(0.0, 0, 0, 0.0),
        ),
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
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.6), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
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
class DualFrankaEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the MuJoCo-style Humanoid walking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=1, env_spacing=5.0)
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

class DualFrankaEnv(ManagerBasedRLEnv):

    def step(self, actions):
        obs, reward, terminated, time_out, extras = super().step(actions)

        ## 获取摄像头 sensor 对象
        cam = self.scene.sensors.get("camera")
#
        if cam is None:
            print("Camera sensor not found in scene.sensors!")
        else:
            #print(cam.data.output.keys())
            rgb = cam.data.output['rgb']
            if rgb is None:
                print("Camera found, but RGB is None (sensor not updated?)")
            else:
                print("Camera OK:", rgb.shape)

        return obs, reward, terminated, time_out, extras