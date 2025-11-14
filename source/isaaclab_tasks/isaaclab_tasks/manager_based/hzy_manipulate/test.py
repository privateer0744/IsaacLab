from isaaclab.sim.spawners.from_files.from_files import spawn_ground_plane
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg

# 配置 ground plane
ground_cfg = GroundPlaneCfg(size=(10, 10))
spawn_ground_plane(
    prim_path="/World/ground_plane",
    cfg=ground_cfg,
    translation=(0, 0, 0),
    orientation=(1, 0, 0, 0)
)
