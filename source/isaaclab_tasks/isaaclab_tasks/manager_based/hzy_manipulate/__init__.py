# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
from .hzy_env_cfg import DualFrankaEnv, DualFrankaEnvCfg
from .nonprehensile_env_cfg import NonprehensileEnv, NonprehensileEnvCfg
from . import agents

gym.register(
    id="Isaac-Hzy-Dual-Franka-v0",
    entry_point="isaaclab_tasks.manager_based.hzy_manipulate.hzy_env_cfg:DualFrankaEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DualFrankaEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DualFrankaPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Nonprehensile-Franka-v0",
    entry_point="isaaclab_tasks.manager_based.hzy_manipulate.nonprehensile_env_cfg:NonprehensileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": NonprehensileEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NonprehensilePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

