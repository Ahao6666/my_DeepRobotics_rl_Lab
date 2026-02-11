# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Flat-Deeprobotics-Lite3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DeeproboticsLite3FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLite3FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsLite3FlatTrainerCfg",
    },
)
# 这段代码的作用是将自定义强化学习环境注册到 Gymnasium 环境库中，
# 使其可以通过 gym.make() 直接创建。
gym.register(
    id="Rough-Deeprobotics-Lite3-v0",           # 环境的唯一标识符
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  # 环境的主类
    disable_env_checker=True,       # 是否禁用环保检查器
    kwargs={        # 	传递给环境的配置参数
        # 环境配置（任务、奖励、观察、事件等）
        # 作用：定义机器人在粗糙地形的训练环境
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:DeeproboticsLite3RoughEnvCfg",
        # PPO 训练器配置（学习率、网络大小、训练步数等）
        # 作用：用于 train.py 训练模型
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsLite3RoughPPORunnerCfg",
        # CuSRL 训练器配置（另一种强化学习算法）
        # 作用：支持不同的训练方法
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsLite3RoughTrainerCfg",
    },
)

