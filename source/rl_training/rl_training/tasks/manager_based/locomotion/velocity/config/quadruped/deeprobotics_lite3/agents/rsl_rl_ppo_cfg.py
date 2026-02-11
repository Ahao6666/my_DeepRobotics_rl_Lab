# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DeeproboticsLite3RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24      # 每个环境在一个数据收集周期内的步数
    max_iterations = 30000      # 最大训练迭代次数
    save_interval = 100         # 每 100 次迭代保存一次模型检查点
    experiment_name = "deeprobotics_lite3_rough"    # 实验名称（日志目录前缀）
    empirical_normalization = False     # 经验归一化
    clip_actions = 100          # 动作裁剪范围
    policy = RslRlPpoActorCriticCfg(        # 策略网络配置
        init_noise_std=1.0,                 # 初始噪声标准差
        noise_std_type="log",               # 噪声类型（对数）
        actor_hidden_dims=[512, 256, 128],    # Actor 隐层维度（3层）
        critic_hidden_dims=[512, 256, 128],   # Critic 隐层维度（3层）
        activation="elu",                   # 激活函数
    )
    algorithm = RslRlPpoAlgorithmCfg(       # PPO 算法超参数
        value_loss_coef=1.0,                # 价值函数损失权重
        use_clipped_value_loss=True,
        clip_param=0.2,                     # PPO 剪裁参数（ε）
        entropy_coef=0.01,                  # 熵奖励系数（鼓励探索）
        num_learning_epochs=5,              # 每个数据批次的学习轮数
        num_mini_batches=4,                 # 数据分割为 4 个小批次
        learning_rate=1.0e-3,               # 学习率
        schedule="adaptive",                # 学习率自适应调度
        gamma=0.99,                         # 折扣因子（长期回报权重）
        lam=0.95,                           # GAE λ（广义优势估计）
        desired_kl=0.01,                    # KL 散度目标（自适应学习率用）
        max_grad_norm=1.0,                  # 梯度剪裁阈值
    )


@configclass
class DeeproboticsLite3FlatPPORunnerCfg(DeeproboticsLite3RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "deeprobotics_lite3_flat"
