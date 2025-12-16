
"""Differential-game-style controller for red missiles' PN gains.

目标：在不大改项目结构的前提下，引入一个“微分博弈”层，
对红方导弹的比例导引参数（导航增益 N）进行 *联合*、*实时* 调整。

建模思路（离散近似）：
- 系统状态为 (blue_pos, blue_vel, missile_pos, missile_vel, nav_gains)；
- 红方控制量是每枚导弹的导航增益向量 nav_gains ∈ R^M；
- 蓝方控制量是其机动（由 RL 智能体决定，不在本模块显式建模）；
- 代价函数采用一阶近似的“瞬时代价”：
    J ≈ sum_i w_dist * ||r_i'||^2 + w_gain * N_i^2
  其中 r_i' 是导弹 i 在下一步与蓝机的相对位置（由 PN 更新预测）。

微分博弈视角：
- 把上式看成连续时间博弈的离散化阶段代价；
- 红方希望通过调节 nav_gains 来 *最小化* J；
- 蓝方隐含地通过其 RL 行为来“对抗”红方，但本模块只实现红方一侧。

数值实现：
- 使用简单的“梯度下降 + 有界控制”迭代：
    N_{k+1} = Proj_[N_min,N_max]( N_k - eta * dJ/dN )
- dJ/dN 通过对每个导弹导航增益做有限差分来近似：
    dJ/dN_i ≈ (J(N_i + δ) - J(N_i - δ)) / (2δ)
- 其中 J 的计算只模拟“一步”导弹更新，蓝机位置假定在这一步内近似不变。

注意：
- 这个实现不试图精确求解 HJI 方程，而是作为微分博弈思想的一个可运行、
  易于扩展的数值近似层，嵌在原有 PN 导引 + Nash 发射位置架构之上。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .missile_dynamics import update_missiles_pn
from config import EnvConfig


@dataclass
class DiffGameConfig:
    step_size: float
    delta_gain: float
    gain_min: float
    gain_max: float
    w_dist: float
    w_gain: float


class DifferentialGameController:
    """Joint differential-game controller for PN gains of all missiles."""

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self.dg_cfg = DiffGameConfig(
            step_size=cfg.diff_step_size,
            delta_gain=cfg.diff_delta_gain,
            gain_min=cfg.diff_gain_min,
            gain_max=cfg.diff_gain_max,
            w_dist=cfg.diff_w_dist,
            w_gain=cfg.diff_w_gain,
        )

    # ------------------------------------------------------------------
    def update_nav_gains(
        self,
        blue_pos: np.ndarray,
        blue_vel: np.ndarray,
        missile_pos: np.ndarray,
        missile_vel: np.ndarray,
        nav_gains: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """One discrete-time update of nav_gains via gradient descent.

        Args:
            blue_pos: (3,)
            blue_vel: (3,)
            missile_pos: (M,3)
            missile_vel: (M,3)
            nav_gains: (M,) current navigation gains
            dt: time step

        Returns:
            new_nav_gains: (M,) updated navigation gains
        """
        nav_gains = np.asarray(nav_gains, dtype=float)
        M = nav_gains.shape[0]

        grad = np.zeros_like(nav_gains)
        delta = self.dg_cfg.delta_gain

        for i in range(M):
            # Positive perturbation
            nav_plus = nav_gains.copy()
            nav_plus[i] += delta
            cost_plus = self._predict_cost(blue_pos, blue_vel, missile_pos, missile_vel, nav_plus, dt)

            # Negative perturbation
            nav_minus = nav_gains.copy()
            nav_minus[i] -= delta
            cost_minus = self._predict_cost(blue_pos, blue_vel, missile_pos, missile_vel, nav_minus, dt)

            grad[i] = (cost_plus - cost_minus) / (2.0 * delta)

        # Gradient descent step (minimization).
        new_nav = nav_gains - self.dg_cfg.step_size * grad

        # Project onto admissible interval [gain_min, gain_max].
        new_nav = np.clip(new_nav, self.dg_cfg.gain_min, self.dg_cfg.gain_max)

        return new_nav

    # ------------------------------------------------------------------
    def _predict_cost(
        self,
        blue_pos: np.ndarray,
        blue_vel: np.ndarray,
        missile_pos: np.ndarray,
        missile_vel: np.ndarray,
        nav_gains: np.ndarray,
        dt: float,
    ) -> float:
        """Predict a one-step running cost for given nav_gains.

        We *simulate* one PN update for the missiles with the given gains
        (without touching the real environment state), then compute:

            J = sum_i w_dist * ||r_i'||^2 + w_gain * N_i^2

        where r_i' = missile_pos_i' - blue_pos (assuming blue position
        approximately constant over the small interval dt).
        """
        nav_gains = np.asarray(nav_gains, dtype=float)
        # One-step PN update (copy to avoid side effects).
        m_pos_next, m_vel_next = update_missiles_pn(
            missile_pos,
            missile_vel,
            blue_pos,
            blue_vel,
            self.cfg.missile_speed,
            dt,
            nav_gains,
        )

        # Relative distance after the step.
        rel = m_pos_next - blue_pos.reshape(1, 3)
        dist_sq = np.sum(rel ** 2, axis=1)  # (M,)

        J_dist = self.dg_cfg.w_dist * float(np.sum(dist_sq))
        J_gain = self.dg_cfg.w_gain * float(np.sum(nav_gains ** 2))

        return J_dist + J_gain
