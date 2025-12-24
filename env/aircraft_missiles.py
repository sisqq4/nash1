
"""Aircraft and radar-missile classes used in the escape environment."""

from __future__ import annotations

from typing import Optional, Tuple

import math as m
import numpy as np

from .missile_dynamics import update_blue_state, update_missiles_pn


class Missiles:
    """雷达导弹模型（红方导弹）。

    - 当前环境：通过 ``step`` 以恒定速度进行比例导引（批量导弹）；
    - 扩展接口：``_advance_time`` + ``MissilePosition`` 可实现带加速和衰减的
      三维比例导引，用于单发导弹的更精细建模。
    """

    def __init__(
        self,
        missile_plist: Optional[list] = None,
        V: Optional[float] = None,
        Pitch: Optional[float] = None,
        Heading: Optional[float] = None,
        dt: float = 0.01,
        g: float = 9.6,
        k1: float = 7.0,
        k2: float = 7.0,
        target_speed: float = 1200.0,
        boost_duration: float = 5.0,
        speed_decay_interval: float = 1.0,
        speed_decay_factor: float = 0.99,
        target_id: Optional[int] = None,
        missile_speed: Optional[float] = None,
    ) -> None:
        # --- 单枚导弹状态（面向后续扩展，当前环境未直接使用） ---
        if missile_plist is None:
            missile_plist = [0.0, 0.0, 0.0]
        self.X, self.Y, self.Z = missile_plist

        if V is None:
            V = missile_speed if missile_speed is not None else target_speed
        self.V = float(V)

        self.Pitch = float(Pitch) if Pitch is not None else 0.0
        self.Heading = float(Heading) if Heading is not None else 0.0

        self.attacking = True
        self.target_id = target_id

        # 导弹自身参数
        self.g = float(g)
        self.dt = float(dt)
        self.k1 = float(k1)
        self.k2 = float(k2)

        # 两段运动（加速 + 末端滑行 + 衰减）参数
        self.target_speed = float(target_speed)
        self.boost_duration = float(boost_duration)
        self.speed_decay_interval = float(speed_decay_interval)
        self.speed_decay_factor = float(speed_decay_factor)
        self.initial_speed = float(V)
        self.time = 0.0
        self._last_decay_time = self.boost_duration

        # 当前项目中“恒速 PN 导引”使用的速度
        self.missile_speed = float(missile_speed) if missile_speed is not None else float(target_speed)

    # ------------------------------------------------------------------
    # 1) 推力段 + 速度衰减模型
    # ------------------------------------------------------------------
    def _advance_time(self) -> None:
        """根据 boost + 衰减模型更新 ``self.V`` 和 ``self.time``。"""
        next_time = self.time + self.dt

        # 推力加速段
        if next_time <= self.boost_duration:
            progress = next_time / self.boost_duration
            self.V = self.initial_speed + (self.target_speed - self.initial_speed) * progress
            self._last_decay_time = self.boost_duration
        else:
            # 推力段刚结束时，将速度拉到目标速度
            if self.time < self.boost_duration:
                self.V = self.target_speed

            # 之后按照设定区间进行指数衰减
            elapsed_since_decay = next_time - self._last_decay_time
            if elapsed_since_decay >= self.speed_decay_interval:
                decay_steps = int(elapsed_since_decay / self.speed_decay_interval)
                if decay_steps > 0:
                    self.V *= self.speed_decay_factor ** decay_steps
                    self._last_decay_time += self.speed_decay_interval * decay_steps

        self.time = next_time
        self.V = max(self.V, 200.0)

    # ------------------------------------------------------------------
    # 2) 单枚导弹三维比例导引 + 速度衰减（参考给出的 MissilePosition）
    # ------------------------------------------------------------------
    def MissilePosition(self, aircraft_plist: list, V_t: float, theta_t: float, fea_t: float) -> list:
        """基于比例导引 + 速度衰减的导弹位置更新（单枚导弹接口）。"""
        # 时间与速度演化
        self._advance_time()

        X_m = self.X
        Y_m = self.Y
        Z_m = self.Z

        V_m = self.V
        Heading_m = self.Heading
        Pitch_m = self.Pitch
        g = self.g
        k1 = self.k1
        k2 = self.k2
        dt = self.dt

        X_t, Y_t, Z_t = aircraft_plist

        dX_m = V_m * m.cos(Pitch_m) * m.cos(Heading_m)
        dY_m = V_m * m.sin(Pitch_m)
        dZ_m = -V_m * m.cos(Pitch_m) * m.sin(Heading_m)

        dX_t = V_t * m.cos(theta_t) * m.cos(fea_t)
        dY_t = V_t * m.sin(theta_t)
        dZ_t = -V_t * m.cos(theta_t) * m.sin(fea_t)

        dist = m.sqrt(
            (X_m - X_t) * (X_m - X_t)
            + (Y_m - Y_t) * (Y_m - Y_t)
            + (Z_m - Z_t) * (Z_m - Z_t)
        )
        dist = max(dist, 1e-6)

        dR = (
            (Y_m - Y_t) * (dY_m - dY_t)
            + (Z_m - Z_t) * (dZ_m - dZ_t)
            + (X_m - X_t) * (dX_m - dX_t)
        ) / dist

        dtheta_L = (
            (dY_t - dY_m) * m.sqrt((X_t - X_m) ** 2 + (Z_t - Z_m) ** 2)
            - (Y_t - Y_m)
            * (
                (X_t - X_m) * (dX_t - dX_m)
                + (Z_t - Z_m) * (dZ_t - dZ_m)
            )
            / m.sqrt((X_t - X_m) ** 2 + (Z_t - Z_m) ** 2)
        ) / (
            (X_m - X_t) ** 2 + (Y_m - Y_t) ** 2 + (Z_m - Z_t) ** 2
        )

        ny = k1 * abs(dR) * dtheta_L / g
        dtheta_m = g / max(V_m, 1e-6) * (ny - m.cos(Pitch_m))
        Pitch_m = Pitch_m + dtheta_m * dt

        dfea_L = (
            (dZ_t - dZ_m) * (X_t - X_m)
            - (Z_t - Z_m) * (dX_t - dX_m)
        ) / ((X_t - X_m) ** 2 + (Z_t - Z_m) ** 2)

        nz = k2 * abs(dR) * dfea_L / g
        dfea_m = -g / max(V_m * m.cos(Pitch_m), 1e-6) * nz
        Heading_m = Heading_m + dfea_m * dt

        X_m = X_m + V_m * m.cos(Pitch_m) * m.cos(Heading_m) * dt
        Y_m = Y_m + V_m * m.sin(Pitch_m) * dt
        Z_m = Z_m - V_m * m.cos(Pitch_m) * m.sin(Heading_m) * dt

        self.X = X_m
        self.Y = Y_m
        self.Z = Z_m
        self.Pitch = Pitch_m
        self.Heading = Heading_m

        if self.Heading > m.pi:
            self.Heading = 2 * m.pi - self.Heading
        elif self.Heading < -m.pi:
            self.Heading = 2 * m.pi + self.Heading

        return [X_m, Y_m, Z_m]

    # ------------------------------------------------------------------
    # 3) 当前环境使用的“批量恒速 PN”接口（兼容旧代码）
    # ------------------------------------------------------------------
    def step(
        self,
        missile_pos: np.ndarray,
        missile_vel: np.ndarray,
        blue_pos: np.ndarray,
        blue_vel: np.ndarray,
        nav_gains: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """批量更新所有已发射导弹的位置和速度。"""
        return update_missiles_pn(
            missile_pos,
            missile_vel,
            blue_pos,
            blue_vel,
            self.missile_speed,
            self.dt,
            nav_gains,
        )


class Aircraft:
    """蓝方飞机模型（机动规避）。

    - ``step``：封装原 ``update_blue_state``，供当前强化学习环境使用；
    - ``AircraftPostition``：连续过载控制接口，供后续精细机动仿真使用。
    """

    def __init__(
        self,
        aircraft_plist: Optional[list] = None,
        V: Optional[float] = None,
        Pitch: Optional[float] = None,
        Heading: Optional[float] = None,
        dt: float = 0.01,
        g: float = 9.6,
        MaxV: float = 408.0,
        MinV: float = 170.0,
        accel_mag: float = 0.0,
        v_max: Optional[float] = None,
    ) -> None:
        if aircraft_plist is None:
            aircraft_plist = [0.0, 0.0, 0.0]
        self.X, self.Y, self.Z = aircraft_plist

        self.V = float(V) if V is not None else 0.0
        self.Pitch = float(Pitch) if Pitch is not None else 0.0
        self.Heading = float(Heading) if Heading is not None else 0.0

        self.g = float(g)
        self.dt = float(dt)

        self.MaxV = float(MaxV if v_max is None else v_max)
        self.MinV = float(MinV)

        self.nx = 0.0
        self.ny = 1.0
        self.roll = 0.0

        self.accel_mag = float(accel_mag)
        self.v_max = float(v_max) if v_max is not None else self.MaxV

    # A) 当前环境使用的离散动作更新接口
    def step(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        action: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pos, vel = update_blue_state(
            pos,
            vel,
            int(action),
            dt=self.dt,
            accel_mag=self.accel_mag,
            v_max=self.v_max,
        )
        # 同步内部状态（便于后续使用）
        self.X, self.Y, self.Z = float(pos[0]), float(pos[1]), float(pos[2])
        self.V = float(np.linalg.norm(vel))
        return pos, vel

    # B) 连续机动接口：参考 AircraftPostition
    def AircraftPostition(
        self,
        missile_plist: Optional[list] = None,
        nx: float = 0.0,
        ny: float = 1.0,
        roll: float = 0.0,
        Pitch: float = 0.0,
    ) -> Tuple[float, float, float]:
        if Pitch == -1:
            self.Pitch = self.Pitch
        elif Pitch == 0:
            self.Pitch = 0.0

        self.roll = roll
        self.ny = ny
        self.nx = nx

        if missile_plist is None:
            missile_plist = []

        _V = self.g * (self.nx - m.sin(self.Pitch))
        _Pitch = (self.g / max(self.V, 1e-6)) * (self.ny * m.cos(self.roll) - m.cos(self.Pitch))
        _Heading = self.g * self.ny * m.sin(self.roll) / max(self.V * m.cos(self.Pitch), 1e-6)

        self.V += _V * self.dt
        self.Pitch += _Pitch * self.dt
        self.Heading += _Heading * self.dt

        if self.Heading > m.pi:
            self.Heading = 2 * m.pi - self.Heading
        elif self.Heading < -m.pi:
            self.Heading = 2 * m.pi + self.Heading

        self.X = self.X + self.V * m.cos(self.Pitch) * m.cos(self.Heading) * self.dt
        self.Y = self.Y + self.V * m.sin(self.Pitch) * self.dt
        self.Z = self.Z - self.V * m.cos(self.Pitch) * m.sin(self.Heading) * self.dt

        return self.X, self.Y, self.Z
