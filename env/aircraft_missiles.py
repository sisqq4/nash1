import math as m
from typing import Optional

MinV = 170  # 飞机最小速度，和你给的参考代码保持一致


class Missiles:
    """
    单枚导弹对象：
    - 初始化一次即代表场景中的一发导弹
    - 负责自身速度随时间的两段式变化（加速+衰减）
    - 负责基于比例导引的轨迹更新（MissilePosition）
    """

    def __init__(
        self,
        missile_plist: list,
        V: float,
        Pitch: float,
        Heading: float,
        dt: float = 0.01,
        g: float = 9.6,
        k1: float = 7.0,
        k2: float = 7.0,
        target_speed: float = 1200.0,
        boost_duration: float = 5.0,
        speed_decay_interval: float = 1.0,
        speed_decay_factor: float = 0.99,
        target_id: Optional[int] = None,
    ):
        # 位置 / 姿态
        self.X, self.Y, self.Z = missile_plist  # 导弹坐标
        self.V = float(V)  # 当前速度
        self.Pitch = float(Pitch)  # 俯仰角
        self.Heading = float(Heading)  # 偏航角

        self.attacking = False  # 是否在飞行（已发射）

        # 常数参数
        self.g = float(g)
        self.dt = float(dt)
        self.k1 = float(k1)
        self.k2 = float(k2)

        # 两段速度模型参数（助推+滑行衰减）
        self.target_speed = float(target_speed)
        self.boost_duration = float(boost_duration)
        self.speed_decay_interval = float(speed_decay_interval)
        self.speed_decay_factor = float(speed_decay_factor)
        self.initial_speed = float(V)

        self.time = 0.0
        self._last_decay_time = self.boost_duration
        self.target_id = target_id

    # ----------------- 速度演化（加速+衰减） -----------------

    def _advance_time(self) -> None:
        """
        根据时间推进导弹速度（助推段线性加速 + 巡航段指数衰减）。
        """
        next_time = self.time + self.dt

        if next_time <= self.boost_duration:
            # 助推段：从 initial_speed 线性加到 target_speed
            progress = next_time / self.boost_duration
            self.V = self.initial_speed + (self.target_speed - self.initial_speed) * progress
            self._last_decay_time = self.boost_duration
        else:
            # 巡航 / 衰减段
            if self.time < self.boost_duration:
                # 第一次进入衰减段，先拉到 target_speed
                self.V = self.target_speed

            elapsed_since_decay = next_time - self._last_decay_time
            if elapsed_since_decay >= self.speed_decay_interval:
                decay_steps = int(elapsed_since_decay / self.speed_decay_interval)
                if decay_steps > 0:
                    self.V *= self.speed_decay_factor**decay_steps
                    self._last_decay_time += self.speed_decay_interval * decay_steps

        self.time = next_time
        # 限制最低速度，避免数值问题
        self.V = max(self.V, 200.0)

    # ----------------- 比例导引 + 位置更新 -----------------

    def MissilePosition(self, aircraft_plist: list, V_t: float, theta_t: float, fea_t: float):
        """
        导弹位置更新函数（比例导引 + 自身速度两段式变化）

        Parameters
        ----------
        aircraft_plist : list [X_t, Y_t, Z_t]
            目标机当前坐标
        V_t : float
            目标机速度标量
        theta_t : float
            目标机俯仰角
        fea_t : float
            目标机偏航角

        Returns
        -------
        [X_m, Y_m, Z_m] : list
            更新后的导弹坐标
        """
        # 更新时间 & 速度
        self._advance_time()

        # 当前导弹状态
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

        # 目标机位置
        X_t, Y_t, Z_t = aircraft_plist

        # 导弹速度分量
        dX_m = V_m * m.cos(Pitch_m) * m.cos(Heading_m)
        dY_m = V_m * m.sin(Pitch_m)
        dZ_m = -V_m * m.cos(Pitch_m) * m.sin(Heading_m)

        # 目标机速度分量
        dX_t = V_t * m.cos(theta_t) * m.cos(fea_t)
        dY_t = V_t * m.sin(theta_t)
        dZ_t = -V_t * m.cos(theta_t) * m.sin(fea_t)

        # 相对距离
        dist = m.sqrt((X_m - X_t) ** 2 + (Y_m - Y_t) ** 2 + (Z_m - Z_t) ** 2)
        # 防止除 0
        if dist < 1e-6:
            dist = 1e-6

        # 视线距离变化率 dR
        dR = (
            (Y_m - Y_t) * (dY_m - dY_t)
            + (Z_m - Z_t) * (dZ_m - dZ_t)
            + (X_m - X_t) * (dX_m - dX_t)
        ) / dist

        # 视线俯仰角变化率 dtheta_L
        hor_dist = m.sqrt((X_t - X_m) ** 2 + (Z_t - Z_m) ** 2)
        if hor_dist < 1e-6:
            hor_dist = 1e-6

        dtheta_L = (
            (dY_t - dY_m) * hor_dist
            - (Y_t - Y_m)
            * ((X_t - X_m) * (dX_t - dX_m) + (Z_t - Z_m) * (dZ_t - dZ_m))
            / hor_dist
        ) / ((X_m - X_t) ** 2 + (Y_m - Y_t) ** 2 + (Z_m - Z_t) ** 2)

        ny = k1 * abs(dR) * dtheta_L / g
        dtheta_m = g / V_m * (ny - m.cos(Pitch_m))
        Pitch_m = Pitch_m + dtheta_m * dt

        # 视线方位角变化率 dfea_L
        dfea_L = ((dZ_t - dZ_m) * (X_t - X_m) - (Z_t - Z_m) * (dX_t - dX_m)) / (
            (X_t - X_m) ** 2 + (Z_t - Z_m) ** 2
        )

        nz = k2 * abs(dR) * dfea_L / g
        dfea_m = -g / (V_m * m.cos(Pitch_m)) * nz
        Heading_m = Heading_m + dfea_m * dt

        # 位置积分
        X_m = X_m + V_m * m.cos(Pitch_m) * m.cos(Heading_m) * dt
        Y_m = Y_m + V_m * m.sin(Pitch_m) * dt
        Z_m = Z_m - V_m * m.cos(Pitch_m) * m.sin(Heading_m) * dt

        # 写回状态
        self.X = X_m
        self.Y = Y_m
        self.Z = Z_m
        self.Pitch = Pitch_m
        self.Heading = Heading_m

        # Heading 限幅到 [-pi, pi]
        if self.Heading > m.pi:
            self.Heading = self.Heading - 2 * m.pi
        elif self.Heading < -m.pi:
            self.Heading = self.Heading + 2 * m.pi

        return [X_m, Y_m, Z_m]


class Aircraft:
    """
    单架飞机对象：
    - 初始化一次代表场景中的一架蓝机
    - 使用 nx, ny, roll 控制机动，AircraftPostition 用于更新位置
    """

    def __init__(
        self,
        aircraft_plist: list,
        V: float,
        Pitch: float,
        Heading: float,
        dt: float = 0.01,
        g: float = 9.6,
        MaxV: float = 408.0,
    ):
        self.X, self.Y, self.Z = aircraft_plist  # 飞机坐标
        self.V = float(V)
        self.Pitch = float(Pitch)
        self.Heading = float(Heading)

        self.MaxV = float(MaxV)
        self.MinV = float(MinV)

        # 环境参数
        self.g = float(g)
        self.dt = float(dt)

        # 飞机自身参数
        self.nx = 0.0  # 切向过载
        self.ny = 1.0  # 法向过载
        self.roll = 0.0  # 当前滚转角

    # ----------------- 约束检查（可按需使用） -----------------

    def action_constraint(self, PitchConstraint: int) -> bool:
        """
        动作约束：当 PitchConstraint == 0 时要求在当前俯仰角为 0 的水平面内机动。
        """
        constraint_flag = True
        aPitch = self.Pitch
        if PitchConstraint == 0 and aPitch != 0:
            constraint_flag = False

        if not (-45 * m.pi / 180 <= self.Pitch <= 60 * m.pi / 180):
            constraint_flag = False

        return constraint_flag

    def speed_constraint(self, nx: float) -> bool:
        """
        简单速度约束：超速时不再允许继续加速；低速时不允许减速。
        """
        constraint_flag = True

        if self.V >= self.MaxV and nx > 0:
            constraint_flag = False

        if self.V < self.MinV and nx < 0:
            constraint_flag = False

        return constraint_flag

    # ----------------- 位置 / 姿态更新 -----------------

    def AircraftPostition(
        self,
        missile_plist=None,
        nx: float = 0.0,
        ny: float = 1.0,
        roll: float = 0.0,
        Pitch: float = 0.0,
    ):
        """
        飞机位置计算函数：基于当前 nx, ny, roll, Pitch 指令更新飞机状态。

        Parameters
        ----------
        missile_plist : list, optional
            预留参数，目前不使用
        nx : float
            切向过载
        ny : float
            法向过载
        roll : float
            滚转角 [rad]
        Pitch : float
            = -1  : 不改变当前俯仰角
            = 0   : 强制置为 0（水平）
        """
        if missile_plist is None:
            missile_plist = []

        # Pitch 指令
        if Pitch == -1:
            # 保持当前俯仰角
            pass
        elif Pitch == 0:
            # 强制水平
            self.Pitch = 0.0

        self.roll = roll
        self.ny = ny
        self.nx = nx

        # 速度 / 角速度
        _V = self.g * (self.nx - m.sin(self.Pitch))  # 速度标量加速度
        _Pitch = (self.g / self.V) * (self.ny * m.cos(self.roll) - m.cos(self.Pitch))  # 俯仰角速度
        _Heading = self.g * self.ny * m.sin(self.roll) / (self.V * m.cos(self.Pitch))  # 偏航角速度

        # 状态积分
        self.V += _V * self.dt
        self.Pitch += _Pitch * self.dt
        self.Heading += _Heading * self.dt

        # Heading 限幅
        if self.Heading > m.pi:
            self.Heading = self.Heading - 2 * m.pi
        elif self.Heading < -m.pi:
            self.Heading = self.Heading + 2 * m.pi

        self.X = self.X + self.V * m.cos(self.Pitch) * m.cos(self.Heading) * self.dt
        self.Y = self.Y + self.V * m.sin(self.Pitch) * self.dt
        self.Z = self.Z - self.V * m.cos(self.Pitch) * m.sin(self.Heading) * self.dt

        return self.X, self.Y, self.Z
