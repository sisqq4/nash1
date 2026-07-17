# M2 Dynamics Refactor Status Report

## 1. 修改文件

本阶段提交集中在坐标系、状态结构、数学工具、单位常量、三自由度动力学和测试，不修改 PPO、DQN、奖励或训练循环。

| 文件 | 修改内容 |
| --- | --- |
| `src/air_combat_rl/domain/states.py` | 新增 `FlightState`，显式表达 `[x,z,y,V,gamma,psi]`；保留 `KinematicState = FlightState` 兼容旧调用。 |
| `src/air_combat_rl/core/units.py` | 新增 SI 单位常量、标准重力加速度、速度和 `cos(gamma)` 数值保护阈值。 |
| `src/air_combat_rl/core/math3d.py` | 新增 XZY 运动学、全局/局部坐标变换、数值保护和通用欧拉步进。 |
| `src/air_combat_rl/domain/dynamics/aircraft_3dof.py` | 将飞机传播重构为三自由度点质量模型，按 `[x,z,y,V,gamma,psi]` 返回导数并积分。 |
| `src/air_combat_rl/domain/dynamics/missile_3dof.py` | 新增导弹三自由度模型、切向/法向/侧向过载命令和比例导引命令适配接口。 |
| `src/air_combat_rl/domain/dynamics/integrators.py` | 新增 `integrate_euler()` 作为三自由度状态积分入口。 |
| `tests/unit/test_m2_3dof.py` | 新增 M2 覆盖测试：运动学、坐标变换、飞机/导弹单步积分、边界保护和 SI 单位一致性。 |

## 2. 数学实现

### 2.1 坐标和状态约定

全局坐标和序列化顺序统一为：

```text
[x, z, y, V, gamma, psi]
```

其中：

- `x`：前向；
- `z`：右向；
- `y`：向上/高度；
- `V`：速度，单位 m/s；
- `gamma`：航迹倾角，单位 rad，增大表示爬升；
- `psi`：航向角，单位 rad，增大表示右转。

### 2.2 运动学

`flight_velocity()` 实现：

```text
dx/dt = V cos(gamma) cos(psi)
dz/dt = V cos(gamma) sin(psi)
dy/dt = V sin(gamma)
```

该函数返回 `VecXZY(dx/dt, dz/dt, dy/dt)`。

### 2.3 飞机三自由度动力学

`aircraft_derivative()` 实现：

```text
dV/dt = g * (nx - sin(gamma))

dgamma/dt = g / V * (nf * cos(gamma_s) - cos(gamma))

dpsi/dt = g * nf * sin(gamma_s) / (V * cos(gamma))
```

其中：

- `g = STANDARD_GRAVITY = 9.80665 m/s^2`；
- `safe_speed()` 保护 `V` 接近 0 的分母；
- `safe_cos_gamma()` 保护 `cos(gamma)` 接近 0 的分母。

### 2.4 导弹三自由度动力学

`MissileCommand` 使用三类过载输入：

```text
nx: tangential overload, g units
nn: normal overload, g units
ns: side overload, g units
```

`missile_derivative()` 实现：

```text
dV/dt = g * (nx - sin(gamma))

dgamma/dt = g / V * (nn - cos(gamma))

dpsi/dt = g * ns / (V * cos(gamma))
```

比例导引接口由 `ProportionalNavigationCommand.to_missile_command()` 和 `proportional_navigation_command()` 提供，将闭合速度、视线角速度和导航比转换为 `MissileCommand`。

### 2.5 全局坐标和随航迹运动的局部坐标

`global_to_local()` 和 `local_to_global()` 基于 `gamma`、`psi` 构造随飞行航迹运动的局部坐标基：

```text
forward = [cos(gamma)cos(psi), cos(gamma)sin(psi), sin(gamma)]
right   = [-sin(psi), cos(psi), 0]
up      = [-sin(gamma)cos(psi), -sin(gamma)sin(psi), cos(gamma)]
```

当前实现是函数式坐标变换，不是独立的 `MovingXZYFrame` 或 `AircraftLocalFrame` 类。

### 2.6 数值积分

`euler_step()` 和 `integrate_euler()` 使用显式欧拉积分：

```text
state(t + dt) = state(t) + derivative(state(t)) * dt
```

积分状态只包含三自由度变量 `[x,z,y,V,gamma,psi]`，不包含滚转角速度、姿态角速度、控制力矩或转动惯量。

## 3. 新旧接口映射

| 旧接口/旧语义 | 新接口/新语义 | 兼容性说明 |
| --- | --- | --- |
| `KinematicState` | `FlightState` | `KinematicState = FlightState`，现有调用继续可用。 |
| 手写 `[x,z,y,V,gamma,psi]` 下标 | `FlightState.position/speed/angles` | 核心动力学优先使用具名字段。 |
| `VecXZY` 位置向量 | `VecXZY` 继续使用 | 明确作为全局 XZY 坐标原语。 |
| `G0 = 9.80665` 局部常量 | `STANDARD_GRAVITY` | 重力常量集中到 `core/units.py`。 |
| 飞机传播直接在 `integrate_aircraft()` 内更新速度/角度/位置 | `aircraft_derivative()` + `euler_step()` | 动力学导数和积分器解耦。 |
| `command.ny` / `command.nz` 参与旧角速度计算 | `command.nf` / `command.gamma_s` 参与规范方程 | `ManeuverCommand` 未改名，三自由度方程按 M2 规格使用 `nf` 和 `gamma_s`。 |
| 导弹动力学占位 | `MissileCommand`、`missile_derivative()`、`integrate_missile()` | 新增三自由度导弹模型。 |
| 比例导引占位 | `ProportionalNavigationCommand`、`proportional_navigation_command()` | 新增 PN 到导弹过载命令的适配接口。 |
| 积分器占位 | `integrate_euler()` | 提供统一三自由度积分入口。 |

## 4. 测试命令和结果

已执行以下命令：

```bash
python -m compileall -q src tests
```

结果：通过，无输出，表示 `src` 和 `tests` 下 Python 文件可成功字节编译。

```bash
pytest -q
```

结果：

```text
17 passed in 0.18s
```

新增测试覆盖：

- 水平匀速直线；
- 正 `gamma` 爬升；
- 负 `gamma` 俯冲；
- 正 `psi` 向 `+z` 运动；
- 全局到局部坐标变换；
- 局部到全局逆变换；
- 飞机单步积分；
- 导弹单步积分；
- 数值边界；
- SI 单位一致性。

## 5. 仍存在的差异

1. **局部坐标系尚未封装为独立类。**  当前有 `global_to_local()` / `local_to_global()` 函数，但没有 `MovingXZYFrame`、`AircraftLocalFrame` 等显式数据结构。

2. **`tasks/blue_escape/local_frame.py` 仍是占位文件。**  本阶段未修改任务层观测、奖励或训练循环，因此任务层局部坐标特征尚未接入。

3. **导弹比例导引接口是基础适配。**  当前 PN 接口提供从 LOS rate 到 `MissileCommand` 的转换，但未接入完整 seeker、锁定状态、推力/阻力或制导闭环调度。

4. **积分器为显式欧拉。**  当前满足 M2 单步积分需求，但没有实现 RK4、自适应步长或误差控制。

5. **现有配置和高层任务逻辑未迁移。**  本阶段按要求不修改 PPO、DQN、奖励和训练循环，因此高层模块可能仍有后续阶段需要统一语义的地方。

## 6. git diff 摘要

基于最新 M2 提交的 diff stat：

```text
src/air_combat_rl/core/math3d.py                   | 62 ++++++++++++++-
src/air_combat_rl/core/units.py                    | 11 ++-
src/air_combat_rl/domain/dynamics/aircraft_3dof.py | 33 ++++----
src/air_combat_rl/domain/dynamics/integrators.py   | 11 ++-
src/air_combat_rl/domain/dynamics/missile_3dof.py  | 69 ++++++++++++++++-
src/air_combat_rl/domain/states.py                 | 20 ++++-
tests/unit/test_m2_3dof.py                         | 88 ++++++++++++++++++++++
7 files changed, 275 insertions(+), 19 deletions(-)
```
