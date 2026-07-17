# M3 Maneuvers Refactor Status Report

## 1. 修改文件

本阶段聚焦蓝方 29 种离散机动动作库、动作到三自由度过载指令的映射、平台动作掩码、动作保持机制和对应测试。不修改环境奖励、PPO/智能体、训练循环或动力学方程本身。

| 文件 | 修改内容 |
| --- | --- |
| `configs/actions/blue_29.yaml` | 重建蓝方 29 动作表为 `blue_29/v2`；保持稳定 `action_id` 0..28；为每个动作显式配置 `name`、`nx`、`nf`、`gamma_s_deg`、三类预期运动效果和 `valid_platforms`。 |
| `src/air_combat_rl/tasks/blue_escape/action_catalog.py` | 重构 `BlueAction` 数据结构和 `ActionCatalog` 加载/校验逻辑；新增动作方向语义校验、平台 `action_mask()`、非法动作安全 fallback。 |
| `src/air_combat_rl/tasks/blue_escape/action_hold.py` | 新增 `HeldAction`，用于把 PPO 速率动作选择保持为一个 policy interval 内的固定物理命令。 |
| `src/air_combat_rl/tasks/blue_escape/environment.py` | 在 `BlueEscapeEnv.step()` 中接入 `HeldAction.select()`，保持 reward、done、observation 和 PPO 接口不变。 |
| `docs/refactor/decisions/ADR-003-maneuver-signs.md` | 新增方向符号决策记录，说明 `gamma_s` 符号按 M2 三自由度导数语义修正，而不是机械复制源表格符号。 |
| `tests/unit/test_m3_blue_actions.py` | 新增 M3 单元测试，覆盖编号稳定性、方向符号、复合动作、平台掩码、fallback 和动作保持。 |
| `tests/contracts/test_layer_contracts.py` | 更新动作目录契约到 `blue_29/v2`，并检查左转动作 `gamma_s` 符号。 |

## 2. 数学实现

### 2.1 状态和动力学约定

M3 继续沿用 M2 的三自由度飞机状态和坐标约定：

```text
[x, z, y, V, gamma, psi]
```

其中：

- `gamma` 增大表示爬升；
- `psi` 增大表示右转；
- `x,z,y` 仍按 XZY 序列化，`y` 是高度。

动作不直接改变状态，而是映射为 `ManeuverCommand(nx, nf, gamma_s)`，并由 M2 已实现的飞机三自由度方程解释：

```text
dV/dt = g * (nx - sin(gamma))

dgamma/dt = g / V * (nf * cos(gamma_s) - cos(gamma))

dpsi/dt = g * nf * sin(gamma_s) / (V * cos(gamma))
```

在水平初始状态 `gamma = 0` 下，方向判断简化为：

```text
sign(dV/dt)      = sign(nx)
sign(dgamma/dt) = sign(nf * cos(gamma_s) - 1)
sign(dpsi/dt)   = sign(nf * sin(gamma_s))
```

### 2.2 29 个动作的稳定顺序

动作表按用户给定表格保留 29 类动作，使用 0-based 稳定编号：

| action_id 范围 | 动作类别 |
| --- | --- |
| `0` | 匀速前飞 |
| `1..2` | 两档加速前飞 |
| `3..4` | 两档减速前飞 |
| `5..7` | 三档左转 |
| `8..10` | 三档右转 |
| `11..13` | 三档爬升 |
| `14..16` | 三档俯冲 |
| `17..19` | 三档左爬升 |
| `20..22` | 三档右爬升 |
| `23..25` | 三档左俯冲 |
| `26..28` | 三档右俯冲 |

`ActionCatalog` 强制动作数量必须为 29，且动作 id 集合必须精确为 `0..28`。

### 2.3 方向符号修正

M3 未直接复制文档表格中的 `gamma_s` 符号，而是按统一动力学语义设置：

- 左转：`nf * sin(gamma_s) < 0`，因此 `dpsi/dt < 0`；
- 右转：`nf * sin(gamma_s) > 0`，因此 `dpsi/dt > 0`；
- 爬升：`nf * cos(gamma_s) > 1`，因此水平状态下 `dgamma/dt > 0`；
- 俯冲：`nf * cos(gamma_s) < 1`，因此水平状态下 `dgamma/dt < 0`。

复合动作使用象限选择同时满足两类符号：

- 左爬升：`gamma_s < 0` 且 `cos(gamma_s) > 0`；
- 右爬升：`gamma_s > 0` 且 `cos(gamma_s) > 0`；
- 左俯冲：`gamma_s < 0` 且 `cos(gamma_s) < 0`；
- 右俯冲：`gamma_s > 0` 且 `cos(gamma_s) < 0`。

纯水平转弯使用：

```text
gamma_s = ±acos(1 / nf)
```

使 `nf * cos(gamma_s) - 1 ≈ 0`，在水平初始状态下基本保持 `dgamma/dt = 0`，同时通过 `sin(gamma_s)` 的正负控制左右转。

### 2.4 平台动作掩码和过载限制

平台限制通过动作表中的 `valid_platforms` 表达：

- `zdj` 最大过载约 9g，当前动作表中 29 个动作均可用；
- `yjj` 最大过载约 3g，仅允许 `max(abs(nx), abs(nf)) <= 3` 的动作。

运行时通过 `ActionCatalog.action_mask(platform)` 返回长度为 29 的布尔掩码。`ActionCatalog.command_for(action_id, platform)` 会先检查掩码：

- 若请求动作允许，返回该动作对应的 `ManeuverCommand`；
- 若请求动作不允许，回退到 `SAFE_FALLBACK_ACTION_ID = 0` 的匀速前飞动作；
- 若某个平台所有动作都被屏蔽，`action_mask()` 会强制只开放 fallback 动作 0，避免无动作可选。

当前实现是离散动作层面的“屏蔽 + fallback”，不是连续控制层面的过载裁剪。也就是说，`yjj` 请求 9g 动作时不会把 9g 裁剪成 3g，而是执行动作 0。

### 2.5 动作保持机制

`HeldAction` 实现 PPO 速率动作保持：

```text
policy_dt = 0.1 s
physics_dt = 0.005 s
substeps_per_policy_step = 20
```

每次策略选择动作时，`HeldAction.select()` 解析平台掩码并保存选中的 `ManeuverCommand`，同时把 `remaining_substeps` 设为 20。`consume_substep()` 在物理子步层面返回同一个保持命令并递减计数。

本阶段只实现接口和保持状态；没有实现 PPO，也没有修改奖励和智能体逻辑。

## 3. 新旧接口映射

| 旧接口/旧语义 | 新接口/新语义 | 兼容性说明 |
| --- | --- | --- |
| `configs/actions/blue_29.yaml` 中旧 `id` 字段 | `action_id` | `from_yaml()` 仍兼容读取旧 `id`，但 M3 配置使用 `action_id`。 |
| 旧动作名按 left/straight/right 与 nx 网格组织 | 新动作名按用户表格 29 类动作组织 | 动作编号改为用户表格顺序的 0-based 稳定编号。 |
| `expected_turn` | `expected_lateral_effect` | 统一为 `left` / `right` / `straight`，并按 `dpsi/dt` 符号校验。 |
| `expected_vertical_motion` | `expected_vertical_effect` | 统一为 `climb` / `dive` / `level`，并按 `dgamma/dt` 符号校验。 |
| 无显式纵向效果字段 | `expected_longitudinal_effect` | 新增 `accelerate` / `decelerate` / `maintain`，并按 `nx` 与 `dV/dt` 语义校验。 |
| `allowed_platforms` | `valid_platforms` | 字段改名，语义保持为动作可用于哪些平台。 |
| 非法平台动作直接抛 `ValueError` | 非法平台动作 fallback 到安全动作 0 | 避免策略选择被平台 mask 屏蔽的动作后仍执行超限过载。 |
| 无显式动作 mask | `ActionCatalog.action_mask(platform)` | PPO 可通过标准 action mask 接口获取平台可用动作。 |
| 环境直接 `command_for(action_id, platform)` | 环境先 `HeldAction.select(...)` | 接入动作保持机制，但 `BlueEscapeEnv.step()` 的外部接口仍接收离散 `action_id`。 |
| 无动作保持状态 | `HeldAction` | 保存当前动作、当前命令和剩余物理子步数。 |

## 4. 测试命令和结果

已执行以下命令：

```bash
git diff --check
```

结果：通过，无输出，表示当前 diff 无空白或格式错误。

```bash
pytest -q
```

结果：

```text
26 passed in 0.16s
```

```bash
python -m compileall -q src tests
```

结果：通过，无输出，表示 `src` 和 `tests` 下 Python 文件可成功字节编译。

新增和更新测试覆盖：

- 29 个动作编号稳定；
- 首尾动作名称稳定；
- 加速/减速与 `dV/dt` 符号一致；
- 左/右转与 `dpsi/dt` 符号一致；
- 爬升/俯冲与 `dgamma/dt` 符号一致；
- 左爬升、右爬升、左俯冲、右俯冲同时满足横向和垂向符号；
- `zdj` / `yjj` 动作掩码符合 9g / 3g 限制；
- 未知平台或无可用动作时 fallback 到安全匀速动作；
- PPO 0.1s 动作选择可保持 20 个 0.005s 物理子步。

## 5. 仍存在的差异

1. **动作编号采用 0-based 编号。** 用户文档表格序号为 1..29，代码中 `action_id` 保持项目既有离散动作习惯为 0..28；二者按顺序一一对应。

2. **平台过载限制是动作级屏蔽，不是连续裁剪。** 对 `yjj` 请求 6g/9g 动作时，当前实现 fallback 到动作 0；没有把过载裁剪到 3g 后继续执行原动作。

3. **动作保持接口已实现，但世界积分仍接收 policy interval 级命令。** `SimulationWorld.step_policy_interval()` 本身已经在 20 个物理子步中使用同一个命令积分；`HeldAction.consume_substep()` 作为显式保持接口和测试对象存在，但尚未进一步下沉到 world 内部逐子步调用。

4. **动作表方向校验基于水平初始状态语义。** 对爬升/俯冲方向的静态校验使用 `nf * cos(gamma_s) - 1`，对应 `gamma=0`；实际飞行中 `cos(gamma)` 会随状态变化。

5. **旧字段兼容只保留在简易 YAML loader 层。** `from_yaml()` 可识别旧 `id`，但新配置和测试均以 `action_id`、`expected_*_effect`、`valid_platforms` 为准。

## 6. git diff 摘要

基于 M3 提交 `7219622 Implement M3 blue maneuver action catalog` 的 diff stat：

```text
configs/actions/blue_29.yaml                       | 479 +++++++++++----------
docs/refactor/decisions/ADR-003-maneuver-signs.md  |  31 ++
src/air_combat_rl/tasks/blue_escape/action_catalog.py            |  92 ++--
src/air_combat_rl/tasks/blue_escape/action_hold.py |  27 ++
src/air_combat_rl/tasks/blue_escape/environment.py |   4 +-
tests/contracts/test_layer_contracts.py            |   4 +-
tests/unit/test_m3_blue_actions.py                 |  96 +++++
7 files changed, 477 insertions(+), 256 deletions(-)
```
