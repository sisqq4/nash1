# M4 Environment Refactor Status Report

## 0. 本轮复核结论

针对上一轮提出的 4 个重点问题，当前代码状态如下：

| 问题 | 结论 | 依据 |
| --- | --- | --- |
| 终局奖励和塑形奖励累计尺度不合理 | 已修正主要风险 | 终局奖励/惩罚与规则机验证方案对齐为 `success/exhausted=+10`、`hit=-10`、`crash=-20`、`timeout=0`；为避免塑形累计过大，非终局塑形统一乘以 `shaping_scale=0.1`。 |
| `success` 没有真实产生路径 | 已补充 | `BlueEscapeEnv` 现在会在无威胁或所有存活锁定威胁安全越过后返回 `success`。 |
| 动作保持接口没有完全下沉到物理子步消费逻辑 | 已补充 | `BlueEscapeEnv.step()` 选择 `HeldAction` 后调用 `SimulationWorld.step_held_policy_interval()`，world 每个物理子步调用 `held_action.consume_substep()`。 |
| closest approach 直接让导弹失效，可能过早触发 `exhausted` | 已修正 | closest approach 现在只记录 `closest_approach_passed` 并发事件，不再将导弹 `alive=False`；安全成功还要求安全距离和非闭合状态。 |

## 1. 修改文件

M4 变更集中在蓝方逃逸环境、场景、观测、奖励、世界子步推进和测试。未实现 PPO。

| 文件 | 修改内容 |
| --- | --- |
| `configs/scenario/fixed_1v1.yaml` | 新增固定 1v1 场景配置，显式配置 `mode`、最大时间和物理/策略时间尺度。 |
| `configs/scenario/randomized_1v1.yaml` | 新增随机 1v1 场景配置，包含随机种子、蓝方高度范围 8–12 km 和时间尺度。 |
| `configs/scenario/high_threat_1v1.yaml` | 新增高威胁 1v1 场景配置。 |
| `configs/scenario/1v2.yaml` | 新增 1v2 多弹场景配置。 |
| `configs/scenario/1vN.yaml` | 新增 1vN 多弹场景配置。 |
| `configs/scenario/powered_launch_1v1.yaml` | 补充 `mode: powered_launch`。 |
| `configs/scenario/powered_launch_1vn.yaml` | 补充 `mode: powered_launch` 和多弹数量。 |
| `configs/scenario/terminal_intercept_1v1.yaml` | 补充 `mode: terminal_intercept`。 |
| `configs/scenario/terminal_intercept_1vn.yaml` | 补充 `mode: terminal_intercept` 和多弹数量。 |
| `src/air_combat_rl/simulation/scenarios/factory.py` | 新增 `ScenarioConfig` 与 `build_scenario()`，按配置创建 fixed/randomized/high_threat/1v2/1vN/powered_launch/terminal_intercept 初始世界。 |
| `src/air_combat_rl/simulation/world.py` | 新增 `WorldConfig`，在 policy interval 内推进物理子步，集成蓝机、导弹、比例导引、动力段、失速、超时、命中、撞地、最近点事件和安全成功判定。 |
| `src/air_combat_rl/tasks/blue_escape/environment.py` | 环境 step 接入 `HeldAction` 子步消费、观测构建、奖励计算、`terminated`/`truncated` 语义和 info 字段。 |
| `src/air_combat_rl/tasks/blue_escape/observation_builder.py` | 实现固定维度观测、导弹槽位 padding、确定性距离排序和 `missile_mask`。 |
| `src/air_combat_rl/tasks/blue_escape/rewards/components.py` | 将强化学习奖励函数重构为与规则机验证方案一致的 1v1/1vN 分段奖励：终局奖励、近距/远距几何塑形、威胁度变化、低空风险、多弹包围惩罚。 |
| `tests/unit/test_m4_blue_escape_env.py` | 新增 M4 单元测试，覆盖 reset、20 子步、子步命中、y 高度撞地、观测 padding/mask、powered/terminal 场景、奖励方向、终止语义、动作保持消费和 closest approach 非失效语义。 |
| `src/air_combat_rl/controllers/baselines/blue_behavior_tree.py` | 将规则机基线从占位实现为低空安全、低速保护、主威胁识别、巡航、横切规避、近距急机动和状态驻留策略；新增 `act(observation, action_mask)`，使规则机可按强化学习策略同形接口调用。 |
| `tests/unit/test_m4_behavior_tree_reward_alignment.py` | 新增规则机与奖励对齐测试，验证安全/速度优先级、beam/break 切换、终局奖励数值，以及规则机 `act(observation, action_mask)` 与 RL 观测/动作掩码接口对齐。 |

## 2. 数学实现

### 2.1 时间尺度与动作保持

环境时间尺度仍由 `SimulationClock` 定义：

```text
physics_dt = 0.005 s
policy_dt  = 0.1 s
action_repeat = policy_dt / physics_dt = 20
```

`SimulationClock.__post_init__()` 强制 `policy_dt / physics_dt` 必须为整数，`substeps_per_policy_step` 返回四舍五入后的整数子步数。

当前 step 逻辑为：

1. `BlueEscapeEnv.step(action_id)` 接收离散动作；
2. `HeldAction.select()` 将动作映射为 `ManeuverCommand` 并将 `remaining_substeps` 设为 20；
3. `SimulationWorld.step_held_policy_interval()` 进入最多 20 次物理子步循环；
4. 每个物理子步调用 `held_action.consume_substep()`，得到同一个保持命令并递减剩余子步；
5. 子步内更新蓝机、导弹、制导、动力段和事件；
6. 命中或蓝机撞地会提前结束 policy interval。

因此，正常情况下每个策略 step 会运行 20 个物理子步；如果子步内发生命中或蓝机撞地，则少于 20 步并提前返回。

### 2.2 世界子步更新

每个物理子步执行：

```text
blue(t + dt) = integrate_aircraft(blue(t), held_command, physics_dt)
```

对每枚 alive 导弹：

```text
age(t + dt) = age(t) + physics_dt
mcmd = proportional_navigation_command(missile, blue, N)
```

若导弹仍处于 powered boost 时间窗：

```text
mcmd.nx = missile_boost_nx_g
```

随后：

```text
missile(t + dt) = integrate_missile(missile(t), mcmd, physics_dt)
```

子步内检查：

- 导弹撞地：`kin.position.y <= 0`；
- 导弹失速：`kin.speed < missile_stall_speed_mps`；
- 导弹制导超时：`age >= max_guidance_time_s`；
- 命中：`distance <= kill_radius_m`；
- 最近点越过：`distance > previous_min_distance + closest_approach_event_threshold_m`；
- 蓝机撞地：`blue.position.y <= 0`。

最近点越过不再让导弹失效，只设置 `closest_approach_passed[i] = True` 并发出 `closest_approach_passed` 事件。

### 2.3 成功语义

`success` 的真实路径为：

```text
无导弹威胁
OR
所有 alive and locked 威胁均满足：
  - 已越过最近点；
  - 当前距离 >= success_distance_m；
  - closing_speed <= 0
```

`exhausted` 仍表示存在导弹列表，但已经没有任何 `alive and locked` 威胁。

### 2.4 观测

蓝方观测当前包含 8 个自机特征：

```text
x_norm, z_norm, y_norm, y_norm, V_norm, gamma_norm, psi_norm, current_action_norm
```

导弹槽每枚包含 11 个特征：

```text
relative_position_local_xzy_norm,
relative_velocity_local_xzy_norm,
distance_norm,
closing_speed_norm,
bearing_norm,
elevation_norm,
threat_summary
```

多弹观测采用：

- 固定 `M_max` 槽位；
- 当前距离升序排序；
- padding 为 0；
- 输出 `missile_mask`；
- 观测维度固定为 `8 + M_max * 11`。

### 2.5 奖励

奖励保持模块化分解：

```text
R = R_terminal
  + R_separation
  + R_threat
  + R_height
  + R_encirclement
  + R_ground
  + R_smooth
```

当前奖励已按规则机验证方案重构为 1v1 分段奖励和 1vN 多弹协同威胁奖励。终局奖励先判定，若 episode 已终止，则只返回终局奖励并将塑形项置 0：

```text
crash / ground = -20
hit            = -10
success        = +10
exhausted      = +10
timeout        = 0
```

1v1 非终局奖励先选择主威胁，再依据最近导弹距离和方位角进入三类模式：

```text
d < 8 km                    -> short-range mode
d >= 8 km and |bearing|<=30° -> mid/small-bearing mode
d >= 8 km and |bearing|>30°  -> mid/large-bearing mode
```

近距离模式实现：

```text
R_short = 1.0 * Δd
        + 0.4 * Sroll_proxy
        + 0.6 * Sturn
        + 0.3 * Sv
        + 0.4 * Sh
```

远距离小方位角模式实现：

```text
R_mid_small = 1.0 * Saz
            + 0.8 * Sh
            + 0.8 * Sopp
            + 0.4 * Sv
            - 0.4 * Plevel
```

远距离大方位角模式实现：

```text
R_mid_large = 1.0 * Δd
            + 1.0 * Saz
            + 0.8 * Sh
            + 0.4 * Sv
            - 0.4 * Plevel
            + 0.6 * Srz
```

威胁变化奖惩实现：

```text
if T_t <= T_(t-1): R_threat = +0.25 * T_t
else:              R_threat = -0.25 * (T_t - T_(t-1))
if T_t > 0.6:      R_threat -= 1.5 * (T_t - 0.6)
```

1vN 多弹奖励实现：

```text
R_multi = Rdist + Rh + Rth + Renc + Rground
Rdist   = 1.0 * (mean(d_t) - mean(d_(t-1))) / d_ref
Rh      = 0.5 * H(y)
Rth     = 0.8 * (T_(t-1) - T_t), if threat decreases
        = -1.2 * (T_t - T_(t-1)), if threat increases
Renc    = -1.0 * Cenc
Rground = -1.0 * Pground(y)
```

单弹威胁度采用 logistic 压缩，输入包括距离倒数、闭合速度、剩余拦截时间、视线角速度和导弹剩余速度能力：

```text
T_i = sigmoid(
    1.0 * 1 / r_i
  + 1.2 * max(0, -r_dot_i)
  + 1.1 * 1 / tgo_i
  + 0.6 * |q_i|
  + 0.8 * ξ_M,i
)
```

多弹威胁度取 active missiles 的均值。多弹包围惩罚为：

```text
Cenc = 0.4 * Cang + 0.35 * Csyn + 0.25 * Ccor
```

其中高度和低空风险均只使用 XZY 坐标中的 `y`。距离奖励仍使用差分距离，不使用正的 `1/d` 作为逃逸距离奖励。威胁度内部包含 `1/r` 作为危险度输入，但其作用是威胁变化奖惩，不是直接的正向逃逸奖励。

### 2.6 规则机基线策略

规则机基线实现为优先级有限状态机，并同时提供 snapshot 调用和 RL 同形调用两种入口。推荐后续训练/测试使用 `act(observation, action_mask)`，与学习型策略保持同一形状；`select_action(snapshot)` 保留为调试和真值规则验证入口。

规则机核心决策为：

```text
a_t = a_safe,  if low-altitude risk
    = a_speed, if low-speed risk
    = π_FSM(S_t, T_t), otherwise
```

优先级从高到低为：

1. 低空安全保护：当前高度低、下降率为负、预测高度低于阈值或低于硬最低高度时，输出爬升动作；
2. 低速保护：速度低于规则阈值时，高度充足则俯冲换速，高度不足则加速前飞；
3. 主威胁识别：从 alive、locked 且有效距离内的导弹中选择最近者；
4. 威胁等级：无/低威胁进入 cruise，中距威胁进入 beam，近距威胁进入 break；
5. 状态驻留：beam 和 break 状态具有 dwell step，避免阈值附近频繁切换；
6. 巡航保持：优先恢复速度、俯仰和航向，最后保持前飞；
7. 横切规避：在两条垂直于主威胁来向的候选航向中选择转向代价较小者；
8. 近距急机动：随机选择左右规避方向；高度充足时用左/右俯冲，低高度时使用水平转弯或爬升。

`act(observation, action_mask)` 会从固定维度蓝方逃逸观测中解析高度、速度、gamma、psi，以及每个导弹槽的距离、闭合速度、方位和 threat summary，选择主威胁后输出同一离散 `action_id`。若规则动作被 `action_mask` 屏蔽，则回退到第一个可用动作；若没有可用动作，则回退到前飞动作 0。


## 3. 新旧接口映射

| 旧接口/旧语义 | 新接口/新语义 | 兼容性说明 |
| --- | --- | --- |
| `SimulationWorld.step_policy_interval(command)` 在一个 policy interval 内重复使用 command | `SimulationWorld.step_held_policy_interval(held_action)` 在每个物理子步调用 `consume_substep()` | 保留旧 `step_policy_interval(command)` 兼容已有集成测试；环境层改用 held-action 子步消费接口。 |
| `HeldAction.select()` 只设置保持状态但 world 不消费 | `HeldAction.select()` + `step_held_policy_interval()` + `consume_substep()` | 动作保持计数现在随物理子步递减，测试检查 20 子步后剩余计数为 0。 |
| closest approach 直接 `alive=False` | closest approach 只发事件并记录 `closest_approach_passed` | 不再因最近点越过立即触发所有威胁失效。 |
| `success` 只在 terminated 集合中出现但没有赋值路径 | `all_live_threats_safely_passed()` + 环境 outcome 判定 | 无威胁或存活锁定威胁安全越过后可产生 `success`。 |
| `success` 与 `exhausted` 共用同一终局奖励 | `success_reward` 与 `exhausted_reward` 分离 | 成功脱离和威胁失效具有不同正奖励。 |
| timeout 给正奖励 | timeout reward 为 0 | 避免把 time-limit truncation 当作可刷的成功奖励；终局 crash/hit/exhausted/success 数值与规则机验证方案保持一致。 |
| 奖励为简单 distance/threat/height 塑形 | 奖励改为规则机验证方案中的 1v1 分段模式与 1vN 多弹协同模式 | 保留 `reward_components` 输出，但 separation 承载分段模式主奖励，threat/height/encirclement/ground/smooth 独立输出。 |
| 规则机基线为空占位 | `BlueBehaviorTreePolicy` + `act(observation, action_mask)` | 新增低空安全、低速保护、主威胁、cruise/beam/break、状态驻留和随机急机动方向；规则机现在可像强化学习策略一样接收 observation/action_mask 并输出 action_id。 |

## 4. 测试命令和结果

已执行以下命令：

```bash
pytest -q
```

结果：

```text
pytest 当前环境缺少 numpy，测试收集阶段失败：ModuleNotFoundError: No module named 'numpy'。
本轮新增规则机同形接口测试已通过 compileall 语法检查，但未能在当前环境执行 pytest。
```

```bash
python -m compileall -q src tests
```

结果：通过，无输出，表示 `src` 和 `tests` 下 Python 文件可成功字节编译。

```bash
git diff --check
```

结果：通过，无输出，表示当前 diff 无空白或格式问题。

新增/更新测试覆盖：

- reset 可复现；
- 正常 policy step 运行 20 个物理子步；
- held action 在 20 子步后消费到 0；
- 子步内命中提前结束；
- `y` 高度与撞地；
- 单弹观测；
- 多弹 padding 和 mask；
- powered_launch；
- terminal_intercept；
- separation/threat 奖励方向；
- 终局奖励数值与规则机验证方案一致；
- timeout 不给正终局奖励；
- 规则机安全/速度优先级；
- 规则机 beam / break 状态切换；
- 规则机 `act(observation, action_mask)` 同形接口与 mask fallback；
- success / exhausted / timeout 区分；
- closest approach 不直接令导弹失效；
- info 字段包含 reward components、missile mask、action mask、substeps、time。

## 5. 仍存在的差异

1. **奖励值按规则机验证方案采用较小终局数值。** crash=-20、hit=-10、success/exhausted=+10 与给定方案一致；为避免塑形累计压过终局，代码增加了 `shaping_scale=0.1`，这是本轮认为必要的缩放调整。

2. **规则机同形接口基于 observation 的信息量受当前观测设计限制。** 规则机 `act()` 已不再需要 `WorldSnapshot`，但 observation 本身仍缺少平台类型或显式机动能力；当前动作 mask 会通过 `info` 输出，但观测向量本身未包含平台类型、最大过载或动作能力摘要。

3. **蓝方高度在观测中重复出现。** 当前自机观测同时包含 `y_norm` 和单独的高度 `y_norm`，符合“归一化位置 + 高度 y”的字面要求，但特征上存在重复加权风险。

4. **多弹排序确定但不具备槽位滞回。** 当前按距离升序排序，确定但在距离接近时仍可能出现槽位交换；后续可引入 missile id、time-to-go 排序或 slot hysteresis。

5. **场景 YAML loader 仍是简易解析器。** 当前仅支持本项目简单 key-value 与数值列表格式，尚未迁移到完整 schema 校验。

6. **导弹动力学仍为简化模型。** 当前包含 PN、动力段 `nx`、失速、超时等语义，但未建模完整气动阻力、推力曲线、 seeker 噪声、脱锁/重锁过程。

7. **success 语义是工程化近似。** 当前基于最近点、安全距离和闭合速度判断安全通过，尚未引入任务级“保持安全若干秒”的确认窗口。

8. **`step_policy_interval(command)` 仍保留兼容接口。** 环境已使用 held-action 子步消费接口，但旧接口仍可直接用固定 command 推进一个 policy interval。

## 6. git diff 摘要

本轮在上一版 M4 基础上继续对齐规则机策略和奖励函数，当前工作区 diff stat：

```text
docs/refactor/status/M4_environment_report.md      | 157 ++++++++--
src/air_combat_rl/controllers/baselines/blue_behavior_tree.py    | 250 +++++++++++++++
src/air_combat_rl/tasks/blue_escape/rewards/components.py        | 328 ++++++++++++++++++---
tests/unit/test_m4_blue_escape_env.py              |   5 +-
tests/unit/test_m4_behavior_tree_reward_alignment.py             |  67 +++++
5 files changed, rule-policy/reward alignment changes included
```
