# 当前项目运行、训练与模型审计

本文基于当前仓库代码进行静态审阅，目的是回答四个实际问题：怎样配置并运行场景、训练链路怎样工作、GPU 在哪里参与、结果怎样展示；同时说明当前逻辑中哪些部分可信，哪些部分仍只是研究原型。这里的“正确”仅指软件接口和内部约定基本自洽，不等于模型已经通过真实飞行数据或武器试验数据验证。

## 1. 结论摘要

当前项目已经形成完整的研究流水线：YAML 场景 → XZY 三自由度仿真 → 29 个离散机动/连续意图投影 → PPO 或 Rainbow 训练 → checkpoint → 固定评估套件 → PNG、Markdown、CSV/JSON 和 ACMI。分层、可复现种子、action mask、训练/评估分离、超时不计成功、checkpoint 类型约束和 GPU/CPU 分工总体合理。

但当前版本更适合**算法接口验证和消融实验**，不适合直接给出具有工程或战术含义的结论。主要原因如下：

1. 场景随机化很弱；除 `randomized_1v1` 的蓝方高度外，初始位置、航向、速度、来袭方位和导弹性能基本固定。
2. `configs/platform/*.yaml` 和 `configs/reward/*.yaml` 已由运行时按平台和威胁数量自动加载；修改参数会进入投影器和奖励模型，并对未知字段严格报错。
3. 飞机模型仍是无推力/阻力/质量的点质量三自由度模型；平台速度范围已用于运行时包线钳制，平台高度与过载参数用于连续动作投影/离散动作 mask，但这不等价于真实气动包线。
4. 导弹按初始爬升角发射，全程使用 PN 接近蓝方，前 7 秒在阻力下加速，之后在重力和二次阻力下滑行；PN 坐标系和过载量纲已修正，但仍需标准算例和外部数据标定。
5. 观测向量前 8 项中高度被重复写入两次；不会造成维度错误，但第二份没有提供新信息。
6. 当前配置的 5 m 杀伤半径、0.005 s 欧拉积分、30 km 成功距离、导弹速度/过载和奖励权重都属于待标定假设，并非由仓库内数据证明的合理参数。

因此，对“当前运行逻辑是否正确”的回答是：**软件主链路基本完整且设计方向合理；物理可信度、场景覆盖度和参数有效性尚不足，不能把测试通过等同于数学模型正确。**

## 2. 场景怎样设置和运行

### 2.1 场景配置入口

场景文件位于 `configs/scenario/`。运行时真正接受的字段由 `ScenarioConfig` 定义，常用字段如下：

| 字段 | 含义 | 当前默认值/说明 |
| --- | --- | --- |
| `mode` | 场景构造模式 | `fixed_1v1`、`randomized_1v1`、`high_threat_1v1`、`terminal_intercept`、`powered_launch`、`1v2`、`1vN` |
| `seed` | 构造随机种子 | CLI 的 `--seed` 会覆盖文件值 |
| `blue_altitude_m` | 蓝方高度范围 | 只有 `randomized_1v1` 真正随机抽样，其他模式取区间中点 |
| `blue_speed_mps` | 蓝方初速 | 默认 300 m/s |
| `missile_count` | 导弹数量 | 1v2 强制为 2；1vN 使用配置值 |
| `terminal_distance_m` / `terminal_speed_mps` | 末段拦截初始距离/速度 | 仅末段模式使用 |
| `powered_speed_mps` | 动力发射初速 | 仅动力发射模式使用 |
| `missile_altitude_m` | 导弹初始高度 | 默认 10 km |
| `physics_dt` / `policy_dt` | 物理步长/决策步长 | 当前配置为 0.005 s / 0.1 s，即每个策略步 20 个物理子步 |
| `max_episode_time_s` | 最大回合时长 | 默认 60 s |
| `world` | `WorldConfig` 覆盖项 | 可设置 PN 系数、杀伤半径、失速速度、动力段参数、成功距离等 |

新增场景时建议显式给出 `name`、`mode`、全部初始条件和 `world` 参数，不要依赖隐式默认值。例如：

```yaml
mode: terminal_intercept
name: terminal_intercept_calibration
seed: 0
blue_altitude_m: [10000.0, 10000.0]
blue_speed_mps: 300.0
missile_count: 1
missile_altitude_m: 10000.0
terminal_distance_m: 30000.0
terminal_speed_mps: 900.0
max_episode_time_s: 60.0
physics_dt: 0.005
policy_dt: 0.1
world:
  navigation_constant: 4.5
  kill_radius_m: 5.0
  missile_stall_speed_mps: 250.0
  success_distance_m: 30000.0
```

注意：当前 factory 不支持在 YAML 中直接配置蓝方/各枚导弹的完整三维坐标、航向角、弹道倾角或各自不同的性能。多弹场景只是在水平面上等角度布置导弹。这类需求应先扩展 schema 和校验器，而不是复制更多只改 `mode` 的文件。

### 2.2 单场景冒烟运行

安装依赖后，先用非训练策略验证场景：

```bash
PYTHONPATH=src python scripts/run_scenario.py \
  --scenario configs/scenario/fixed_1v1.yaml \
  --actions configs/actions/blue_29.yaml \
  --platform zdj --policy random_valid --seed 0 \
  --output-dir runs/smoke_fixed_1v1
```

输出包括 `manifest.json`、逐步的 `steps.jsonl` 和 `episode_summary.json`。建议每次改物理或场景配置后，至少检查：时间是否单调、每个策略步的子步数、最低高度、最小弹目距离、终止原因、动作是否合法，以及相同 seed 是否复现。

坐标约定必须保持一致：物理状态序列为 `[x, z, y, V, gamma, psi]`，`x-z` 是水平面，`y` 是高度；`psi` 从 `+x` 向 `+z` 右转为正，`gamma` 爬升为正。

## 3. 当前训练设计和调用链

### 3.1 环境和动作

`BlueEscapeEnv` 每 0.1 s 接收一个 `action_id`，动作在 20 个 0.005 s 子步内保持。动作表 `blue_29/v2` 定义 `[nx, nf, gamma_s]` 及适用平台。`zdj` 可使用完整动作集，`yjj` 只允许动作表明确列出的约 3g 子集；非法动作由 action mask 排除。

固定观测维度是 `8 + 4 × 11 = 52`：8 个蓝方/当前动作特征，最多 4 个按距离排序的有效导弹槽，每槽 11 个相对位置、相对速度、距离、闭合速度、方位、俯仰和威胁特征；空槽补零，并另给 `missile_mask`。

终止语义为：命中 `hit`、蓝方撞地 `crash`、导弹失效 `exhausted`、安全越过且距离满足门限 `success`；时间或步数上限是截断 `timeout`。timeout 不作为 success，这是合理的评估设计。

### 3.2 三种算法

- **Projected PPO（主路径）**：高斯 actor 输出 3 维连续意图，经 tanh 和物理命令映射后，投影到当前合法的 29 动作之一；环境仍执行离散动作。PPO 的概率比针对投影前的连续样本计算。该设计能训练，但投影是多对一且不可微，必须监控投影距离和动作频率，避免 actor 的连续输出与实际执行脱节。
- **Discrete PPO（基线）**：直接对经过 mask 的 29 动作做 categorical PPO，适合作为判断投影设计是否真正有收益的对照。
- **Rainbow DQN（基线）**：离散 Q 网络和 replay buffer。当前实现名称叫 Rainbow，但是否具备完整 Rainbow 的 distributional、dueling、double、noisy、prioritized、n-step 全套组件，应按代码逐项确认，不应仅凭名称解释实验。

CPU 小规模训练示例：

```bash
PYTHONPATH=src python scripts/train.py \
  --scenario configs/scenario/fixed_1v1.yaml \
  --actions configs/actions/blue_29.yaml \
  --algorithm configs/algorithm/ppo_projected.yaml \
  --platform zdj --seed 0 --total-steps 100000 \
  --checkpoint-interval 10000 \
  --output-dir runs/ppo_projected_cpu
```

训练产生 `train_metrics.jsonl`、`episodes.jsonl`、`manifest.json` 和 `checkpoints/`。恢复训练时 `--total-steps` 表示最终累计步数，而不是追加步数。

### 3.3 课程学习

`configs/curriculum/blue_escape.yaml` 从 fixed 1v1、随机 1v1、高威胁、多威胁逐步推进，按滚动 success/survival 门限晋级，并保留部分简单样本以减轻遗忘。思路合理，但当前晋级阈值是人为设定，且基础场景本身缺少足够随机化，因此“达到 80%”可能表示记住有限初态，而不是获得泛化能力。

建议训练集和最终评估集分离：评估必须增加未见过的连续初始条件、不同来袭方位/高度差/速度、参数扰动和至少 20–30 个独立 seed；报告均值的同时报告置信区间。不要用课程晋级窗口作为最终性能结论。

## 4. GPU 如何加速

只有 `ppo_projected_torch.yaml` 的 PyTorch backend 使用 GPU。其架构是：

1. 多个 CPU 环境进程并行运行物理、制导、碰撞、奖励和动作投影；
2. 主进程把所有环境的 52 维观测组成 batch，送入同一个 GPU actor-critic；
3. GPU 批量完成策略推理、价值估计和 PPO minibatch 反向传播；
4. 模型、optimizer 和 AMP scaler 保存在 checkpoint，环境 worker 不复制 CUDA 模型。

因此 GPU **不加速当前 Python 物理仿真本身**。当环境步进占主导时，应先增加 CPU worker、做 profiling 或向量化仿真；小网络、小 batch 或单环境时，GPU 甚至可能更慢。

先检查 CUDA：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

推荐命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src python scripts/train.py \
  --curriculum configs/curriculum/blue_escape.yaml \
  --actions configs/actions/blue_29.yaml \
  --algorithm configs/algorithm/ppo_projected_torch.yaml \
  --platform zdj --seed 0 --device cuda:0 \
  --num-envs 8 --env-backend subprocess \
  --worker-start-method spawn --total-steps 5000000 \
  --checkpoint-interval 100000 \
  --output-dir runs/projected_ppo_curriculum_gpu
```

`global_step` 是所有环境 transition 的总数，必须能被 `num_envs` 整除。配置默认开启 AMP 和 TF32；前者确实接入训练，当前代码虽然记录 `allow_tf32`，但没有显式把它写入 PyTorch backend 开关，因此不要假定该字段已生效。复现实验应使用 `device.deterministic: true`、关闭 AMP，并记录 GPU、CUDA、PyTorch 版本；性能实验则可开启 AMP。

## 5. 评估和结果展示

训练曲线应从 `train_metrics.jsonl` 观察 reward、value/policy loss、entropy、KL、clip fraction、梯度范数、吞吐，以及 Projected PPO 特有的投影距离和实际动作频率。仅看 reward 上升不足以证明策略变好。

固定套件评估：

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --evaluation-suite configs/evaluation/full_suite.yaml \
  --actions configs/actions/blue_29.yaml \
  --algorithm configs/algorithm/ppo_projected_torch.yaml \
  --checkpoint runs/projected_ppo_curriculum_gpu/checkpoints/latest.pt \
  --platform zdj --device cuda:0 --num-envs 8 \
  --env-backend subprocess --deterministic \
  --output-dir runs/eval_projected_ppo
```

评估目录提供 `metrics.json`、`episodes.csv`、`report.md`、逐步 JSONL 和轨迹。离线画图：

```bash
PYTHONPATH=src python scripts/plot_evaluation.py \
  --evaluation-dir runs/eval_projected_ppo
```

单场景轨迹使用 `plot_run.py`；三算法在条件完全一致时使用 `compare_algorithms.py`。需要 Tacview 时用 `export_acmi.py`，并显式提供经纬度原点和参考时间。展示时建议至少同时报告：success、survival、hit、crash、exhausted、timeout 比率，回合 reward，最小弹目距离，最低高度，时长，分场景结果和 seed 方差。

## 6. 数学模型与参数审计

### 6.1 可以保留的设计

- XZY 坐标约定在动力学、环境、绘图和 ACMI 中已有明确约束。
- 物理步与策略步分离是必要的；0.1 s 控制周期配合更小物理步在结构上合理。
- 飞机点质量模型中速度、航迹倾角和航向角导数的形式与所声明的过载命令接口基本自洽，适合作为低保真原型。
- 终止/截断区分、action mask、固定尺寸多弹观测、统一算法评估接口和离线展示层是良好的工程设计。

### 6.2 必须优先校正或验证的问题

| 优先级 | 问题 | 影响 | 建议验证 |
| --- | --- | --- | --- |
| P0 | PN 与导弹阻力仍未外部标定 | 公式已统一坐标系和量纲，但参数仍会直接决定威胁难度 | 用恒速交叉、正碰、尾追解析算例验证，并用参考弹道标定导航系数、推力和阻力系数 |
| P0 | 无飞行包线和平台配置接线 | 9g 纵向加速度可让速度无限增长/变负，平台 YAML 上下限不起作用 | 把平台 config 注入 world/projector；实现速度、过载、动压、高度边界或至少明确 clip；增加能量/包线测试 |
| P0 | 场景覆盖不足 | 训练和评估可能共享少数确定初态，指标虚高 | 参数化完整初态，拆分 train/validation/test seed 与范围，做域随机化和外推测试 |
| P1 | 配置追溯仍可增强 | 平台/奖励 YAML 已接入，但训练 manifest 还应保存文件内容摘要 | 在 checkpoint 中绑定平台、奖励配置签名，恢复时做兼容检查 |
| P1 | 观测高度重复 | 浪费一个特征并隐藏 schema 意图 | 明确第 4 项应为高度裕度、归一化地高或删掉；变更时提升 observation schema/checkpoint 版本 |
| P1 | 导弹动力过简 | 无阻力、质量、推力曲线、最大过载/seeker 脱锁，`locked` 基本恒真 | 接入现有 propulsion/guidance 模块或删除误导配置；用可解释弹道数据标定 |
| P1 | 成功/失效定义 | 30 km 且过最近点即成功、失速即正奖励，策略可能利用模型漏洞 | 分别报告 escape 与 exhausted；对门限做敏感性分析；确认失锁/失速不是奖励捷径 |
| P2 | 欧拉积分和 5 m 杀伤半径 | 高速弹每 0.005 s 可移动数米，离散采样仍可能漏碰或产生步长敏感结果 | 启用连续 closest-approach 碰撞判断；做 dt=0.01/0.005/0.0025 收敛测试 |
| P2 | PPO 超参数未系统标定 | CPU 配置学习率 0.003 较激进且 rollout 32 很短；GPU 配置更接近常见起点但非证明 | 3–5 个 seed 做 LR、entropy、rollout、batch 消融；记录 explained variance 和梯度统计 |

### 6.3 参数合理性的判定方法

不要仅凭数值“看起来合理”。建议设置四类验收门槛：

1. **单元量纲**：每个导数和制导量标明 SI 单位，并通过量纲测试。
2. **解析基准**：无机动匀速、定常平飞、恒定过载转弯、正碰 PN 等场景与解析解比较。
3. **数值收敛**：物理步长减半后，命中结果、最近距离和终止时间误差低于预设阈值。
4. **外部标定**：用允许使用的公开或内部参考曲线校准速度、高度、转弯率、射程和制导误差，并保存数据来源与版本。

完成以上工作之前，最稳妥的表述是“该项目验证了 RL 管线在自定义低保真环境中的行为”，而不是“验证了真实空战逃逸算法的有效性”。

## 7. 建议执行顺序

1. 先修复/验证 PN，并增加坐标系、量纲和解析算例测试。
2. 把 platform/reward YAML 的内容签名加入 checkpoint 兼容检查。
3. 为完整初态和导弹参数建立可随机化 schema，固定独立评估分布。
4. 加入连续最近点碰撞和步长收敛测试，再确定 `physics_dt` 与 kill radius。
5. 最后进行多 seed 超参数搜索和三算法公平对比；否则调参会拟合尚未可信的物理模型。

这套顺序优先解决“环境是否值得学习”，再解决“学习是否足够快”，能避免用 GPU 高速放大建模误差。
