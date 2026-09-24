# ATEC Task E 独立交接工程

本目录用于把桌面操作任务交给另一位成员独立开发。任务 ID 为 `ATEC-TaskE-Piper`，使用固定 Piper 机械臂，将桌上的糖盒、芥末瓶和香蕉放入篮子。

这是当前本地工程的 Task E 副本，不是重新下载的官方原始版本。Task E 环境、公共环境、奖励、终止条件、示范控制器和训练算法原样保留；独立化修改限于任务注册、推理入口、评测说明和调试输出、模型与抓取资产路径、交接打包。原工程的环境和算法代码未修改。

**交接目标：在保持评测场景、奖励和成功条件不变的前提下，开发一个仅依据标准观测、能够稳定把三件物体全部放入篮子的策略，并提供可重复运行的评测结果。当前交接包是开发起点，不是已经验证通关的成品。**

## 1. 任务是什么

### 1.1 机器人和操作流程

本任务使用固定安装的 Piper 机械臂，工作范围是桌面三物体抓取和入篮。

```text
观察物体 → 确定抓取位置和姿态 → 接近 → 闭合夹爪 → 抬起
         → 搬运到篮子上方 → 放下并张开夹爪 → 撤回 → 下一个物体
```

| 物体编号 | 物体 | 需要关注的操作问题 |
| --- | --- | --- |
| 1 | 糖盒 | 夹取位置、盒体姿态、抬起后的夹持稳定性 |
| 2 | 芥末瓶 | 瓶身夹取高度、滑脱、放置姿态 |
| 3 | 香蕉 | 长轴方向、夹爪朝向、夹取位置 |

三件物体要在同一个完整 episode 中完成搬运。单物体成功、夹爪空走完状态机或出现部分奖励，都不能记为完整任务成功。

### 1.2 可使用的观测和动作

- `obs["proprio"]`：按配置顺序排列的关节相对位置、相对速度和上一帧动作。当前 Piper 共 8 个关节。
- `obs["image"]`：外部相机 `video_rgb/video_depth` 和末端相机 `ee_rgb/ee_depth`，配置分辨率为 640 × 480。
- 当前 ACT 策略实际使用的是恢复后的 8 维绝对关节位置和外部 RGB，图像缩放到 224 × 224；尚未接入深度、末端图像或关节速度作为网络输入。
- 动作包含 6 个机械臂关节和 2 个夹爪关节，顺序为 `joint1` 到 `joint8`。必须核对数据、策略输出和环境的顺序一致。

默认动作配置是位置控制，`scale=0.5`、`use_default_offset=True`。不要把绝对关节角直接当成环境动作：

```text
joint_target = default_joint_pos + 0.5 × action
action = (joint_target - default_joint_pos) / 0.5
```

当前 `get_action_spec()` 返回空字典，仍使用默认动作配置。若自行调整动作定义，必须同步采集、训练和推理，不能只修改其中一处。

### 1.3 如何判断完成

当前本地环境的 `ObjectsInBasketDone` 要求三个物体的中心**同时**满足：

- X 距篮子判定中心不超过 0.20 m。
- Y 距篮子判定中心不超过 0.11 m。
- Z 在桌面高度到桌面高度 + 0.15 m 之间。

判定使用各环境的局部坐标。实现见 `source/atec_rl_lab/atec_rl_lab/tasks/task_e/mdp/terminations.py`。抓取奖励只是基于末端距离和物体抬升高度的代理指标，不能代替完整成功判定；入篮条件本身也没有额外的持续稳定时间检查。

当前公共环境默认超时为 1200 s，控制步长由 `sim.dt=0.005` 和 `decimation=4` 得到 0.02 s。采集器另设较短时限，不要混为一谈。这些是当前本地配置，不是对最新比赛规则的重新确认。

## 2. 已经有什么，验证到了哪一步

| 模块 | 已提供内容 | 验证状态与限制 |
| --- | --- | --- |
| 环境 | Piper、桌子、篮子、三件物体、RGB-D 观测、奖励和终止条件 | 已保留当前本地代码，尚未完成独立包完整仿真测试 |
| 资产 | Piper USD、桌面和三件抓取物体 USD、贴图、HDR 光照 | 已补齐光照资源，抓取资产统一收纳到 Task E 目录；尚未验证完整仿真图像与任务表现 |
| 专家控制 | 笛卡尔控制器、抓取姿态求解、抓放状态机 | 已有实现；物体 1、2 和完整三物体流程仍需实测 |
| 数据采集 | HDF5 关节状态、动作、末端状态和外部 RGB 保存 | 已有脚本；交接目录的数据集为空，成功终止处理需修复 |
| ACT 训练 | RGB 编码、动作序列预测、归一化、EMA、checkpoint 保存 | 已有训练实现；未在独立目录重新训练 |
| 基线权重 | `atec_robot_model/baseline/act/policy.pt` | 严格加载及 CPU 合成观测推理通过，输出形状为 `(1, 30, 8)` |
| 推理接口 | `demo/solution.py` 导出 `AlgSolution`，可切换权重 | 已检查导入与权重兼容性，当前运行入口使用 CUDA |
| 交接检查 | 源码语法、wheel 子包、文件哈希、来源记录 | 已通过静态检查，不代表抓取任务成功 |

已有 ACT 会预测未来 30 步动作，并在后续观测下重新预测，通过时间聚合生成当前动作；不是只看一张图后盲执行整段轨迹。

专家采集器直接读取仿真物体位置和姿态，用于离线生成示范。**这些真值没有作为标准策略观测提供，不能直接把这个 oracle 状态机当成已经实现的视觉策略。**

## 3. 还需要做什么

### 3.1 首先处理已发现的运行和数据缺口

以下是源码核对发现的算法和数据问题，**此次目录整理没有修复这些算法问题**：

1. **区分成功终止、失败终止和超时。** `scripts/act/task_e/collector.py` 当前遇到任何 `terminated/truncated` 都返回 `None`，若三物体成功触发终止，也会被丢弃。应在采集端识别终止原因并保存成功轨迹，不应删除评测成功条件。环境可能自动 reset，末次状态应在 reset 前取证，不能只检查返回后的场景。
2. **统一采集分布与评测分布。** 例如糖盒的 Y 采集范围是 `0.20–0.25 m`，当前评测配置是 `0.25–0.29 m`。应让示范覆盖评测分布；不要缩窄评测范围迁就策略。
3. **真正验证每件物体的抓取。** 原采集脚本提示物体 1、2 可能需要调整抓取距离。应先观察接近、闭合、抬起、搬运、释放各阶段，再调整每物体抓取参数。
4. **只把经过检查的数据用于训练。** `--only_success` 默认关闭；开启该选项也不能替代成功终止修复。采集自带的入篮检查和环境的 Z 判定并不完全一致，应在数据验收时对齐。`filter_demos.py` 的近零动作过滤不等于成功轨迹筛选。

此前遗漏的 HDR 光照文件已补齐，并加入交接文件检查；仍需在仿真中检查照明和相机图像是否正常。

### 3.2 分阶段开发与验收

| 顺序 | 工作 | 进入下一阶段的条件 |
| --- | --- | --- |
| 1 | 配好独立环境、查看场景 | 无缺失资产，相机图像和机器人初始化正常 |
| 2 | 原样运行已有 ACT，记录视频、完成物体数、终止原因 | 建立可复现基线，明确失败阶段，不能只看分数 |
| 3 | 分别验证三个物体的专家抓取，再验证连续三物体流程 | 各物体可反复成功，完整成功轨迹能正确保存 |
| 4 | 采集覆盖评测分布的完整示范，检查 RGB 与动作对齐 | 数据不含跨 reset 拼接，帧数一致，动作定义一致 |
| 5 | 训练或微调策略，比较多个 checkpoint | 权重能通过统一推理接口运行 |
| 6 | 用未参与采集的位置/种子做闭环评测 | 给出完整成功率、完成耗时和失败分类，改善才保留 |

建议先用每件物体少量试验定位问题，再扩大完整任务数据量。数据量、batch size 和训练迭代数依据显存、示范质量及评测结果决定，后面的命令只是示例。

## 4. 可以采用什么方案

### 方案 A：沿用现有 ACT 模仿学习（优先建立基线）

```text
修复并验证专家 → 采集 RGB + 关节状态 + 环境动作
              → 训练 ACT → 标准观测推理 → 完整任务评测
```

- 优点：已有采集、训练、网络、基线权重和推理入口，改动最小。
- 重点：提高示范质量、覆盖物体位置变化，先解决完整三物体序列，再按失败证据考虑末端相机、深度或其他输入。
- 限制：专家演示失败或训练分布遗漏会传递给策略；训练损失低不保证实际成功。
- 主要开发位置：`scripts/act/`、`source/atec_rl_lab/atec_rl_lab/train/act/`、`demo/solution_act.py`。
- 当前 `best_loss.pt` 按训练损失选取，并不是按独立验证集损失或任务成功率选取。最终模型应比较闭环成功率。

### 方案 B：RGB-D 感知 + 抓放状态机 + 逆运动学

```text
RGB 识别物体/篮子 → 深度恢复三维位置 → 相机坐标转换
                → 抓取姿态选择 → IK → 夹爪控制 → 检查并重试
```

- 优点：接近、抓取和放置阶段可解释，能针对不同物体设计抓取点和失败恢复。
- 可复用：抓放状态机、关节动作转换，以及现有 `CartesianController` 的 IK 思路。该控制器目前读取仿真机器人末端状态和 PhysX Jacobian，不能直接放进只有 `obs` 的提交接口；策略端需依据机器人模型和关节观测计算正运动学/Jacobian，再实现 IK。
- 需要新增：物体与篮子视觉定位、有效深度筛选、相机内外参处理、视觉抓取反馈和重试机制。
- 限制：现有专家从真值取得物体位姿，不能原样作为这个方案的感知模块。标准 `obs` 不直接提供标定参数，需确认评测可用配置或增加合法的标定读取方式。
- 开发边界：最终 `predicts()` 仍只能依据约定输入生成动作，不能依赖仿真内部物体真值。

### 方案 C：学习策略 + 分阶段反馈的混合方案

- 让 ACT 负责主要接近/搬运动作，使用关节反馈和图像判断是否进入预期阶段。
- 基于可观测证据加入抓取失败、滑脱、放置失败的有限重试；不能仅因预测动作结束就宣布成功。
- 可先增加记录和阶段识别，再逐项启用恢复控制，并与纯 ACT 对照。
- 当前未提供完整阶段识别/重试实现；这是候选开发路线，不是已有功能或已验证增益。

### 方案 D：开源 VLA 微调（先做 SmolVLA，保留 ACT 对照）

VLA（Vision-Language-Action，视觉—语言—动作模型）根据图像、机器人状态和语言指令生成动作。当前 ACT 基线没有语言输入，不属于语言条件 VLA。本任务是固定 Piper、固定三类物体和固定目标篮子的操作任务，VLA 不一定优于现有 ACT；它更适合探索语言条件控制、物体变化和场景泛化。

**建议先微调 SmolVLA，效果不足时再评估 X-VLA；有较强 GPU 服务器时再考虑 π0.5 或 OpenVLA-OFT。该顺序依据接入成本和任务匹配度，不代表已经在 Task E 上验证成功。**

#### 开源模型推荐

| 方案 | 推荐程度 | 对 Task E 的适配理由 | 主要限制 |
| --- | --- | --- | --- |
| [SmolVLA](https://huggingface.co/docs/lerobot/smolvla) | 首选 | 约 450M 参数，支持图像、机器人状态、语言指令和动作分块；LeRobot 提供数据与微调流程 | 需要 Piper 成功示范和微调，不能直接零样本完成任务 |
| [X-VLA](https://huggingface.co/docs/lerobot/xvla) | 第二候选 | 约 0.9B 参数，重视跨机器人适配，支持多相机及自定义动作维度 | 需要适配动作格式、归一化、损失和输出后处理 |
| [π0.5 / openpi](https://github.com/Physical-Intelligence/openpi) | 有较强服务器时考虑 | 提供预训练权重、连续动作生成和自有数据微调流程 | 官方列出的 GPU 显存要求为推理超过 8GB、LoRA 微调超过 22.5GB、完整微调超过 70GB；具体需求取决于训练配置和后端 |
| [OpenVLA-OFT](https://github.com/moojink/openvla-oft) | 研究型备选 | 基于 7B OpenVLA，采用动作分块和连续动作预测，适合研究高效视觉语言控制 | 资源和接入成本较高，不建议作为第一轮方案 |

上述官方示例和其他机器人上的表现不是 Piper 或本赛题的成功证明。openpi 的不同训练后端功能不完全一致，不能假设 PyTorch 分支支持相同的 LoRA 流程；显存预算还需单独考虑 Isaac Sim，不能把模型显存要求当作仿真与训练同时运行的总需求。

#### 为什么优先 SmolVLA

- 当前采集数据已经包含外部 RGB、8 维绝对关节位置和 8 维环境动作，适合转换成 LeRobot 数据集；这只是数据接口适配基础，不代表已有可直接控制 Piper 的 VLA 权重。
- [SmolVLA 官方配置](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/configuration_smolvla.py) 支持把较短状态和动作向量填充到配置上限，因此具备接入本任务 8 维接口的条件。仍需计算自己的归一化统计并使用本任务数据微调。
- 第一轮只使用外部 RGB，与现有 ACT 保持可比；后续加入腕部 RGB 时，需要补采对应图像并同步训练与推理。不要仅修改输入键，也不要直接将深度图当作预训练模型的 RGB 输入。
- [官方介绍](https://huggingface.co/blog/smolvla) 提到 CPU 推理和单张消费级 GPU 训练的可行性，但不能据此保证本任务的实时控制频率或任意显存配置都能训练。若使用 CPU 推理，应实测图像预处理、动作生成和传输的总延迟；微调建议使用 GPU。

#### 接入现有 Task E 代码的步骤

1. **先完善成功示范。** 按第 3 节修复成功终止保存和采集分布问题，每条完整轨迹覆盖三个物体全部入篮。训练与评测使用分离的位置或种子，不通过修改评测条件提高成绩。
2. **转换为 LeRobot 数据集。** 保留 RGB、绝对关节位置和当前环境的 8 维动作，核对时间对齐和 episode 边界。当前采集基线没有保存腕部图像，新增相机输入需要补采。
3. **添加语言任务标签。** 当前标准观测不包含语言指令，可在数据和推理适配器中配置相同的固定指令：`Put the sugar box, mustard bottle, and banana into the basket.`。第一轮使用完整任务指令，不在推理时依赖物体真值选择阶段。
4. **微调并保持动作语义。** 首先保留 6 个臂关节和 2 个夹爪关节的 8 维环境动作，按自身数据计算归一化统计，并按第 1.2 节转换到目标关节位置。其他机器人的 7 维末端位姿增量不能直接当作 Piper 的 8 维关节动作；X-VLA 若采用 `auto` 动作模式，也需核对真实动作维度、填充、损失和输出裁剪。
5. **只新增 VLA 推理适配器。** 可新增 `demo/solution_vla.py`，再让统一入口 `demo/solution.py` 选择 ACT 或 VLA，保留 `AlgSolution.predicts(obs, current_score)` 和返回格式。动作分块应结合新观测定期重算，在 episode 重置时清空动作缓存，并实测推理延迟与 0.02 s 控制步长的衔接；模型不必每个控制步都重新推理。
6. **同条件闭环比较。** 固定评测场景、位置集合、种子和完整成功判定，比较 ACT 与 VLA 的三物体完整成功率、掉落/空抓次数、完成耗时和推理延迟。仅在回归集改善时保留，不能用训练损失或单次抓取代替完整任务验收。

推荐实验路线：**ACT 可复现基线 → SmolVLA 数据转换与微调 → 闭环对照 → 必要时 X-VLA。** 当前交接包未安装或接入上述 VLA，未提供对应微调权重，也未验证它们能提升本任务成功率；此节仅提供候选开发方案，不改变当前运行命令和依赖。

从头做强化学习也属于可探索方向，但本包没有 Task E 专用的完整 RL 训练入口和已验证训练流程，且原工程说明评测环境不直接支持并行训练。第一轮不建议同时重写环境、奖励和策略；如探索 RL，应在独立训练环境开发，回到不变的评测环境验收。

**推荐顺序：先方案 A 建立可复现基线；若失败主要由视觉定位或抓取反馈缺失造成，再有针对性地选择方案 B 或 C；若希望探索 VLA，则按方案 D 先做 SmolVLA 微调并与 ACT 对照。不要一次更换全部模块。上述方案均需实测，不能预先保证成功率。**

### 最终交付与实验记录

- 可直接运行的 `demo/solution.py`，保留 `AlgSolution`、`predicts(obs, current_score)` 接口，返回 `{"action": ..., "giveup": False}`；权重路径不应依赖开发者个人目录。
- 最终策略权重、归一化统计、训练配置、依赖与启动命令。
- 完整三物体搬运视频，以及覆盖不同位置/种子的逐次评测记录。
- 每次记录代码版本、权重哈希、场景配置/种子、完成物体数、成功终止与否、耗时、失败阶段。
- 建议建立固定回归集，先小规模调试，再进行例如 100 次独立回合的成功率统计；此数量是工程建议，不是比赛规定。当前尚需实现批量评测和结果汇总工具。
- 不修改评测奖励、成功条件或物体分布来提高成绩。训练环境可单独设计，但最终验收必须回到约定评测环境。

## 5. 目录结构

```text
ATEC_TaskE_Standalone/
├── README.md
├── HANDOFF_CHANGELOG.md
├── requirements.txt
├── LICENSE
├── SHA256SUMS
├── ORIGIN_MANIFEST.json
├── source/atec_rl_lab/
│   └── atec_rl_lab/
│       ├── tasks/task_e/          # 场景、观测、奖励、成功条件
│       ├── tasks/task_base/       # 公共环境与动作接口
│       ├── assets/                # Piper 和物体的 Python 配置
│       ├── utils/                 # 示范采集的笛卡尔控制器
│       └── train/act/             # ACT 训练网络
├── scripts/
│   ├── view_task_e.py             # 静态场景查看
│   ├── play_atec_task.py           # Task E 评测入口
│   ├── check_handoff.py           # 文件、语法与哈希检查
│   └── act/                      # 示范采集、过滤和训练
│       ├── task_e/               # 专家示范状态机
│       └── runs/                 # 新训练结果，交接时为空
├── demo/
│   ├── solution.py               # 统一评测/提交接口
│   ├── solution_act.py           # ACT 推理实现
│   └── act/                      # 推理网络
├── atec_robot_model/
│   ├── robot/piper/              # 机械臂与其 USD 子层
│   ├── objects/task_e/           # 桌子、篮子、贴图与材质
│   │   └── pick_objects/         # 糖盒、芥末瓶、香蕉及其贴图
│   ├── scene/                   # 公共地面资产与 HDR 光照
│   └── baseline/act/policy.pt    # 随原工程提供的 ACT 基线
├── datasets/atec_task_e/          # 新采集数据，交接时为空
└── docs/                         # Piper 图片、Task E 示例动画和验证记录
```

`tasks/task_base/` 是 Task E 继承的公共环境依赖，不是一个独立比赛任务。`objects/task_e/pick_objects/` 中只保留本任务使用的三件物体；USD 的配置、纹理目录必须保持相对布局。

## 6. 安装与隔离

Isaac Lab、Isaac Sim、NVIDIA 驱动和 Python 环境不包含在交接目录中。本任务的来源工程标注的验证框架版本是 Isaac Lab 2.3.2；请按该版本配套安装 Isaac Sim。

请使用已配置 Isaac Lab 的独立环境。本工程使用 `atec_rl_lab` 导入名称，同一环境中重复 editable 安装同名包会互相覆盖。

下面命令中的环境名应替换为接收者的专用 Isaac Lab 环境名。项目根路径可以任意更换。

```bash
conda activate atec_task_e
cd /path/to/ATEC_TaskE_Standalone
python -m pip install -r requirements.txt
python -m pip install --no-deps -e ./source/atec_rl_lab
export PYTHONPATH="$PWD/source/atec_rl_lab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/check_handoff.py
```

PyTorch 和 torchvision 应沿用 Isaac Lab 所需的匹配版本，未在 requirements 中强制替换。`OmniPBR.mdl` 是仿真运行时提供的材质依赖，不在本目录中；普通 USD 库在 Isaac Sim 外部扫描时会提示该材质未解析。

本工程需要保留 `source/` 和 `atec_robot_model/` 的相对布局，并使用 editable 安装。Python wheel 构建检查不代表可以仅安装 wheel 而丢弃资产目录。

运行仿真前先确认 Task E 光照资源存在：

```bash
test -f atec_robot_model/scene/kloofendal_43d_clear_puresky_4k.hdr
```

`scripts/check_handoff.py` 检查列出的文件（包含光照资产）、语法与哈希，不能代替完整仿真测试。

## 7. 按步骤运行

以下命令均在项目根目录执行，并使用上一节激活的专用环境与 PYTHONPATH。

### 第一步：只查看场景

```bash
python scripts/view_task_e.py --num_envs 1 --enable_cameras
```

### 第二步：运行已有 ACT 基线

```bash
unset ATEC_TASK_E_POLICY
python scripts/play_atec_task.py --task ATEC-TaskE-Piper --num_envs 1 --enable_cameras
```

默认权重是本目录的 `atec_robot_model/baseline/act/policy.pt`。当前推理实现沿用 CUDA 设备配置，CPU 合成观测测试不代表整个评测入口支持 CPU。

### 第三步：采集自己的示范

先用少量单物体示范检查夹取、放置和图像保存，再扩大采集规模。以下是物体 3 的调试示例，不是完整三物体训练数据采集：

```bash
python scripts/act/collect_demos_task_e.py \
  --pick_objects 3 --num_demos 5 \
  --output_dir ./datasets/atec_task_e \
  --headless --enable_cameras --save_images --only_success
```

原采集脚本注释提示物体 1、2 可能仍需调整抓取距离；不能把单物体示范成功视为三物体任务完成。需要全任务数据时，应先完成第 3 节的成功终止修复和分布对齐，再验证 `--pick_objects 1 2 3`。**当前不应直接启动大量三物体采集：成功轨迹可能被当作早停丢弃，采集循环没有尝试次数上限。**

**采集脚本会覆盖输出目录内已有的 trajectory.hdf5，重复采集应使用新的 output_dir 或先备份。** 下面的过滤仅移除近零动作，需先确认示范本身成功，并检查过滤是否误删必要的停留片段：

```bash
python scripts/act/filter_demos.py \
  --input datasets/atec_task_e/trajectory.hdf5 \
  --output datasets/atec_task_e/trajectory_filtered.hdf5 \
  --threshold 0.001
```

### 第四步：训练 ACT

完成并验证所需示范后运行；完整三物体策略应使用覆盖完整流程的数据，不能只用上面的单香蕉示范就声称完成任务。示例的 num_demos 必须与实际数据量相符，batch size 应依据设备资源调整：

```bash
(
  cd scripts/act
  python train_task_e.py \
    --demo_path ../../datasets/atec_task_e/trajectory_filtered.hdf5 \
    --num_demos 100 --include_rgb \
    --total_iters 100000 --batch_size 256 \
    --exp_name act-task-e-rgb
)
```

结果保存在 `scripts/act/runs/<运行名称>/checkpoints/`。示例不启用 W&B；原 `baseline.sh` 保留，使用它之前需检查其 `--track` 和账户配置。

### 第五步：评测新权重

将 `<运行名称>` 替换为实际训练输出目录名：

```bash
export ATEC_TASK_E_POLICY="$PWD/scripts/act/runs/<运行名称>/checkpoints/best_loss.pt"
python scripts/play_atec_task.py --task ATEC-TaskE-Piper --num_envs 1 --enable_cameras
```

任务配置提供末端和外部 RGB-D 观测，但当前 ACT 采集/训练基线只使用保存的外部 RGB 图像。另一位开发者可在 `demo/solution_act.py`、`scripts/act/` 和 `source/atec_rl_lab/atec_rl_lab/train/act/` 中开发新策略；不要为了提高得分而修改评测奖励或成功条件。

## 8. 验证范围与交接注意

已检查基线权重严格加载、CPU 合成观测推理、Python wheel 子包完整性，以及主要任务资产的本地 USD/贴图依赖。Task E 与公共环境算法文件和原工程逐文件比对；目录整理不改变三物体的物理参数、训练算法或成功条件。HDR 已补齐，资产迁移后的依赖和文件校验记录见 `docs/VALIDATION.json`。

**未执行 Isaac Sim 全任务成功率测试，未证明基线能完成三物体任务。** 接收者应首先查看场景和运行基线，再分别验证三件物体的成功条件。详情见 `docs/VALIDATION.json`。

`ORIGIN_MANIFEST.json` 记录本任务当前交接文件及其来源，`docs/VALIDATION.json` 记录实际检查范围；不应把它们当作抓取成功率已经验证的证明。`SHA256SUMS` 随目录整理同步刷新，之后自行修改文件导致哈希检查失败是预期结果，不代表修改必然有错。建议接收者自行初始化版本管理，保留交接快照后再开发。
