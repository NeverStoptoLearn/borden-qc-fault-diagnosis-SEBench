# 基于 Borden 三维地下水场景的监测数据质量控制与传感器故障诊断

## Role

你是一位环境监测数据质量控制工程师，正在处理一个 Borden-style 三维地下水污染监测网络的数据。你的任务不是污染源反演、监测井选址或应急抽水调度，而是识别地下水监测数据中的质量问题，并判断这些异常是传感器/记录故障，还是真实污染羽迁移过程。

本任务不需要 GPU，也不要求深度学习。推荐使用稳健统计、时序趋势分析、空间邻近井一致性检查、污染羽到达时间判断、垂向层位一致性和规则/启发式推理方法完成。

## 任务目标

给定公开 Borden 三维地下水场景参数、监测井坐标、少量标注样本、公开验证数据和待评测监测数据，你需要完成：

1. 对 `eval_monitoring_noisy.csv` 中每条记录判断异常类型；
2. 将连续异常记录合并为事件，输出事件级诊断；
3. 给出每类异常的检测规则和物理依据；
4. 输出清洗后的浓度序列；
5. 撰写质控报告，解释主要异常、证据和不确定性。

高分解法需要区分两类容易混淆的情况：

- 真实污染羽到达导致的浓度上升；
- 传感器漂移、时间戳错位、坐标/深度录入错误、单位错误等数据质量问题。

## 背景说明

本任务基于 Borden 三维地下水污染迁移场景参数化生成。公开和隐藏数据都遵循同一类水文地质背景，包括：

- 主流向为 `+x` 方向；
- 监测井具有三维坐标 `x, y, z`；
- 浓度单位为 `mg/L`；
- 时间单位为 `days`；
- 污染羽随地下水流动发生对流、弥散、尾迹衰减和空间扩散。

普通异常检测只能识别一部分明显错误，例如负值、缺测、单点突刺。真正困难的是利用 Borden 场景物理一致性判断：

- 下游井是否应晚于上游井出现峰值；
- 相邻井是否应具有相似趋势；
- 深层井和浅层井响应是否符合垂向分布；
- 某个快速上升是否是真实污染羽到达，而不是传感器故障；
- 某口井整段曲线是否更像其他坐标/深度位置的响应。

## 输入文件

### 场景与元数据

- `public_problem_config.json`  
  Borden-style 场景配置，包括流向、水文参数、弥散参数、单位说明和允许标签。

- `borden_grid.npz`  
  三维网格数组，包含 `x/y/z` 网格坐标，用于理解场地尺度和井位分布。

- `public_wells.csv`  
  监测井信息，包含 `well_id,x,y,z,screen_depth_m`。

### 公开训练与验证数据

- `train_monitoring_noisy.csv`  
  公开训练监测记录，包含带噪声和部分故障的浓度观测。

- `train_fault_labels.csv`  
  少量训练标签。注意：该文件不是完整标注，只能作为异常类型示例。

- `public_validation_noisy.csv`  
  公开验证记录，用于本地调试。

- `public_fault_labels_small.csv`  
  公开验证集的少量标签，用于本地验证，不覆盖全部记录。

- `public_clean_reference_subset.csv`  
  少量 clean concentration 参考值，用于校准清洗策略。

### 待提交数据

- `eval_monitoring_noisy.csv`  
  需要你最终诊断的监测记录。你必须为其中所有 `record_id` 生成预测标签和清洗浓度。

### 工具文件

- `baseline_detector.py`  
  一个很弱但合法的 baseline，主要用于生成输出格式。

- `tools/evaluate_public.py`  
  公开本地验证脚本，只使用公开小样本标签，不代表最终隐藏分数。

- `answer_template.json`  
  输出字段说明。

## 异常标签

`anomaly_label` 必须是以下值之一：

```text
normal
spike
drift
stuck_zero
missing
unit_error
time_shift
coordinate_or_depth_error
true_plume_arrival
negative
```

各类含义：

- `normal`：正常观测记录。
- `spike`：单点或短时突刺，与前后时刻和邻近井不一致。
- `drift`：传感器缓慢漂移，表现为整段曲线相对邻近井持续偏高或偏低。
- `stuck_zero`：传感器或记录系统长期固定为 0。
- `missing`：缺测或无效记录。
- `unit_error`：单位或量纲错误，例如 mg/L 与 ug/L 混淆造成数量级异常。
- `time_shift`：时间戳整体提前或滞后，曲线形态合理但峰值时间不符合传播顺序。
- `coordinate_or_depth_error`：井坐标或深度记录错误，曲线更像其他空间位置的响应。
- `true_plume_arrival`：真实污染羽到达，不是传感器故障。
- `negative`：负浓度或明显无物理意义的负值。

重要：`true_plume_arrival` 是真实污染过程。将真实污染羽到达误判为故障会被扣分。

## 输出文件

你需要在任务根目录生成 5 个文件。

### 1. `answer.csv`

必须包含以下列：

```csv
record_id,well_id,time_days,anomaly_label,confidence,event_id,root_cause
```

字段说明：

- `record_id`：必须与 `eval_monitoring_noisy.csv` 中的记录对应。
- `well_id`：监测井编号。
- `time_days`：观测时间。
- `anomaly_label`：异常标签，必须来自允许标签集合。
- `confidence`：置信度，建议在 `[0, 1]` 范围内。
- `event_id`：事件编号。连续异常应尽量归并到同一事件。
- `root_cause`：简短说明判断依据，例如 `single-point spike inconsistent with neighboring wells`。

### 2. `cleaned_monitoring_data.csv`

必须包含以下列：

```csv
record_id,well_id,time_days,cleaned_concentration_mg_L,cleaning_action
```

字段说明：

- `cleaned_concentration_mg_L`：清洗/修复后的浓度值，必须为有限数值。
- `cleaning_action`：清洗动作，例如 `none`、`rolling_median_replace`、`neighbor_interpolation`、`time_shift_corrected`、`scaled_unit_error`。

### 3. `fault_events.csv`

必须包含以下列：

```csv
event_id,anomaly_label,well_id,start_day,end_day,confidence,evidence
```

该文件用于事件级评分。你应将连续异常记录合并为事件，而不是只逐行输出标签。

### 4. `fault_types.csv`

必须包含以下列：

```csv
anomaly_label,description,detection_rule,physical_rationale
```

该文件用于说明你的类型定义、检测规则和 Borden 物理依据。

### 5. `fault_report.md`

报告应说明：

- 总体数据质量情况；
- 主要异常类型和数量；
- 关键故障井和异常时间段；
- 如何区分真实污染羽到达和传感器故障；
- 清洗策略；
- 不确定性和可能误判来源。

## 推荐工作流

先生成合法 baseline：

```bash
python baseline_detector.py
```

在公开验证集上调试：

```bash
python baseline_detector.py \
  --input public_validation_noisy.csv \
  --answer public_answer.csv \
  --cleaned public_cleaned.csv \
  --report public_fault_report.md

python tools/evaluate_public.py \
  --answer public_answer.csv \
  --cleaned public_cleaned.csv
```

最终提交前，确保你的方法已经对 `eval_monitoring_noisy.csv` 生成任务根目录下的正式输出：

```bash
python your_detector.py
ls answer.csv cleaned_monitoring_data.csv fault_events.csv fault_types.csv fault_report.md
```

## 方法建议

可以从以下方向逐步改进：

- 使用 rolling median / MAD 识别 spike、negative、stuck_zero；
- 按 `well_id` 分组分析趋势、斜率、峰值时间；
- 使用相邻井和同层井比较判断空间一致性；
- 使用主流向 `+x` 判断上游/下游峰值时间顺序；
- 使用 cross-correlation 或峰值对齐检测 `time_shift`；
- 对比曲线形态与井坐标/深度，识别 `coordinate_or_depth_error`；
- 对低于检测限或缺测记录使用保守插值；
- 将连续异常合并成事件；
- 对真实污染羽到达保持保守，不要把所有快速上升都判为故障。

## 评分概要

最终 Judge 使用隐藏标签和隐藏 clean concentration 评分，总分 100：

```text
format：3
record anomaly detection：15
event detection：15
type classification：20
cleaned data quality：25
physical consistency：10
report：5
robustness/generalization：7
```

评分采用基线扣除的非线性函数。中等 F1 不会线性换成高分，必须超过较高阈值才会明显得分。

存在以下分数上限：

- `fault_events.csv` 缺失或不合规：总分上限 15；
- `cleaned_monitoring_data.csv` 缺失或不合规：总分上限 20；
- 清洗数据相对原始观测没有有效改善：总分上限 30；
- 类型分类不足：总分上限 35；
- 报告缺失或无实质内容：总分上限 60。

Judge stdout 只返回粗粒度反馈，例如：

```text
TOTAL_SCORE ...
TASK_RESULT ...
SCORE_BREAKDOWN_JSON ...
QC_FEEDBACK detection=low classification=low event=low cleaning=missing_or_poor
PHYSICS_FEEDBACK consistency=low
```

不会返回每类隐藏召回率或隐藏答案。

## 约束

- 不允许联网；
- 不需要 GPU；
- 不要读取 `scoring/`、隐藏标签或 Judge 文件；
- 不要硬编码隐藏记录、隐藏标签或评分结果；
- 不要调用 MODFLOW、MT3DMS、FloPy 等外部地下水模拟程序；
- 可以使用 NumPy、pandas、SciPy、scikit-learn、matplotlib；
- 提交时应保留生成输出所需的脚本。

## 注意事项

- `train_fault_labels.csv` 和 `public_fault_labels_small.csv` 都只是少量公开标签，不是完整答案。
- 公开验证分数只用于调试，不代表隐藏分数。
- 高分必须同时做好异常检测、事件合并、类型归因、清洗重建和物理一致性判断。
- 单纯阈值检测通常只能得到低分。
