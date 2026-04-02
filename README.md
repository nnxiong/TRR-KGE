---
## 目录结构

```
Combine-Rule_git/
├── preprocess/                 # 数据预处理：生成 *.pickle / history / to_skip / stat
│   ├── process_datasets.py
│   └── process_icews.py
├── rule/                       # 阶段1：规则挖掘与（可选）规则推理评估
│   ├── mining.py               # 规则挖掘主入口（并行）
│   ├── rule_mining.py          # 规则构造、置信度估计、保存
│   ├── temporal_walk.py        # 时间游走采样
│   ├── grapher2.py             # 读取pickle数据并自动加 inverse 边
│   ├── apply.py                # (可选) 仅用规则生成候选
│   └── evaluate_new.py         # (可选) 规则候选评估
├── models/                     # 阶段2：基于规则训练（CTRule）
│   ├── learner.py              # 训练主入口
│   ├── datasets.py             # 读取pickle数据、训练循环、评测
│   ├── models.py              # 模型实现（默认 learner.py 引用）
│   ├── rule_utils.py           # 规则过滤/裁剪工具
│   └── contrastive_learning.py # 历史交集对比学习损失
└── (outputs)
    ├── data/<DATASET>/         # 预处理后的数据（你需要准备/生成）
    └── rule_result/<DATASET>/  # 规则文件（阶段1生成）
```


---

## 环境依赖

推荐 Python 3.8+，核心依赖大致如下（以实际环境为准）：

- `torch`
- `numpy`
- `tqdm`
- `joblib`
- `pandas`（规则/候选部分有用到）
- `pickle`（标准库）

你可以用类似方式安装（仅供参考）：

```bash
pip install torch numpy tqdm joblib pandas
```

---

## 数据准备（必须）

模型与规则模块都默认从 `../data/<DATASET>/` 读取数据（相对脚本所在目录）。

### 1) 原始数据格式

预处理脚本期望你在 `data/<DATASET>/` 下有三个文件（**无扩展名**）：

- `train`
- `valid`
- `test`

每行一个四元组：

```
lhs<TAB>rel<TAB>rhs<TAB>timestamp
```

两种情况：

- **如果 lhs/rel/rhs/timestamp 已经是整数 ID**：用 `preprocess/process_datasets.py`
- **如果 lhs/rel/rhs/timestamp 是字符串（例如 ICEWS 的原始实体/关系）**：用 `preprocess/process_icews.py`（会生成 `ent2id.json/rel2id.json/ts2id.json` 再转成整数）

### 2) 运行预处理，生成训练所需的 pickle

从仓库根目录执行（推荐）：

```bash
cd preprocess

# 情况A：数据已经是 int（lhs/rel/rhs/timestamp 都是整数）
python process_datasets.py

# 情况B：数据是字符串（会先映射到 id）
python process_icews.py
```

预处理后，`data/<DATASET>/` 里通常会生成/需要包含：

- `train.pickle / valid.pickle / test.pickle`：`(N,4)` 的 `int` 数组
- `to_skip.pickle`：过滤评测用的答案集合（lhs/rhs 两种缺失）
- `history.pickle`：训练时用到的历史事件列表（按 (entity, rel, ts) 索引）
- `stat`：数据统计（实体/关系/时间戳数量）

---

## 规则挖掘（Rule Mining）

规则生成（Rule Mining）：
在训练图上做时间游走（temporal walk），挖掘 **Symmetry / Inverse / Equivalent / Transitive(k-hop)** 等规则，并计算置信度（confidence）、support 等统计量，输出为 `json` 规则文件。

- 规则挖掘入口：`rule/mining.py`  
- 输出目录：`rule_result/<DATASET>/`

### 运行示例

```bash
cd rule

# 例：在 ICEWS14 上挖掘规则，最长路径长度 5，随机游走 100 次，并行 10 进程，每个 head relation 取 top-2
python mining.py --dataset ICEWS14 --rule_lengths 5 --num_walks 100 --num_processes 10 --top 2
```

常用参数（见 `rule/mining.py`）：

- `--dataset / -d`：数据集名（对应 `../data/<DATASET>/`）
- `--rule_lengths / -l`：最大规则长度（Transitive 会从 2-hop 到 maxlen）
- `--num_walks / -n`：每个关系/类型的采样游走次数
- `--transition_distr / -t`：游走采样分布（默认 `exp`）
- `--num_processes / -p`：并行进程数
- `--top / -top`：每个 head relation 保留的规则条数（按 conf 排序截断）

### 输出规则文件

规则会保存为类似文件名：

```
rule_result/ICEWS14/<timestamp>_maxlen[5]_top2_rules.json
```

文件内容结构（简化）：

```json
{
  "head_rel_id": [
    {
      "rule_type": "Transitive3",
      "head_rel": 12,
      "body_rels": [5, 7, 9],
      "conf": 0.123456,
      "rule_supp": 10,
      "body_supp": 81,
      "var_constraints": [[0,2]]
    }
  ]
}
```

---

## 模型训练

训练入口：`models/learner.py`  
默认使用的模型实现：`models/models6.py` 中的 `CTRule`

### 运行示例

```bash
cd models

python learner.py \
  --dataset ICEWS14 \
  --rank 1000 \
  --learning_rate 0.1 \
  --lambda1 0.06 \
  --rules_dir ../rule_result/ \
  --rules "<your_rule_file>.json" \
  --gpu 1 \
  --cuda cuda:0
```

训练输出会写到：

```
models/results/<DATASET>/<...timestamp...>/
├── log.txt
└── best.pth
```

常用参数（见 `models/learner.py`）：

- `--dataset`：数据集名（对应 `../data/<DATASET>/`）
- `--rank`：embedding 维度（注意模型内部做了复数/四元数分块运算，rank 需满足代码约束）
- `--learning_rate`：Adagrad 学习率
- `--lambda1`：对比学习损失权重（`contrastive_learning.py`）
- `--rules_dir`：规则目录（默认 `../rule_result/`）
- `--rules`：规则文件名（位于 `rules_dir/<dataset>/`）
- `--gpu`：是否启用 CUDA（1/0）
- `--cuda`：设备字符串（例如 `cuda:0`）



---

## 常见注意事项

1. **请从对应目录执行脚本**  
   规则挖掘：`cd rule && python mining.py ...`  
   模型训练：`cd models && python learner.py ...`  

2. **CUDA 设备号**  
   部分代码在评测/张量创建时写死了 `"cuda:0"`（例如 `models6.py` 里生成 `targets`），如果你用多卡或非 0 号卡，可能需要小改代码统一用 `args.cuda`。

3. **数据必须先跑预处理**  
   否则 `stat / *.pickle / history.pickle / to_skip.pickle` 不存在会直接报错。

---

## 快速复现流程（汇总）

```bash
# 0. 准备 data/<DATASET>/{train,valid,test}

# 1) 预处理
cd preprocess
python process_datasets.py   # 或 python process_icews.py

# 2) 规则挖掘
cd ../rule
python mining.py --dataset ICEWS14 --rule_lengths 5 --num_walks 100 --num_processes 10 --top 2

# 3) 基于规则训练
cd ../models
python learner.py --dataset ICEWS14 --rank 1000 --learning_rate 0.1 --lambda1 0.06 \
  --rules_dir ../rule_result/ --rules "<timestamp>_maxlen[5]_top2_rules.json" --gpu 1 --cuda cuda:0
```

