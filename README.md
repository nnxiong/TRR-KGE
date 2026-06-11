## Directory Structure
```
TRR-KGE/
├── preprocess/                 # Data Preprocessing: Generate *.pickle / history / to_skip / stat
│   ├── process_datasets.py
│   └── process_icews.py
├── rule/                       # Stage 1: Rule Mining & (Optional) Rule Inference Evaluation
│   ├── mining.py               # Main entry for rule mining (parallel execution)
│   ├── rule_mining.py          # Rule construction, confidence estimation and saving
│   ├── temporal_walk.py        # Temporal walk sampling
│   ├── grapher2.py             # Load pickle data and automatically add inverse edges
│   ├── apply.py                # (Optional) Generate candidates using rules only
│   └── evaluate_new.py         # (Optional) Evaluate rule-based candidates
├── models/                     # Stage 2: Rule-based Training (CTRule)
│   ├── learner.py              # Main training entry
│   ├── datasets.py              # Load pickle data, training loop and evaluation
│   ├── models.py               # Model implementation (referenced by learner.py by default)
│   ├── rule_utils.py           # Rule filtering & pruning tools
│   └── contrastive_learning.py # Contrastive learning loss for historical intersection
└── (outputs)
    ├── data/<DATASET>/         # Preprocessed data (to be prepared/generated manually)
    └── rule_result/<DATASET>/  # Rule files (generated in Stage 1)
```

---

## Environment Dependencies

Python 3.8+ is recommended. Core dependencies are listed below (subject to actual environment):

- `torch`
- `numpy`
- `tqdm`
- `joblib`
- `pandas` (Used for rule and candidate modules)
- `pickle` (Standard Library)

---

## Data Preparation

Both model and rule modules read data from `../data/<DATASET>/` by default (relative to the script directory).

### 1) Raw Data Format

The preprocessing scripts require three **extension-free** files under `data/<DATASET>/`:
- `train`
- `valid`
- `test`

Each line contains a quadruple in the following format:
```
subject<TAB>rel<TAB>object<TAB>timestamp
```

Two scenarios:
- **If subject/object/relation/timestamp are integer IDs**: Use `preprocess/process_datasets.py`
- **If subject/rel/object/timestamp are strings (e.g. raw entities and relations of ICEWS)**: Use `preprocess/process_icews.py` (It will generate `ent2id.json/rel2id.json/ts2id.json` and convert content to integers)

### 2) Run Preprocessing to Generate Pickle Files for Training

Execute commands under the root directory of the repository (recommended):
```bash
cd preprocess

# Case A: Data are integers (lhs/rel/rhs/timestamp are all integers)
python process_datasets.py

# Case B: Data are strings (will map content to IDs first)
python process_icews.py
```

After preprocessing, the `data/<DATASET>/` folder will contain the following files:
- `train.pickle / valid.pickle / test.pickle`: `int` array with shape `(N,4)`
- `to_skip.pickle`: Answer set for evaluation filtering (two types of missing values for lhs/rhs)
- `history.pickle`: Historical event list for training (indexed by (entity, rel, ts))
- `stat`: Data statistics (count of entities, relations and timestamps)

---

## Rule Mining

Rule Mining:
Perform temporal walk on the graph, mine rules such as **Symmetry / Inverse / Equivalent / Transitive(k-hop)**, calculate statistical metrics including confidence and support, and export results as JSON rule files.

- Entry for rule mining: `rule/mining.py`
- Output directory: `rule_result/<DATASET>/`

### Run Example
```bash
cd rule

# Example: Mine rules on ICEWS14, max path length = 5, 100 random walks, 10 parallel processes, keep top-2 rules for each head relation
python mining.py --dataset ICEWS14 --rule_lengths 5 --num_walks 100 --num_processes 10 --top 2
```

Common parameters (refer to `rule/mining.py`):
- `--dataset / -d`: Dataset name (corresponds to `../data/<DATASET>/`)
- `--rule_lengths / -l`: Maximum rule length (Transitive rules range from 2-hop to the set maximum length)
- `--num_walks / -n`: Sampling walk times for each relation type
- `--transition_distr / -t`: Walk sampling distribution (`exp` by default)
- `--num_processes / -p`: Number of parallel processes
- `--top / -top`: Number of reserved rules for each head relation (sorted and truncated by confidence)

### Output Rule File

Rule files are named in the following format:
```
rule_result/ICEWS14/<timestamp>_maxlen[5]_top2_rules.json
```

Simplified file structure:
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

## Model Training

Training entry: `models/learner.py`
Default model implementation: `CTRule` in `models/models6.py`

### Run Example
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

Training outputs are saved to:
```
models/results/<DATASET>/<...timestamp...>/
├── log.txt
└── best.pth
```

Common parameters (refer to `models/learner.py`):
- `--dataset`: Dataset name (corresponds to `../data/<DATASET>/`)
- `--rank`: Embedding dimension (Note: Complex/quaternion block operations are implemented inside the model, the rank must comply with code constraints)
- `--learning_rate`: Learning rate for Adagrad optimizer
- `--lambda1`: Weight of contrastive learning loss (defined in `contrastive_learning.py`)
- `--rules_dir`: Directory of rule files (`../rule_result/` by default)
- `--rules`: Name of the rule file (stored under `rules_dir/<dataset>/`)
- `--gpu`: Enable CUDA or not (1 for enable, 0 for disable)
- `--cuda`: Device string (e.g. `cuda:0`)

---

## General Notes

1. **Execute scripts in corresponding directories**
   Rule mining: `cd rule && python mining.py ...`
   Model training: `cd models && python learner.py ...`

2. **CUDA Device ID**
   Part of the code hardcodes `"cuda:0"` during evaluation and tensor initialization (e.g. target generation in `models6.py`). If you use multiple GPUs or non-zero device ID, minor code modifications are required to uniformly use `args.cuda`.

3. **Run preprocessing first**
   Missing files including `stat`, `*.pickle`, `history.pickle` and `to_skip.pickle` will directly cause runtime errors.

---

## Quick Reproduction Workflow (Summary)
```bash
# 0. Prepare files: data/<DATASET>/{train,valid,test}

# 1) Data Preprocessing
cd preprocess
python process_datasets.py   # Or run python process_icews.py

# 2) Rule Mining
cd ../rule
python mining.py --dataset ICEWS14 --rule_lengths 5 --num_walks 100 --num_processes 10 --top 2

# 3) Rule-based Model Training
cd ../models
python learner.py --dataset ICEWS14 --rank 1000 --learning_rate 0.1 --lambda1 0.06 \
  --rules_dir ../rule_result/ --rules "<timestamp>_maxlen[5]_top2_rules.json" --gpu 1 --cuda cuda:0
```
