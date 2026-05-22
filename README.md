# When Self-Belief Misleads: Active Label Acquisition for Reinforcement Learning with Verifiable Rewards

This repository contains the code for running CARE. The codebase supports two experiment tracks:

| Track | Task | Training scripts | Reward |
| --- | --- | --- | --- |
| **Math** | DAPO-Math RLVR | `examples/unsupervised_rlvr/` | `verl/utils/reward_score/ttrl_math/` |
| **KK** | Knights-and-Knaves logic puzzles ([Logic-RL](https://github.com/Unakar/Logic-RL) style) | `examples/unsupervised_rlvr_kk/` | `verl/utils/reward_score/kk.py` |

## Environment Setup

Install the package from the repository root. The environment is configured through `setup.py`.

```bash
cd CARE
pip3 install -e .
```

---

## Math Track

### Data Preparation

Download the DAPO-Math-17k dataset from Hugging Face:

```bash
mv "$DATA_DIR/train_data/dapo/data/dapo-math-17k.parquet" \
   "$DATA_DIR/train_data/dapo/train.parquet"
```

The default scripts also expect evaluation files under `$DATA_DIR/test_data/`:

```text
aime24.parquet
aime25.parquet
amc23.parquet
olympiad.parquet
hmmt25.parquet
math500.parquet
```

If your evaluation files are stored elsewhere, override `TEST_FILE` when launching the script.

### Configure Training

Edit `examples/unsupervised_rlvr/al-rule.sh` before running.

```bash
export DATA_DIR=/path/to/data
export PROJECT_ROOT=/path/to/CARE
export TENSORBOARD_DIR=/path/to/tensorboard
export ACTOR_MODEL_PATH=/path/to/base/model
export PROJECT_PATH=/path/to/output/project
```

CARE hyperparameters are set at the top of `examples/unsupervised_rlvr/al-rule.sh`. The first-stage unsupervised sample selection ratio $p_2$ is `PROBE_PROB_GATE`.

### Run

After setting the model path, dataset path, project root, and output path, launch training from the repository root:

```bash
bash examples/unsupervised_rlvr/al-rule.sh
```

---

## KK Track (Knights-and-Knaves Logic)

The KK branch trains and evaluates on knights-and-knaves (Logic-RL style) puzzles. Models are scored with format + answer rewards defined in `verl/utils/reward_score/kk.py`: the model must place the final role assignment inside `<answer>...</answer>` (canonical form: `name=knight|name=knave`).

### Data Preparation

Download the KK dataset from Hugging Face ([K-and-K/knights-and-knaves](https://huggingface.co/datasets/K-and-K/knights-and-knaves)) and place the parquet files under `$DATA_DIR`.

**Training**

```text
$DATA_DIR/train_data/kk_train.parquet
```

**Evaluation** (test sets with 3–8 people per puzzle):

```text
$DATA_DIR/test_data/3ppl_test.parquet
$DATA_DIR/test_data/4ppl_test.parquet
$DATA_DIR/test_data/5ppl_test.parquet
$DATA_DIR/test_data/6ppl_test.parquet
$DATA_DIR/test_data/7ppl_test.parquet
$DATA_DIR/test_data/8ppl_test.parquet
```

### Configure Training

Edit `examples/unsupervised_rlvr_kk/al-rule.sh` (or another script in that folder). At minimum, set:

```bash
export DATA_DIR=/path/to/data
export PROJECT_ROOT=/path/to/CARE
export TENSORBOARD_DIR=/path/to/tensorboard
export ACTOR_MODEL_PATH=/path/to/base/model
export PROJECT_PATH=/path/to/output/project
```

Probe hyperparameters are configured the same way in `examples/unsupervised_rlvr_kk/al-rule.sh`.

### Run

From the repository root, after editing paths in the script:

```bash
# CARE on KK
bash examples/unsupervised_rlvr_kk/al-rule.sh
```
