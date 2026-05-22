#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid 
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator
# ==================== only edit these 3 ====================
MODEL_NAME_PREFIX = "phi-random-mask-run2"
CHECKPOINT_DIR = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/wangli/save_checkpoints/checkpoints/random_gt_zero_adv-top_0p2-mask_non_gt_True-phi-run2"
HF_MODEL_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/wangli/llm_model/Phi-4-mini-instruct"
# ==================== defaults (aligned with eval.sh) ====================
REPO_ROOT = Path(__file__).resolve().parents[1]
REWARD_FUNCTION_PATH = REPO_ROOT / "verl" / "utils" / "reward_score" / "ttrl_math" / "__init__.py"
DATA_DIR = Path("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/wangli/datasets")
PROJECT_PATH = Path("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/wangli/save_checkpoints")

SAVE_MODEL_ROOT = Path("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/wangli/save_model/al-rlvr") / MODEL_NAME_PREFIX
TB_ROOT = Path("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/wangli/al-rlvr/tensorboard_log/test_log") / MODEL_NAME_PREFIX
RESULTS_JSON = TB_ROOT / "all_eval_results.json"

DATASET_NAMES = ["aime24", "aime25", "amc23", "olympiad", "hmmt25", "math500"]
STEPS = None

SKIP_CONVERT = False
SKIP_EVAL = False
SLEEP_AFTER_EVAL = 10

PROJECT_NAME = "URLVR"
REWARD_TYPE = "gt"
ZERO = False
MAX_RESP_LENGTH = 4000
MAX_VAL_RESP_LENGTH = 4000
MINI_BATCH_SIZE = 32
TEMPERATURE = 1.0
N_RESPONSES = 8
USE_KL = False
LR_SCHEDULER = ""
PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.75
CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"


def _b(v: bool) -> str:
    return "True" if v else "False"


def _run(cmd, cwd=None, env=None):
    pretty = " ".join(shlex.quote(str(x)) for x in cmd)
    print(f"[cmd] {pretty}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)
    return


def _is_hf_model_dir(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    if not (model_dir / "config.json").exists():
        return False
    hf_weight_files = [
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    ]
    return any((model_dir / f).exists() for f in hf_weight_files)


def _collect_step_dirs(ckpt_dir: Path):
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoint dir not found: {ckpt_dir}")

    entries = []
    p_global = re.compile(r"global_step_(\d+)$")
    p_step = re.compile(r".*step[_-]?(\d+)$")

    for d in ckpt_dir.iterdir():
        if not d.is_dir():
            continue

        step = None
        m = p_global.match(d.name)
        if m:
            step = int(m.group(1))
        else:
            m = p_step.match(d.name)
            if m:
                step = int(m.group(1))

        if step is None:
            continue

        if _is_hf_model_dir(d):
            entries.append((step, d, "hf"))
            continue

        if (d / "actor").exists():
            entries.append((step, d, "verl"))
            continue

    # fallback: checkpoint_dir itself is a single HF model directory
    if not entries and _is_hf_model_dir(ckpt_dir):
        entries.append((0, ckpt_dir, "hf"))

    # de-duplicate same step id, prefer hf if mixed
    dedup = {}
    for step, step_dir, step_format in sorted(entries, key=lambda x: x[0]):
        if step not in dedup or step_format == "hf":
            dedup[step] = (step_dir, step_format)

    return [(step, dedup[step][0], dedup[step][1]) for step in sorted(dedup.keys())]


def _hydra_list(paths: list[str]) -> str:
    return "[" + ",".join(paths) + "]"


def _merge(step: int, step_dir: Path, hf_model_path: Path):
    src_actor = step_dir / "actor"
    if not src_actor.exists():
        raise FileNotFoundError(f"actor checkpoint not found: {src_actor}")

    target_dir = SAVE_MODEL_ROOT / f"{MODEL_NAME_PREFIX}-step{step}"
    if (target_dir / "config.json").exists():
        print(f"[skip] step={step} already converted: {target_dir}")
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/legacy_model_merger.py",
        "merge",
        "--backend", "fsdp",
        "--local_dir", str(src_actor),
        "--target_dir", str(target_dir),
        "--hf_model_path", str(hf_model_path),
    ]
    _run(cmd, cwd=REPO_ROOT)
    return target_dir


def _eval_cmd(actor_model_path: Path, exp_name: str):
    train_files = _hydra_list([str(DATA_DIR / "train_data" / "dapo" / "train.parquet")])
    val_files = _hydra_list([str(DATA_DIR / "test_data" / f"{n}.parquet") for n in DATASET_NAMES])
    ppo_max = max(1024 + MAX_RESP_LENGTH, 32768)

    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        "--config-name=ppo_trainer_ttrl.yaml",
        "algorithm.adv_estimator=grpo",
        f"+data.zero={_b(ZERO)}",
        "data.shuffle=False",
        f"data.train_files={train_files}",
        f"data.val_files={val_files}",
        "data.train_batch_size=64",
        "data.max_prompt_length=1024",
        f"data.max_response_length={MAX_RESP_LENGTH}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        f"actor_rollout_ref.model.path={actor_model_path}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_activation_offload=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={MINI_BATCH_SIZE}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={ppo_max}",
        f"actor_rollout_ref.actor.ulysses_sequence_parallel_size={PARALLEL_SIZE}",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.actor.fsdp_config.forward_prefetch=True",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={ppo_max}",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.temperature={TEMPERATURE}",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={PARALLEL_SIZE}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={GPU_MEMORY_UTILIZATION}",
        f"actor_rollout_ref.rollout.n={N_RESPONSES}",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        f"+actor_rollout_ref.rollout.val_kwargs.max_new_tokens={MAX_VAL_RESP_LENGTH}",
        "actor_rollout_ref.rollout.val_kwargs.n=32",
        f"+trainer.val_dataset_rollout_n.math500=8",
        "actor_rollout_ref.rollout.val_kwargs.temperature=0.6",
        "actor_rollout_ref.rollout.val_kwargs.top_p=0.95",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "reward_model.enable=False",
        "reward_model.reward_manager=naive",
        "unsupervised_reward.enable=False",
        f"custom_reward_function.path={REWARD_FUNCTION_PATH}",
        "custom_reward_function.name=reward_func",
        "trainer.val_before_train=True",
        "trainer.log_val_generations=0",
        "trainer.logger=['console','tensorboard']",
        f"trainer.project_name={PROJECT_NAME}",
        f"trainer.experiment_name={exp_name}",
        "trainer.n_gpus_per_node=8",
        "trainer.nnodes=1",
        "trainer.save_freq=40",
        "trainer.test_freq=20",
        "trainer.total_epochs=1",
        "trainer.val_only=True",
        f"trainer.default_local_dir={PROJECT_PATH / 'checkpoints' / PROJECT_NAME / exp_name}",
    ]

    if LR_SCHEDULER == "cosine":
        cmd += [
            "actor_rollout_ref.actor.optim.warmup_style=cosine",
            "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03",
        ]
    if USE_KL:
        cmd += [
            "actor_rollout_ref.actor.use_kl_loss=True",
            "actor_rollout_ref.actor.kl_loss_coef=0.005",
            "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        ]
    else:
        cmd.append("actor_rollout_ref.actor.use_kl_loss=False")
    return cmd


def _env(tb_dir: Path):
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PROJECT_NAME"] = PROJECT_NAME
    env["DATA_DIR"] = str(DATA_DIR)
    env["TENSORBOARD_DIR"] = str(tb_dir)
    env["REWARD_TYPE"] = REWARD_TYPE
    env["MAX_RESP_LENGTH"] = str(MAX_RESP_LENGTH)
    env["MAX_VAL_RESP_LENGTH"] = str(MAX_VAL_RESP_LENGTH)
    env["MINI_BATCH_SIZE"] = str(MINI_BATCH_SIZE)
    env["TEMPERATURE"] = str(TEMPERATURE)
    env["N_RESPONSES"] = str(N_RESPONSES)
    env["USE_KL"] = _b(USE_KL)
    env["PARALLEL_SIZE"] = str(PARALLEL_SIZE)
    env["PROJECT_PATH"] = str(PROJECT_PATH)
    env["CKPT_PATH"] = str(PROJECT_PATH / "checkpoints")
    env["NCCL_DEBUG"] = "WARN"
    env["TOKENIZERS_PARALLELISM"] = "true"
    env["HYDRA_FULL_ERROR"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    env["OUTLINES_CACHE_DIR"] = str(Path.home() / ".cache" / "outlines" / str(uuid.uuid4()))
    env.pop("ROCR_VISIBLE_DEVICES", None)
    return env


def _scores(tb_dir: Path):
    event_files = [p for p in tb_dir.rglob("events.out.tfevents*") if p.is_file()]
    if not event_files:
        return {}
    res = {}
    for ef in event_files:
        try:
            ea = event_accumulator.EventAccumulator(str(ef))
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            filt = [t for t in tags if "val-core" in t and "mean@" in t]
            for tag in filt:
                s = ea.Scalars(tag)
                if s:
                    res[tag] = s[-1].value
        except Exception as e:
            print(f"[warn] parse {ef} failed: {e}")
    if res:
        res["average_accuracy"] = sum(res.values()) / len(res)
    return res


def _save(step: int, payload: dict):
    data = {}
    if RESULTS_JSON.exists():
        try:
            data = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    data[str(step)] = payload
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    checkpoint_dir = Path(CHECKPOINT_DIR).resolve()
    hf_model_path = Path(HF_MODEL_PATH).resolve()

    step_entries = _collect_step_dirs(checkpoint_dir)
    step_map = {step: (step_dir, step_format) for step, step_dir, step_format in step_entries}
    run_steps = sorted(step_map.keys()) if STEPS is None else sorted(set(int(x) for x in STEPS))

    print(f"[info] checkpoint_dir={checkpoint_dir}")
    print(f"[info] detected_steps={[(s, step_map[s][1]) for s in sorted(step_map.keys())]}")
    print(f"[info] run_steps={run_steps}")
    if not step_entries:
        print("[warn] no step dirs found")
        return

    SAVE_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    TB_ROOT.mkdir(parents=True, exist_ok=True)
    (PROJECT_PATH / "logs").mkdir(parents=True, exist_ok=True)

    for step in run_steps:
        print("=" * 80)
        print(f"[step] {step}")
        print("=" * 80)
        try:
            if step not in step_map:
                raise ValueError(f"step={step} not found. Available steps: {sorted(step_map.keys())}")

            step_dir, step_format = step_map[step]
            print(f"[info] step={step} format={step_format} step_dir={step_dir}")

            if step_format == "hf":
                model_dir = step_dir
                print(f"[skip] step={step} detected HF format, skip conversion: {model_dir}")
            elif SKIP_CONVERT:
                model_dir = SAVE_MODEL_ROOT / f"{MODEL_NAME_PREFIX}-step{step}"
                if not (model_dir / "config.json").exists():
                    raise FileNotFoundError(f"merged model missing: {model_dir}")
            else:
                model_dir = _merge(step, step_dir, hf_model_path)

            tb_dir = TB_ROOT / f"step_{step}"
            tb_dir.mkdir(parents=True, exist_ok=True)
            exp_name = f"gt-{REWARD_TYPE}-step{step}"

            if not SKIP_EVAL:
                _run(_eval_cmd(model_dir, exp_name), cwd=REPO_ROOT, env=_env(tb_dir))
                if SLEEP_AFTER_EVAL > 0:
                    time.sleep(SLEEP_AFTER_EVAL)

            sc = _scores(tb_dir)
            _save(step, {
                "model_dir": str(model_dir),
                "tb_dir": str(tb_dir),
                "scores": sc,
            })

            if sc:
                print(f"[done] step={step} average_accuracy={sc.get('average_accuracy', 0):.6f}")
            else:
                print(f"[done] step={step} no scalar score found")

        except Exception as e:
            print(f"[error] step={step} failed: {e}")
            _save(step, {"error": str(e)})

    print(f"[finish] results saved to: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
