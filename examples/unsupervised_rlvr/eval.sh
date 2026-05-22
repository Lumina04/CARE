set -x

export PYTHONUNBUFFERED=1
export PROJECT_NAME='URLVR'
export REPO_ROOT=xxx/al-rlvr
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT"
export DATA_DIR=xxx
export TENSORBOARD_DIR="xxx/al-rlvr/tensorboard_log/test_log/1.7b_base"
export REWARD_TYPE=gt

ZERO=False
export MAX_RESP_LENGTH=4000
export MAX_VAL_RESP_LENGTH=4000
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-32}
export TEMPERATURE=${TEMPERATURE:-1.0}
export N_RESPONSES=8
export USE_KL=${USE_KL:-True}

export EXPERIMENT_NAME=gt-1.7b-${REWARD_TYPE}

TRAIN_DATASET=${TRAIN_FILE:-["$DATA_DIR/train_data/dapo/train.parquet"]}
TEST_DATASET=${TEST_FILE:-["$DATA_DIR/test_data/aime24.parquet","$DATA_DIR/test_data/aime25.parquet","$DATA_DIR/test_data/amc23.parquet","$DATA_DIR/test_data/olympiad.parquet","$DATA_DIR/test_data/hmmt25.parquet","$DATA_DIR/test_data/math500.parquet"]}

# TODO: Set your model path
export ACTOR_MODEL_PATH=xxx/llm_model/Qwen3-1.7B-Base
export PROJECT_PATH=xxx/save_checkpoints
export LOG_FILE=${PROJECT_PATH}/logs/${EXPERIMENT_NAME}.log

export PARALLEL_SIZE=1
export CKPT_PATH=${PROJECT_PATH}/checkpoints
export OUTLINES_CACHE_DIR=~/.cache/outlines/$(uuidgen)
export NCCL_DEBUG=WARN

export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1

echo "[seed] GLOBAL_SEED=${GLOBAL_SEED}"

unset ROCR_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

KL_ARGS=""
if [ "$USE_KL" = "True" ]; then
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl"
else
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=False"
fi

LR_ARGS=""
if [ "$LR_SCHEDULER" = "cosine" ]; then
    LR_ARGS="actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03"
fi

PPO_MAX_TOKEN_LEN_PER_GPU=$(( ((1024 + MAX_RESP_LENGTH) > 32768) ? (1024 + MAX_RESP_LENGTH) : 32768))
echo "PPO_MAX_TOKEN_LEN_PER_GPU: $PPO_MAX_TOKEN_LEN_PER_GPU"

python3 -m verl.trainer.main_ppo \
    --config-name='ppo_trainer_ttrl.yaml'\
    algorithm.adv_estimator=grpo \
    +data.zero=$ZERO \
    data.shuffle=True \
    +data.seed=1 \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TEST_DATASET" \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=$MAX_RESP_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    data.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    $LR_ARGS \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    $KL_ARGS \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=$N_RESPONSES \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.max_new_tokens=$MAX_VAL_RESP_LENGTH \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    +trainer.val_dataset_rollout_n.math500=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.enable=False \
    reward_model.reward_manager=naive \
    unsupervised_reward.enable=False \
    custom_reward_function.path="$REPO_ROOT/verl/utils/reward_score/ttrl_math/__init__.py" \
    custom_reward_function.name=reward_func \
    trainer.val_before_train=True \
    trainer.log_val_generations=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=20 \
    trainer.total_epochs=1 \
    trainer.val_only=True \
    trainer.default_local_dir="$CKPT_PATH"/"$EXPERIMENT_NAME" \
    2>&1 | tee -a "$LOG_FILE"

# pip3 install latex2sympy2_extended[antlr4_13_2]
# pip3 install math-verify[antlr4_13_2]
# pip3 install --upgrade omegaconf
# pip3 install pylatexenc
# xxx/.local/bin/hope run ./hope/run.hope


