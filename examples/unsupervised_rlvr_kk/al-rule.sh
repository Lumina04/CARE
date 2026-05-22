set -x

export PYTHONUNBUFFERED=1
export PROJECT_NAME='URLVR_KK'
export DATA_DIR=xxx
export PROJECT_ROOT=xxx/al-rlvr
export REPO_ROOT=$PROJECT_ROOT
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT"
export REWARD_FUNCTION_PATH=${REWARD_FUNCTION_PATH:-$REPO_ROOT/verl/utils/reward_score/kk.py}

export PROBE_GT_TOP_PCT=${PROBE_GT_TOP_PCT:-0.2}
export PROBE_PROB_GATE=${PROBE_PROB_GATE:-0.25}
export PROBE_PROMPT_HIDDEN_DIM=${PROBE_PROMPT_HIDDEN_DIM:-128}
export PROBE_PROMPT_MID_DIM=${PROBE_PROMPT_MID_DIM:-256}
export PROBE_RESP_MLP_HIDDEN_DIM=${PROBE_RESP_MLP_HIDDEN_DIM:-64}
export PROBE_RESP_MLP_OUT_DIM=${PROBE_RESP_MLP_OUT_DIM:-512}
export PROBE_ACTIVATION=${PROBE_ACTIVATION:-relu}
export PROBE_LR=${PROBE_LR:-1e-4}
export PROBE_TRAIN_STEPS=${PROBE_TRAIN_STEPS:-3}
export PROBE_WEIGHT_DECAY=${PROBE_WEIGHT_DECAY:-0.0}
export PROBE_GRAD_CLIP_NORM=${PROBE_GRAD_CLIP_NORM:-1.0}
export PROBE_AUX_LOSS_WEIGHT=${PROBE_AUX_LOSS_WEIGHT:-1.5}
export PROBE_CLASS_BALANCE_ENABLE=${PROBE_CLASS_BALANCE_ENABLE:-True}
export PROBE_CLASS_BALANCE_POWER=${PROBE_CLASS_BALANCE_POWER:-0.50}
export PROBE_CLASS_BALANCE_MIN=${PROBE_CLASS_BALANCE_MIN:-0.25}
export PROBE_CLASS_BALANCE_MAX=${PROBE_CLASS_BALANCE_MAX:-4.0}
# For the wobuffer ablation, disable replay by default.
export PROBE_BUFFER_SIZE=${PROBE_BUFFER_SIZE:-2048}
# If PROBE_BUFFER_SIZE>0, batch_size=0 means use the full history buffer.
export PROBE_BUFFER_BATCH_SIZE=${PROBE_BUFFER_BATCH_SIZE:-16}
# The probe is created inside the TaskRunner actor. Use cuda:0 by default
# without reserving an extra Ray GPU; RUNNER_NUM_GPUS=0 keeps trainer resources unchanged.
export PROBE_DEVICE=${PROBE_DEVICE:-cuda:0}
export RUNNER_NUM_GPUS=${RUNNER_NUM_GPUS:-0}
export SOFT_ADV=${SOFT_ADV:-1.0}
export REWARD_TYPE=${REWARD_TYPE:-grpo_adv_diff_probe_setscorer}
export TENSORBOARD_DIR="xxx/al-rlvr/tensorboard_log/train_log/8b_care_p_${PROBE_GT_TOP_PCT}_p2_${PROBE_PROB_GATE}"
if [[ "$TENSORBOARD_DIR" != *kk* && "$TENSORBOARD_DIR" != *KK* ]]; then
    export TENSORBOARD_DIR="${TENSORBOARD_DIR}-kk"
fi

ZERO=False
export MAX_RESP_LENGTH=4000
export MAX_VAL_RESP_LENGTH=4000
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-32}
export TEMPERATURE=${TEMPERATURE:-1.0}
export N_RESPONSES=8
export USE_KL=${USE_KL:-True}

export EXPERIMENT_NAME=care-p_${PROBE_GT_TOP_PCT}_p2_${PROBE_PROB_GATE}-8b
if [[ "$EXPERIMENT_NAME" != *kk* && "$EXPERIMENT_NAME" != *KK* ]]; then
    export EXPERIMENT_NAME="${EXPERIMENT_NAME}-kk"
fi

# ROLLOUT_DIR=xxx/al-rlvr/logs/phi_care_p_${PROBE_GT_TOP_PCT}_p2_${PROBE_PROB_GATE}

TRAIN_DATASET=${TRAIN_FILE:-["$DATA_DIR/train_data/kk_train.parquet"]}
TEST_DATASET=${TEST_FILE:-["$DATA_DIR/test_data/3ppl_test.parquet","$DATA_DIR/test_data/4ppl_test.parquet","$DATA_DIR/test_data/5ppl_test.parquet","$DATA_DIR/test_data/6ppl_test.parquet","$DATA_DIR/test_data/7ppl_test.parquet","$DATA_DIR/test_data/8ppl_test.parquet"]}

# TODO: Set your model path
export ACTOR_MODEL_PATH=xxx/llm_model/Qwen3-8B-Base
export PROJECT_PATH=xxx/save_checkpoints
export LOG_FILE=${PROJECT_PATH}/logs/${EXPERIMENT_NAME}.log

export PARALLEL_SIZE=1
export CKPT_PATH=${PROJECT_PATH}/checkpoints
export OUTLINES_CACHE_DIR=~/.cache/outlines/$(uuidgen)
export NCCL_v3=WARN

export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1

export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

echo "[seed] GLOBAL_SEED=${GLOBAL_SEED}"
echo "[soft_adv] SOFT_ADV=${SOFT_ADV}"
echo "[probe] PROBE_DEVICE=${PROBE_DEVICE}, RUNNER_NUM_GPUS=${RUNNER_NUM_GPUS}, PROBE_PROB_GATE=${PROBE_PROB_GATE}"

unset ROCR_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RUNNER_CUDA_VISIBLE_DEVICES=${RUNNER_CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}

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
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    +trainer.val_dataset_rollout_n.kk_logic_3ppl=1 \
    +trainer.val_dataset_rollout_n.kk_logic_4ppl=1 \
    +trainer.val_dataset_rollout_n.kk_logic_5ppl=1 \
    +trainer.val_dataset_rollout_n.kk_logic_6ppl=1 \
    +trainer.val_dataset_rollout_n.kk_logic_7ppl=1 \
    +trainer.val_dataset_rollout_n.kk_logic_8ppl=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.enable=False \
    reward_model.reward_manager=naive \
    unsupervised_reward.enable=True \
    unsupervised_reward.type=ensemble \
    unsupervised_reward.estimator=$REWARD_TYPE \
    ++unsupervised_reward.probe_gt_top_pct=$PROBE_GT_TOP_PCT \
    ++unsupervised_reward.probe_prob_gate=$PROBE_PROB_GATE \
    ++unsupervised_reward.probe_prompt_hidden_dim=$PROBE_PROMPT_HIDDEN_DIM \
    ++unsupervised_reward.probe_prompt_mid_dim=$PROBE_PROMPT_MID_DIM \
    ++unsupervised_reward.probe_resp_mlp_hidden_dim=$PROBE_RESP_MLP_HIDDEN_DIM \
    ++unsupervised_reward.probe_resp_mlp_out_dim=$PROBE_RESP_MLP_OUT_DIM \
    ++unsupervised_reward.probe_activation=$PROBE_ACTIVATION \
    ++unsupervised_reward.probe_lr=$PROBE_LR \
    ++unsupervised_reward.probe_weight_decay=$PROBE_WEIGHT_DECAY \
    ++unsupervised_reward.probe_grad_clip_norm=$PROBE_GRAD_CLIP_NORM \
    ++unsupervised_reward.probe_aux_loss_weight=$PROBE_AUX_LOSS_WEIGHT \
    ++unsupervised_reward.probe_train_steps=$PROBE_TRAIN_STEPS \
    ++unsupervised_reward.probe_class_balance_enable=$PROBE_CLASS_BALANCE_ENABLE \
    ++unsupervised_reward.probe_class_balance_power=$PROBE_CLASS_BALANCE_POWER \
    ++unsupervised_reward.probe_class_balance_min=$PROBE_CLASS_BALANCE_MIN \
    ++unsupervised_reward.probe_class_balance_max=$PROBE_CLASS_BALANCE_MAX \
    ++unsupervised_reward.probe_buffer_size=$PROBE_BUFFER_SIZE \
    ++unsupervised_reward.probe_buffer_batch_size=$PROBE_BUFFER_BATCH_SIZE \
    ++unsupervised_reward.probe_device=$PROBE_DEVICE \
    unsupervised_reward.soft_adv=$SOFT_ADV \
    custom_reward_function.path="$REPO_ROOT/verl/utils/reward_score/kk.py" \
    custom_reward_function.name=reward_func \
    trainer.val_before_train=False \
    trainer.log_val_generations=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.runner_num_gpus=$RUNNER_NUM_GPUS \
    trainer.runner_cuda_visible_devices="'$RUNNER_CUDA_VISIBLE_DEVICES'" \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=20 \
    trainer.total_epochs=3 \
    trainer.rollout_data_dir=$ROLLOUT_DIR \
    trainer.default_local_dir="$CKPT_PATH"/"$EXPERIMENT_NAME" \
    "$@" \
    2>&1 | tee -a "$LOG_FILE"
