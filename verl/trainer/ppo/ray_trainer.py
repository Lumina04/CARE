# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import re
import shutil
import tempfile
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, Type

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator,
    gamma=1.0,
    lam=1.0,
    num_repeat=1,
    multi_turn=False,
    norm_adv_by_std_in_grpo=True,
    config=None,
):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.get("pf_ppo_reweight_method", "pow"),
                config.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to "cuda".
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # Probe state for active-labeling estimator `grpo_adv_diff_probe_setscorer`
        self.adv_diff_probe = None
        self.adv_diff_probe_optimizer = None
        self.adv_diff_probe_input_dim = None
        self.adv_diff_probe_device = torch.device("cpu")
        self.adv_diff_probe_lr = None
        self.adv_diff_probe_weight_decay = None
        self.adv_diff_probe_arch_signature = None
        self.adv_diff_probe_set_cfg = None

        # Best-checkpoint tracking by validation metric
        self.best_val_metric_key: Optional[str] = None
        self.best_val_metric_value: float = float("-inf")
        self.best_val_step: int = -1

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    @staticmethod
    def _cfg_to_bool(value, default=False) -> bool:
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off"}:
                return False
        return bool(value)



    def _ensure_adv_diff_probe(
        self,
        input_dim: int,
        unsupervised_reward_cfg,
        group_size: int,
    ):
        from verl.trainer.ppo.ttrl_utils import (
            PromptFeatRespMLPBinaryClassifier,
            PromptFeatRespMLPBinaryStage2Classifier,
        )

        probe_device_cfg = str(unsupervised_reward_cfg.get("probe_device", "cpu"))
        if probe_device_cfg.lower() == "auto":
            probe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            try:
                probe_device = torch.device(probe_device_cfg)
            except Exception:
                probe_device = torch.device("cpu")

        lr = float(unsupervised_reward_cfg.get("probe_lr", 1e-3))
        weight_decay = float(unsupervised_reward_cfg.get("probe_weight_decay", 0.0))
        input_dim = int(input_dim)
        group_size = max(1, int(group_size))
        feat_input_dim = 1 + 2 * group_size
        resp_input_dim = input_dim
        prompt_hidden_dim = int(unsupervised_reward_cfg.get("probe_prompt_hidden_dim", 256))
        prompt_mid_dim = int(unsupervised_reward_cfg.get("probe_prompt_mid_dim", 256))
        resp_mlp_hidden_dim = int(unsupervised_reward_cfg.get("probe_resp_mlp_hidden_dim", 1024))
        resp_mlp_out_dim = int(unsupervised_reward_cfg.get("probe_resp_mlp_out_dim", 1024))
        activation_name = str(unsupervised_reward_cfg.get("probe_activation", "gelu"))

        set_cfg = {
            "prompt_hidden_dim": prompt_hidden_dim,
            "prompt_mid_dim": prompt_mid_dim,
            "resp_mlp_hidden_dim": resp_mlp_hidden_dim,
            "resp_mlp_out_dim": resp_mlp_out_dim,
            "activation_name": activation_name,
            "group_size": group_size,
            "feat_input_dim": feat_input_dim,
            "resp_input_dim": resp_input_dim,
            "cluster_count_input_dim": group_size,
        }
        arch_signature = (
            "prompt_len_cluster_dual_probe_count_stage2_no_len_no_aux_v1",
            input_dim,
            feat_input_dim,
            resp_input_dim,
            prompt_hidden_dim,
            prompt_mid_dim,
            resp_mlp_hidden_dim,
            resp_mlp_out_dim,
            activation_name,
            group_size,
        )

        need_reinit = (
            self.adv_diff_probe is None
            or not isinstance(self.adv_diff_probe, dict)
            or "stage1" not in self.adv_diff_probe
            or "stage2" not in self.adv_diff_probe
            or self.adv_diff_probe_input_dim != input_dim
            or self.adv_diff_probe_device != probe_device
            or self.adv_diff_probe_arch_signature != arch_signature
        )

        if need_reinit:
            stage1_model = PromptFeatRespMLPBinaryClassifier(
                hidden_input_dim=input_dim,
                feat_input_dim=feat_input_dim,
                resp_input_dim=resp_input_dim,
                cluster_count_input_dim=group_size,
                hidden_dim=prompt_hidden_dim,
                mid_dim=prompt_mid_dim,
                resp_mlp_hidden_dim=resp_mlp_hidden_dim,
                resp_mlp_out_dim=resp_mlp_out_dim,
                activation_name=activation_name,
            ).to(probe_device)

            stage2_model = PromptFeatRespMLPBinaryStage2Classifier(
                hidden_input_dim=input_dim,
                feat_input_dim=feat_input_dim,
                resp_input_dim=resp_input_dim,
                cluster_count_input_dim=group_size,
                hidden_dim=prompt_hidden_dim,
                mid_dim=prompt_mid_dim,
                resp_mlp_hidden_dim=resp_mlp_hidden_dim,
                resp_mlp_out_dim=resp_mlp_out_dim,
                activation_name=activation_name,
            ).to(probe_device)


            self.adv_diff_probe = {
                "stage1": stage1_model,
                "stage2": stage2_model,
            }
            self.adv_diff_probe_optimizer = {
                "stage1": torch.optim.AdamW(stage1_model.parameters(), lr=lr, weight_decay=weight_decay),
                "stage2": torch.optim.AdamW(stage2_model.parameters(), lr=lr, weight_decay=weight_decay),
            }

        self.adv_diff_probe_input_dim = input_dim
        self.adv_diff_probe_device = probe_device
        self.adv_diff_probe_set_cfg = dict(set_cfg)
        self.adv_diff_probe_lr = lr
        self.adv_diff_probe_weight_decay = weight_decay
        self.adv_diff_probe_arch_signature = arch_signature
        if need_reinit:
            print(
                "Initialized advantage-diff dual probes (offline-v2): "
                f"prompt_input_dim={input_dim}, feat_input_dim={feat_input_dim}, resp_input_dim={resp_input_dim}, "
                f"group_size={group_size}, prompt_hidden/mid=({prompt_hidden_dim},{prompt_mid_dim}), "
                f"resp_mlp=({resp_mlp_hidden_dim},{resp_mlp_out_dim}), activation={activation_name}, "
                f"device={probe_device}, lr={lr}, weight_decay={weight_decay}"
            )


    @staticmethod
    def _extract_consistent_probe_hidden_size(hidden_sizes) -> int:
        if isinstance(hidden_sizes, (list, tuple)):
            if len(hidden_sizes) == 0:
                raise RuntimeError("get_prompt_hidden_size returned an empty list")
            values = [int(v) for v in hidden_sizes]
        else:
            values = [int(hidden_sizes)]

        uniq = sorted(set(values))
        if len(uniq) != 1:
            raise RuntimeError(f"Inconsistent hidden_size across workers: {values}")
        return int(uniq[0])


    def _maybe_init_adv_diff_probe_once(self):
        unsupervised_reward = self.config.get("unsupervised_reward", None)
        if not unsupervised_reward or not unsupervised_reward.get("enable", False):
            return
        if unsupervised_reward.get("type", None) != "ensemble":
            return
        if unsupervised_reward.get("estimator", None) != "grpo_adv_diff_probe_setscorer":
            return

        if self.adv_diff_probe is not None:
            return

        hidden_sizes = self.actor_rollout_wg.get_prompt_hidden_size()
        probe_input_dim = self._extract_consistent_probe_hidden_size(hidden_sizes)
        group_size = int(self.config.actor_rollout_ref.rollout.n)
        self._ensure_adv_diff_probe(
            input_dim=probe_input_dim,
            unsupervised_reward_cfg=unsupervised_reward,
            group_size=group_size,
        )


    @staticmethod
    def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device):
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)


    def _save_adv_diff_probe_checkpoint(self, local_global_step_folder: str):
        if self.adv_diff_probe is None or self.adv_diff_probe_input_dim is None:
            return
        if not isinstance(self.adv_diff_probe, dict):
            return
        if "stage1" not in self.adv_diff_probe or "stage2" not in self.adv_diff_probe:
            return

        probe_local_path = os.path.join(local_global_step_folder, "adv_diff_probe.pt")
        probe_model = self.adv_diff_probe
        if not isinstance(probe_model, dict):
            return

        stage1_model = probe_model.get("stage1", None)
        stage2_model = probe_model.get("stage2", None)
        if stage1_model is None or stage2_model is None:
            return

        probe_state = {
            "probe_kind": "prompt_len_cluster_dual_probe_count_stage2_no_len_no_aux_v1",
            "arch_signature": self.adv_diff_probe_arch_signature,
            "input_dim": int(self.adv_diff_probe_input_dim),
            "set_cfg": self.adv_diff_probe_set_cfg,
            "lr": self.adv_diff_probe_lr,
            "weight_decay": self.adv_diff_probe_weight_decay,
            "device": str(self.adv_diff_probe_device),
            "stage1_model_state_dict": stage1_model.state_dict(),
            "stage2_model_state_dict": stage2_model.state_dict(),
        }

        if isinstance(self.adv_diff_probe_optimizer, dict):
            if self.adv_diff_probe_optimizer.get("stage1", None) is not None:
                probe_state["stage1_optimizer_state_dict"] = self.adv_diff_probe_optimizer["stage1"].state_dict()
            if self.adv_diff_probe_optimizer.get("stage2", None) is not None:
                probe_state["stage2_optimizer_state_dict"] = self.adv_diff_probe_optimizer["stage2"].state_dict()

        torch.save(probe_state, probe_local_path)
        print(f"Saved advantage-diff probe checkpoint to {probe_local_path}")


    def _load_adv_diff_probe_checkpoint(self, global_step_folder: str):
        probe_local_path = os.path.join(global_step_folder, "adv_diff_probe.pt")
        if not os.path.exists(probe_local_path):
            return

        probe_state = torch.load(probe_local_path, map_location="cpu", weights_only=False)
        if not isinstance(probe_state, dict):
            print(f"Warning: Invalid probe checkpoint format at {probe_local_path}, skip loading probe")
            return

        stage1_model_state_dict = probe_state.get("stage1_model_state_dict", None)
        stage2_model_state_dict = probe_state.get("stage2_model_state_dict", None)
        input_dim = probe_state.get("input_dim", None)
        if stage1_model_state_dict is None or stage2_model_state_dict is None or input_dim is None:
            print(f"Warning: Probe checkpoint missing keys at {probe_local_path}, skip loading probe")
            return

        probe_kind = str(probe_state.get("probe_kind", "")).strip()
        if probe_kind != "prompt_len_cluster_dual_probe_count_stage2_no_len_no_aux_v1":
            print(
                f"Warning: Probe checkpoint kind={probe_kind} at {probe_local_path} is incompatible with "
                "offline-v2 count-stage2 no-len/no-aux dual probes. Skip loading probe."
            )
            return

        saved_set_cfg = probe_state.get("set_cfg", None)
        if not isinstance(saved_set_cfg, dict):
            print(f"Warning: Probe checkpoint missing set_cfg at {probe_local_path}, skip loading probe")
            return

        unsupervised_reward_cfg = self.config.get("unsupervised_reward", {})
        probe_device_cfg = str(unsupervised_reward_cfg.get("probe_device", probe_state.get("device", "cpu")))
        if probe_device_cfg.lower() == "auto":
            probe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            try:
                probe_device = torch.device(probe_device_cfg)
            except Exception:
                probe_device = torch.device("cpu")

        lr = float(probe_state.get("lr", unsupervised_reward_cfg.get("probe_lr", 1e-3)))
        weight_decay = float(probe_state.get("weight_decay", unsupervised_reward_cfg.get("probe_weight_decay", 0.0)))

        from verl.trainer.ppo.ttrl_utils import (
            PromptFeatRespMLPBinaryClassifier,
            PromptFeatRespMLPBinaryStage2Classifier,
        )

        group_size = int(saved_set_cfg.get("group_size", unsupervised_reward_cfg.get("probe_group_size", 8)))
        group_size = max(1, group_size)
        expected_feat_input_dim = 1 + 2 * group_size
        feat_input_dim = int(saved_set_cfg.get("feat_input_dim", expected_feat_input_dim))
        if feat_input_dim != expected_feat_input_dim:
            print(
                f"Warning: Probe checkpoint at {probe_local_path} has feat_input_dim={feat_input_dim}, "
                f"but current offline-v2 representation expects {expected_feat_input_dim}. Skip loading probe."
            )
            return

        resp_input_dim = int(saved_set_cfg.get("resp_input_dim", int(input_dim)))
        prompt_hidden_dim = int(saved_set_cfg.get("prompt_hidden_dim", unsupervised_reward_cfg.get("probe_prompt_hidden_dim", 256)))
        prompt_mid_dim = int(saved_set_cfg.get("prompt_mid_dim", unsupervised_reward_cfg.get("probe_prompt_mid_dim", 256)))
        resp_mlp_hidden_dim = int(saved_set_cfg.get("resp_mlp_hidden_dim", unsupervised_reward_cfg.get("probe_resp_mlp_hidden_dim", 1024)))
        resp_mlp_out_dim = int(saved_set_cfg.get("resp_mlp_out_dim", unsupervised_reward_cfg.get("probe_resp_mlp_out_dim", 1024)))
        activation_name = str(saved_set_cfg.get("activation_name", unsupervised_reward_cfg.get("probe_activation", "gelu")))

        stage1_base = PromptFeatRespMLPBinaryClassifier(
            hidden_input_dim=int(input_dim),
            feat_input_dim=feat_input_dim,
            resp_input_dim=resp_input_dim,
            cluster_count_input_dim=group_size,
            hidden_dim=prompt_hidden_dim,
            mid_dim=prompt_mid_dim,
            resp_mlp_hidden_dim=resp_mlp_hidden_dim,
            resp_mlp_out_dim=resp_mlp_out_dim,
            activation_name=activation_name,
        ).to(probe_device)

        stage2_base = PromptFeatRespMLPBinaryStage2Classifier(
            hidden_input_dim=int(input_dim),
            feat_input_dim=feat_input_dim,
            resp_input_dim=resp_input_dim,
            cluster_count_input_dim=group_size,
            hidden_dim=prompt_hidden_dim,
            mid_dim=prompt_mid_dim,
            resp_mlp_hidden_dim=resp_mlp_hidden_dim,
            resp_mlp_out_dim=resp_mlp_out_dim,
            activation_name=activation_name,
        ).to(probe_device)

        try:
            stage1_base.load_state_dict(stage1_model_state_dict)
            stage2_base.load_state_dict(stage2_model_state_dict)
        except Exception as e:
            print(f"Warning: Failed to load dual probe models from {probe_local_path}: {e}")
            return

        stage1_probe = stage1_base
        stage2_probe = stage2_base

        stage1_optimizer = torch.optim.AdamW(stage1_probe.parameters(), lr=lr, weight_decay=weight_decay)
        stage2_optimizer = torch.optim.AdamW(stage2_probe.parameters(), lr=lr, weight_decay=weight_decay)

        stage1_opt_state = probe_state.get("stage1_optimizer_state_dict", None)
        stage2_opt_state = probe_state.get("stage2_optimizer_state_dict", None)
        if stage1_opt_state is not None:
            try:
                stage1_optimizer.load_state_dict(stage1_opt_state)
                self._move_optimizer_state_to_device(stage1_optimizer, probe_device)
            except Exception as e:
                print(
                    f"Warning: Failed to load stage1 probe optimizer from {probe_local_path}: {e}. "
                    "Using re-initialized stage1 optimizer"
                )
        if stage2_opt_state is not None:
            try:
                stage2_optimizer.load_state_dict(stage2_opt_state)
                self._move_optimizer_state_to_device(stage2_optimizer, probe_device)
            except Exception as e:
                print(
                    f"Warning: Failed to load stage2 probe optimizer from {probe_local_path}: {e}. "
                    "Using re-initialized stage2 optimizer"
                )

        self.adv_diff_probe = {
            "stage1": stage1_probe,
            "stage2": stage2_probe,
        }
        self.adv_diff_probe_optimizer = {
            "stage1": stage1_optimizer,
            "stage2": stage2_optimizer,
        }
        self.adv_diff_probe_input_dim = int(input_dim)
        self.adv_diff_probe_device = probe_device
        self.adv_diff_probe_set_cfg = {
            "prompt_hidden_dim": prompt_hidden_dim,
            "prompt_mid_dim": prompt_mid_dim,
            "resp_mlp_hidden_dim": resp_mlp_hidden_dim,
            "resp_mlp_out_dim": resp_mlp_out_dim,
            "activation_name": activation_name,
            "group_size": group_size,
            "feat_input_dim": feat_input_dim,
            "resp_input_dim": resp_input_dim,
            "cluster_count_input_dim": group_size,
        }
        self.adv_diff_probe_lr = lr
        self.adv_diff_probe_weight_decay = weight_decay
        self.adv_diff_probe_arch_signature = (
            "prompt_len_cluster_dual_probe_count_stage2_no_len_no_aux_v1",
            int(input_dim),
            feat_input_dim,
            resp_input_dim,
            prompt_hidden_dim,
            prompt_mid_dim,
            resp_mlp_hidden_dim,
            resp_mlp_out_dim,
            activation_name,
            group_size,
        )
        print(f"Loaded advantage-diff probe from {probe_local_path}")

    def _default_local_dir_abs(self) -> str:
        default_local_dir = self.config.trainer.default_local_dir
        if not os.path.isabs(default_local_dir):
            default_local_dir = os.path.join(os.getcwd(), default_local_dir)
        return default_local_dir

    def _get_val_dataset_rollout_n_overrides(self) -> dict[str, int]:
        cfg = OmegaConf.select(self.config, "trainer.val_dataset_rollout_n")
        if cfg is None:
            return {}

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg_dict, dict):
            return {}

        overrides = {}
        for pattern, value in cfg_dict.items():
            try:
                n = int(value)
            except Exception:
                continue
            if n < 1:
                continue
            overrides[str(pattern)] = n
        return overrides

    @staticmethod
    def _resolve_val_rollout_n_for_source(data_source, default_n: int, overrides: dict[str, int]) -> int:
        src = str(data_source)
        if overrides:
            for pattern, n in sorted(overrides.items(), key=lambda x: len(x[0]), reverse=True):
                if pattern and pattern in src:
                    return max(int(n), 1)
        return max(int(default_n), 1)

    @staticmethod
    def _select_best_checkpoint_metric(val_metrics: dict, preferred_key: Optional[str] = None):
        if not isinstance(val_metrics, dict) or len(val_metrics) == 0:
            return None, None

        if preferred_key and preferred_key in val_metrics:
            try:
                return preferred_key, float(val_metrics[preferred_key])
            except Exception:
                pass

        # Prefer unified all-datasets core metric (dataset-level acc means averaged together),
        # which is robust when different datasets use different rollout n.
        unified_key = "val-core/all-datasets/acc/mean"
        if unified_key in val_metrics:
            try:
                return unified_key, float(val_metrics[unified_key])
            except Exception:
                pass

        def _pick(candidates):
            if not candidates:
                return None, None
            max_n = max(n for n, _ in candidates)
            keys = sorted([k for n, k in candidates if n == max_n])
            for key in keys:
                try:
                    return key, float(val_metrics[key])
                except Exception:
                    continue
            return None, None

        candidates_all = []
        candidates_any = []
        for key in val_metrics.keys():
            key_str = str(key)
            match = re.search(r"/acc/mean@(\d+)$", key_str)
            if match is None:
                continue
            n = int(match.group(1))
            if key_str.startswith("val-core/all-datasets/acc/mean@"):
                candidates_all.append((n, key_str))
            if key_str.startswith("val-core/"):
                candidates_any.append((n, key_str))

        key, value = _pick(candidates_all)
        if key is not None:
            return key, value

        key, value = _pick(candidates_any)
        if key is not None:
            return key, value

        return None, None

    def _save_best_checkpoint_state(self):
        if self.best_val_metric_key is None:
            return

        default_local_dir = self._default_local_dir_abs()
        os.makedirs(default_local_dir, exist_ok=True)
        best_checkpoint_path = self._best_checkpoint_dir_abs()

        state = {
            "best_metric_key": self.best_val_metric_key,
            "best_metric_value": float(self.best_val_metric_value),
            "best_global_step": int(self.best_val_step),
            "best_checkpoint_path": best_checkpoint_path,
        }

        state_path = os.path.join(default_local_dir, "best_checkpoint.json")
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
        print(f"Updated best checkpoint state at {state_path}: {state}")

    def _best_checkpoint_dir_abs(self) -> str:
        default_local_dir = self._default_local_dir_abs()
        best_dirname = str(self.config.trainer.get("best_checkpoint_dirname", "best_checkpoint"))
        return os.path.join(default_local_dir, best_dirname)

    @staticmethod
    def _remove_optimizer_states_in_dir(checkpoint_dir: str):
        """Delete optimizer shard files under a checkpoint directory."""
        if not os.path.isdir(checkpoint_dir):
            return

        optim_file_pattern = re.compile(r"^optim_world_size_\d+_rank_\d+\.pt$")
        removed_count = 0
        for root, _, files in os.walk(checkpoint_dir):
            for filename in files:
                if not optim_file_pattern.match(filename):
                    continue
                file_path = os.path.join(root, filename)
                try:
                    os.remove(file_path)
                    removed_count += 1
                except OSError as e:
                    print(f"Warning: failed to remove optimizer shard in best checkpoint {file_path}: {e}")

        if removed_count > 0:
            print(f"Removed {removed_count} optimizer shard files from best checkpoint folder {checkpoint_dir}.")

    def _save_best_checkpoint(self):
        """Save current model state into a stable best-checkpoint directory (overwritten on new best)."""
        from verl.utils.fs import local_mkdir_safe

        best_local_folder = self._best_checkpoint_dir_abs()
        if os.path.lexists(best_local_folder):
            if os.path.islink(best_local_folder) or os.path.isfile(best_local_folder):
                os.remove(best_local_folder)
            else:
                shutil.rmtree(best_local_folder)

        print(f"best_local_folder: {best_local_folder}")
        actor_local_path = os.path.join(best_local_folder, "actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, os.path.basename(best_local_folder), "actor")
        )
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=None)

        if self.use_critic:
            critic_local_path = os.path.join(best_local_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, os.path.basename(best_local_folder), "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=None
            )

        local_mkdir_safe(best_local_folder)
        dataloader_local_path = os.path.join(best_local_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        self._save_adv_diff_probe_checkpoint(best_local_folder)
        self._remove_optimizer_states_in_dir(best_local_folder)

    def _load_best_checkpoint_state(self):
        default_local_dir = self._default_local_dir_abs()
        state_path = os.path.join(default_local_dir, "best_checkpoint.json")
        if not os.path.exists(state_path):
            return

        try:
            with open(state_path, "r") as f:
                state = json.load(f)
            self.best_val_metric_key = state.get("best_metric_key", None)
            self.best_val_metric_value = float(state.get("best_metric_value", float("-inf")))
            self.best_val_step = int(state.get("best_global_step", -1))
            print(
                f"Loaded best checkpoint state from {state_path}: "
                f"metric={self.best_val_metric_key}, value={self.best_val_metric_value}, step={self.best_val_step}"
            )
        except Exception as e:
            print(f"Warning: Failed to load best checkpoint state from {state_path}: {e}")

    def _cleanup_old_optimizer_states(self, keep_step: int):
        """Delete optimizer shard files from old checkpoints, keep latest checkpoint resumable."""
        if not self.config.trainer.get("cleanup_old_optimizer_states", True):
            return

        default_local_dir = self._default_local_dir_abs()
        if not os.path.isdir(default_local_dir):
            return

        step_dir_pattern = re.compile(r"^global_step_(\d+)$")
        optim_file_pattern = re.compile(r"^optim_world_size_\d+_rank_\d+\.pt$")

        removed_count = 0
        scanned_step_dirs = 0

        for step_dir_name in os.listdir(default_local_dir):
            step_dir_match = step_dir_pattern.match(step_dir_name)
            if step_dir_match is None:
                continue

            step = int(step_dir_match.group(1))
            if step == int(keep_step):
                continue

            step_dir_path = os.path.join(default_local_dir, step_dir_name)
            if not os.path.isdir(step_dir_path):
                continue
            scanned_step_dirs += 1

            for model_subdir in ("actor", "critic"):
                model_dir = os.path.join(step_dir_path, model_subdir)
                if not os.path.isdir(model_dir):
                    continue

                for filename in os.listdir(model_dir):
                    if not optim_file_pattern.match(filename):
                        continue
                    file_path = os.path.join(model_dir, filename)
                    try:
                        os.remove(file_path)
                        removed_count += 1
                    except FileNotFoundError:
                        continue
                    except Exception as e:
                        print(f"Warning: failed to remove old optimizer shard {file_path}: {e}")

        if removed_count > 0:
            print(
                f"Removed {removed_count} optimizer shard files from {scanned_step_dirs} old checkpoint step dirs "
                f"(kept full optimizer state for global_step_{keep_step})."
            )

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert (
                config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None
                or config.actor_rollout_ref.rollout.multi_turn.interaction_config_path is not None
            ), (
                "tool_config_path or interaction_config_path must be set when enabling multi_turn with tool, "
                "due to no role-playing support"
            )
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], (
                "only GRPO is tested for multi-turn with tool"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # Repeat validation prompts with per-dataset rollout counts.
            prompt_count = int(test_batch.batch["input_ids"].shape[0])
            prompt_data_sources = np.asarray(
                test_batch.non_tensor_batch.get("data_source", ["unknown"] * prompt_count), dtype=object
            ).reshape(-1)
            if prompt_data_sources.size != prompt_count:
                prompt_data_sources = np.array(["unknown"] * prompt_count, dtype=object)

            default_val_rollout_n = max(int(self.config.actor_rollout_ref.rollout.val_kwargs.n), 1)
            val_rollout_overrides = self._get_val_dataset_rollout_n_overrides()
            prompt_val_rollout_n = np.array(
                [
                    self._resolve_val_rollout_n_for_source(src, default_val_rollout_n, val_rollout_overrides)
                    for src in prompt_data_sources
                ],
                dtype=int,
            )
            repeat_times = int(prompt_val_rollout_n.max()) if prompt_val_rollout_n.size > 0 else default_val_rollout_n
            repeat_times = max(repeat_times, 1)

            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)

            # Keep only the first n responses for prompts that need fewer validation rollouts.
            if prompt_val_rollout_n.size > 0 and np.any(prompt_val_rollout_n != repeat_times):
                selected_indices = []
                for prompt_idx, prompt_n in enumerate(prompt_val_rollout_n.tolist()):
                    start = prompt_idx * repeat_times
                    selected_indices.extend(range(start, start + int(prompt_n)))
                test_batch = test_batch[selected_indices]

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        # Aggregate per-dataset core accuracy means into one unified metric:
        # val-core/all-datasets/acc/mean
        # This works even when datasets use different rollout n (e.g., math500@1, others@8).
        dataset_acc_mean_vals: dict[str, list[float]] = defaultdict(list)
        for metric_name, metric_val in metric_dict.items():
            if ("/acc/mean@" not in metric_name) or (not metric_name.startswith("val-core/")):
                continue
            match = re.match(r"^val-core/(.+)/acc/mean@\d+$", metric_name)
            if match is None:
                continue
            dataset_name = match.group(1)
            if dataset_name == "all-datasets":
                continue
            dataset_acc_mean_vals[dataset_name].append(float(metric_val))

        if len(dataset_acc_mean_vals) > 0:
            per_dataset_means = [float(np.mean(vals)) for vals in dataset_acc_mean_vals.values()]
            metric_dict["val-core/all-datasets/acc/mean"] = float(np.mean(per_dataset_means))

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()
        self._maybe_init_adv_diff_probe_once()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        self._save_adv_diff_probe_checkpoint(local_global_step_folder)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

        self._cleanup_old_optimizer_states(keep_step=self.global_steps)

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        self._load_adv_diff_probe_checkpoint(global_step_folder)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        self._load_best_checkpoint_state()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        repeat_sampling_sglang_grpo = (
            self.config.actor_rollout_ref.rollout.name == "sglang"
            and self.config.actor_rollout_ref.rollout.multi_turn.enable
        )

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {}
                timing_raw = {}
                pending_probe_train_batch = None

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]

                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                if repeat_sampling_sglang_grpo:
                    uids_for_prompts = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    batch.non_tensor_batch["uid"] = uids_for_prompts
                    gen_batch.non_tensor_batch["uid"] = uids_for_prompts
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    assert np.array_equal(batch.non_tensor_batch["uid"], gen_batch.non_tensor_batch["uid"]), (
                        "UIDs must be identical for SGLang rollout"
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    unsupervised_reward = self.config.get("unsupervised_reward", None)
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if self.config.get("ttrl", {}).get("enable", False):
                            from verl.trainer.ppo.ttrl_utils import select_top_k_per_prompt, apply_ttrl_gt

                            gen_batch.meta_info["kwargs"] = {"n": self.config.ttrl.n_votes_per_prompt}
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                            assert len(gen_batch_output) == len(batch) * self.config.ttrl.n_votes_per_prompt

                            batch = apply_ttrl_gt(batch, gen_batch_output, self.config.ttrl.n_votes_per_prompt, self.tokenizer)
                            gen_batch_output = select_top_k_per_prompt(gen_batch_output, self.config.ttrl.n_votes_per_prompt, self.config.ttrl.n_samples_per_prompt)

                            assert len(gen_batch_output) == len(batch) * self.config.ttrl.n_samples_per_prompt
                        elif unsupervised_reward and unsupervised_reward.get("enable", False) and unsupervised_reward.get("type", None) == "ensemble":
                            from verl.trainer.ppo.ttrl_utils import apply_ttrl_gt as apply_majority_voting_gt
                            from verl.trainer.ppo.ttrl_utils import apply_random_ranked_gt
                            from verl.trainer.ppo.ttrl_utils import apply_entropy_ranked_gt
                            from verl.trainer.ppo.ttrl_utils import apply_avg_prob_ranked_gt
                            from verl.trainer.ppo.ttrl_utils import apply_grpo_adv_diff_top_gt
                            from verl.trainer.ppo.ttrl_utils import apply_grpo_adv_diff_probe_setscorer_gt
                            from verl.trainer.ppo.ttrl_utils import train_grpo_adv_diff_probe_setscorer

                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                            n_samples_per_prompt = self.config.actor_rollout_ref.rollout.n
                            assert len(gen_batch_output) == len(batch) * n_samples_per_prompt

                            ensemble_estimator = unsupervised_reward.get("estimator", "majority_voting")
                            if ensemble_estimator == "majority_voting":
                                batch = apply_majority_voting_gt(
                                    batch,
                                    gen_batch_output,
                                    n=n_samples_per_prompt,
                                    tokenizer=self.tokenizer,
                                )
                            elif ensemble_estimator == "random_gt_zero_adv":
                                random_gt_top_pct = float(unsupervised_reward.get("random_gt_top_pct", 0.1))
                                batch = apply_random_ranked_gt(
                                    batch,
                                    gen_batch_output,
                                    n=n_samples_per_prompt,
                                    tokenizer=self.tokenizer,
                                    top_pct=random_gt_top_pct
                                )
                            elif ensemble_estimator == "entropy_gt_zero_adv":
                                entropy_gt_top_pct = float(
                                    unsupervised_reward.get(
                                        "entropy_gt_top_pct",
                                        unsupervised_reward.get("random_gt_top_pct", 0.2),
                                    )
                                )
                                batch = apply_entropy_ranked_gt(
                                    batch,
                                    gen_batch_output,
                                    n=n_samples_per_prompt,
                                    tokenizer=self.tokenizer,
                                    top_pct=entropy_gt_top_pct,
                                )
                            elif ensemble_estimator == "avg_prob_gt_zero_adv":
                                avg_prob_gt_top_pct = float(
                                    unsupervised_reward.get(
                                        "avg_prob_gt_top_pct",
                                        unsupervised_reward.get("random_gt_top_pct", 0.2),
                                    )
                                )
                                with marked_timer("avg_prob_rank", timing_raw, color="blue"):
                                    gen_batch_output.meta_info["calculate_entropy"] = False
                                    avg_prob_log_prob = self.actor_rollout_wg.compute_log_prob(gen_batch_output)
                                    gen_batch_output.meta_info.pop("calculate_entropy", None)
                                    avg_prob_response_mask = compute_response_mask(gen_batch_output)
                                    avg_prob_old_log_probs = avg_prob_log_prob.batch["old_log_probs"]
                                    avg_prob_token_probs = torch.exp(avg_prob_old_log_probs) * avg_prob_response_mask
                                    avg_prob_lengths = avg_prob_response_mask.sum(dim=-1).clamp(min=1)
                                    response_avg_probs = (
                                        avg_prob_token_probs.sum(dim=-1) / avg_prob_lengths
                                    ).detach().cpu().numpy()
                                    batch = apply_avg_prob_ranked_gt(
                                        batch,
                                        gen_batch_output,
                                        n=n_samples_per_prompt,
                                        tokenizer=self.tokenizer,
                                        response_avg_probs=response_avg_probs,
                                        top_pct=avg_prob_gt_top_pct,
                                    )
                                    metrics["train/ensemble/avg_probability_mean"] = float(np.mean(response_avg_probs))
                                    metrics["train/ensemble/avg_probability_min"] = float(np.min(response_avg_probs))
                                    metrics["train/ensemble/avg_probability_max"] = float(np.max(response_avg_probs))
                            elif ensemble_estimator == "grpo_adv_diff_probe_setscorer":
                                probe_gt_top_pct = float(
                                    unsupervised_reward.get("probe_gt_top_pct", 0.2)
                                )
                                probe_gt_top_pct = max(0.0, min(1.0, probe_gt_top_pct))
                                probe_prob_gate = float(
                                    unsupervised_reward.get("probe_prob_gate", 0.1)
                                )
                                probe_prob_gate = max(0.0, min(1.0, probe_prob_gate))
                                probe_feature_proto = self.actor_rollout_wg.compute_probe_stage1_features(gen_batch_output)
                                probe_prompt_hidden = probe_feature_proto.batch["probe_prompt_hidden"]
                                probe_resp_hidden = probe_feature_proto.batch["probe_resp_hidden"]

                                if (
                                    self.adv_diff_probe is None
                                    or self.adv_diff_probe_optimizer is None
                                    or self.adv_diff_probe_device is None
                                ):
                                    raise RuntimeError(
                                        "Adv-diff probe is not initialized. "
                                        "Expected initialization during init_workers."
                                    )

                                batch, probe_metrics, probe_train_batch = apply_grpo_adv_diff_probe_setscorer_gt(
                                    batch,
                                    gen_batch_output,
                                    n=n_samples_per_prompt,
                                    tokenizer=self.tokenizer,
                                    prompt_hidden=probe_prompt_hidden,
                                    resp_last_hidden=probe_resp_hidden,
                                    probe=self.adv_diff_probe,
                                    top_pct=probe_gt_top_pct,
                                    probe_device=str(self.adv_diff_probe_device),
                                    global_step=int(self.global_steps),
                                    prob_gate=probe_prob_gate,
                                )

                                pending_probe_train_batch = probe_train_batch

                                for key, value in probe_metrics.items():
                                    metrics[f"train/ensemble/{key}"] = value
                            elif ensemble_estimator == "grpo_adv_diff_top_gt":
                                grpo_adv_top_pct = float(
                                    unsupervised_reward.get(
                                        "grpo_adv_top_pct",
                                        unsupervised_reward.get("random_gt_top_pct", 0.1),
                                    )
                                )
                                batch = apply_grpo_adv_diff_top_gt(
                                    batch,
                                    gen_batch_output,
                                    n=n_samples_per_prompt,
                                    tokenizer=self.tokenizer,
                                    top_pct=grpo_adv_top_pct,
                                )
                            else:
                                raise ValueError(
                                    f"Unknown ensemble estimator: {ensemble_estimator}. "
                                    "Supported estimators are 'majority_voting', "
                                    "'random_gt_zero_adv', 'entropy_gt_zero_adv', 'avg_prob_gt_zero_adv', 'grpo_adv_diff_probe_setscorer', and 'grpo_adv_diff_top_gt'."
                                )

                            assert len(gen_batch_output) == len(batch) * n_samples_per_prompt
                        else:
                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            else:
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    if not repeat_sampling_sglang_grpo:
                        batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                        )
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch:
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs (moved before reward to make certainty metrics available)
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        # Enable self_certainty calculation when certainty-based unsupervised reward is used
                        need_self_certainty = (
                            unsupervised_reward and unsupervised_reward.get("enable", False)
                            and unsupervised_reward.get("type", None) == "certainty"
                            and unsupervised_reward.get("estimator", None) == "self_certainty"
                        )
                        if need_self_certainty:
                            batch.meta_info["calculate_self_certainty"] = True
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        # Keep entropys in batch for certainty-based methods
                        if not need_self_certainty:
                            old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # Compute certainty-based/self-verify pseudo reward (before rule-based reward)
                        pseudo_reward_tensor = None
                        pseudo_reward_extra_infos_dict = None
                        if unsupervised_reward and unsupervised_reward.get("enable", False):
                            if unsupervised_reward.get("type", None) == "certainty":
                                if "response_mask" not in batch.batch.keys():
                                    batch.batch["response_mask"] = compute_response_mask(batch)
                                from verl.trainer.ppo.ttrl_utils import compute_certainty_reward
                                pseudo_reward_tensor, pseudo_reward_extra_infos_dict = compute_certainty_reward(
                                    batch, unsupervised_reward.get("estimator", None)
                                )

                            # Compute self-verify reward
                            if unsupervised_reward.get("type", None) == "external":
                                if unsupervised_reward.get("estimator", None) == "self_verify":
                                    from verl.trainer.ppo.ttrl_utils import apply_self_verify
                                    pseudo_reward_tensor, pseudo_reward_extra_infos_dict = apply_self_verify(
                                        batch, self.tokenizer, self.actor_rollout_wg, verify_prompt=None
                                    )

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)

                        batch.batch["token_level_scores"] = reward_tensor

                        # Override token_level_scores with pseudo reward for certainty-based/self-verify methods
                        if unsupervised_reward and unsupervised_reward.get("enable", False):
                            if unsupervised_reward.get("type", None) == "certainty" and pseudo_reward_tensor is not None:
                                batch.batch["token_level_scores"] = pseudo_reward_tensor
                                if pseudo_reward_extra_infos_dict:
                                    batch.non_tensor_batch.update({k: np.array(v) for k, v in pseudo_reward_extra_infos_dict.items()})
                            if unsupervised_reward.get("type", None) == "external" and unsupervised_reward.get("estimator", None) == "self_verify" and pseudo_reward_tensor is not None:
                                batch.batch["token_level_scores"] = pseudo_reward_tensor
                                if pseudo_reward_extra_infos_dict:
                                    batch.non_tensor_batch.update({k: np.array(v) for k, v in pseudo_reward_extra_infos_dict.items()})

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                        )

                        if (
                            unsupervised_reward
                            and unsupervised_reward.get("enable", False)
                            and unsupervised_reward.get("type", None) == "ensemble"
                            and unsupervised_reward.get("estimator", None) == "majority_voting"
                            and unsupervised_reward.get("zero_wrong_majority_adv", False)
                        ):
                            majority_vote_correct_mask = batch.non_tensor_batch.get("majority_vote_correct_mask", None)
                            if majority_vote_correct_mask is None:
                                raise ValueError(
                                    "majority_vote_correct_mask is required when "
                                    "unsupervised_reward.zero_wrong_majority_adv=True"
                                )

                            correct_mask_np = np.asarray(majority_vote_correct_mask, dtype=bool).reshape(-1)
                            adv_batch_size = int(batch.batch["advantages"].shape[0])

                            if correct_mask_np.size != adv_batch_size:
                                raise ValueError(
                                    "Expected sample-level majority_vote_correct_mask aligned with current batch order, "
                                    f"but got mask_size={correct_mask_np.size}, advantage_batch={adv_batch_size}."
                                )

                            correct_mask = torch.as_tensor(
                                correct_mask_np,
                                dtype=batch.batch["advantages"].dtype,
                                device=batch.batch["advantages"].device,
                            ).unsqueeze(-1)
                        
                            batch.batch["advantages"] = batch.batch["advantages"] * correct_mask
                            if "returns" in batch.batch:
                                batch.batch["returns"] = batch.batch["returns"] * correct_mask

                            metrics["train/ensemble/majority_vote_correct_ratio_effective"] = float(correct_mask_np.mean())
                            metrics["train/ensemble/majority_vote_zero_adv_ratio_effective"] = float((~correct_mask_np).mean())

                        if (
                            unsupervised_reward
                            and unsupervised_reward.get("enable", False)
                            and unsupervised_reward.get("type", None) == "ensemble"
                            and unsupervised_reward.get("estimator", None) in {"grpo_adv_diff_top_gt", "grpo_adv_diff_probe_setscorer"}
                        ):
                            soft_adv = float(unsupervised_reward.get("soft_adv", 0.0))
                            if soft_adv > 0:
                                use_ground_truth_mask = batch.non_tensor_batch.get("use_ground_truth_mask", None)
                                adv_diff_norm_list = batch.non_tensor_batch.get("adv_diff_norm_list", None)
                                if use_ground_truth_mask is None:
                                    raise ValueError(
                                        "use_ground_truth_mask is required when unsupervised_reward.soft_adv > 0 "
                                        "for grpo_adv_diff estimators"
                                    )
                                if adv_diff_norm_list is None:
                                    raise ValueError(
                                        "adv_diff_norm_list is required when unsupervised_reward.soft_adv > 0 "
                                        "for grpo_adv_diff estimators"
                                    )

                                gt_mask_np = np.asarray(use_ground_truth_mask, dtype=bool).reshape(-1)
                                adv_diff_np = np.asarray(adv_diff_norm_list, dtype=float).reshape(-1)
                                adv_diff_np = np.nan_to_num(adv_diff_np, nan=0.0, posinf=0.0, neginf=0.0)
                                adv_batch_size = int(batch.batch["advantages"].shape[0])

                                # Important for correctness with batch balancing:
                                # both arrays must already be sample-level and aligned with current batch order.
                                if adv_diff_np.size != adv_batch_size or gt_mask_np.size != adv_batch_size:
                                    raise ValueError(
                                        "Expected sample-level non-tensor arrays after repeat/reorder, but got "
                                        f"adv_diff_size={adv_diff_np.size}, mask_size={gt_mask_np.size}, "
                                        f"advantage_batch={adv_batch_size}."
                                    )

                                pseudo_mask_np = ~gt_mask_np
                                soft_coef_prompt = np.ones_like(adv_diff_np, dtype=float)
                                metrics["train/ensemble/soft_adv_applied_ratio"] = float(np.mean(pseudo_mask_np.astype(float)))

                                # Exponential reliability decay:
                                # coef = exp(-soft_adv * |A|)
                                # soft_coef_prompt[pseudo_mask_np] = np.exp(-soft_adv * adv_diff_np[pseudo_mask_np])

                                soft_coef_prompt[pseudo_mask_np] = soft_adv * adv_diff_np[pseudo_mask_np]
                                soft_coef_prompt = np.clip(soft_coef_prompt, 0.0, 1.0)

                                if soft_coef_prompt.size != adv_batch_size:
                                    raise ValueError(
                                        "Shape mismatch for soft_adv coefficients: "
                                        f"coef_size={soft_coef_prompt.size}, advantage_batch={adv_batch_size}"
                                    )
                                soft_coef_np = soft_coef_prompt

                                soft_coef = torch.as_tensor(
                                    soft_coef_np,
                                    dtype=batch.batch["advantages"].dtype,
                                    device=batch.batch["advantages"].device,
                                ).unsqueeze(-1)

                                # oracle-mask
                                if unsupervised_reward.get("estimator", None) in {"grpo_adv_diff_top_gt"}:
                                    soft_coef[pseudo_mask_np] = soft_coef[pseudo_mask_np] * 0

                                batch.batch["advantages"] = batch.batch["advantages"] * soft_coef
                                if "returns" in batch.batch:
                                    batch.batch["returns"] = batch.batch["returns"] * soft_coef

                        if (
                            unsupervised_reward
                            and unsupervised_reward.get("enable", False)
                            and unsupervised_reward.get("type", None) == "ensemble"
                            and (
                                (
                                    unsupervised_reward.get("estimator", None) == "random_gt_zero_adv"
                                    and self._cfg_to_bool(
                                        unsupervised_reward.get("random_gt_mask_non_gt", True),
                                        default=True,
                                    )
                                )
                                or (
                                    unsupervised_reward.get("estimator", None) == "entropy_gt_zero_adv"
                                    and self._cfg_to_bool(
                                        unsupervised_reward.get("entropy_gt_mask_non_gt", True),
                                        default=True,
                                    )
                                )
                                or (
                                    unsupervised_reward.get("estimator", None) == "avg_prob_gt_zero_adv"
                                    and self._cfg_to_bool(
                                        unsupervised_reward.get("avg_prob_gt_mask_non_gt", True),
                                        default=True,
                                    )
                                )
                            )
                        ):
                            use_ground_truth_mask = batch.non_tensor_batch.get("use_ground_truth_mask", None)
                            if use_ground_truth_mask is None:
                                raise ValueError(
                                    "use_ground_truth_mask is required for zero-adv ensemble estimators"
                                )

                            gt_mask_np = np.asarray(use_ground_truth_mask, dtype=bool).reshape(-1)
                            adv_batch_size = int(batch.batch["advantages"].shape[0])

                            if gt_mask_np.size != adv_batch_size:
                                raise ValueError(
                                    "Expected sample-level use_ground_truth_mask aligned with current batch order, "
                                    f"but got mask_size={gt_mask_np.size}, advantage_batch={adv_batch_size}."
                                )

                            gt_mask = torch.as_tensor(
                                gt_mask_np,
                                dtype=batch.batch["advantages"].dtype,
                                device=batch.batch["advantages"].device,
                            ).unsqueeze(-1)

                            batch.batch["advantages"] = batch.batch["advantages"] * gt_mask
                            if "returns" in batch.batch:
                                batch.batch["returns"] = batch.batch["returns"] * gt_mask

                            metrics["train/ensemble/use_ground_truth_ratio_effective"] = float(gt_mask_np.mean())

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                        if pending_probe_train_batch is not None:
                            with marked_timer("update_probe", timing_raw, color="cyan"):
                                probe_train_metrics = train_grpo_adv_diff_probe_setscorer(
                                    probe=self.adv_diff_probe,
                                    probe_optimizer=self.adv_diff_probe_optimizer,
                                    probe_train_batch=pending_probe_train_batch,
                                    probe_device=str(self.adv_diff_probe_device),
                                    train_steps=int(unsupervised_reward.get("probe_train_steps", 1)) if unsupervised_reward else 1,
                                    probe_grad_clip_norm=float(unsupervised_reward.get("probe_grad_clip_norm", 1.0)) if unsupervised_reward else 1.0,
                                    probe_buffer_size=int(unsupervised_reward.get("probe_buffer_size", 256)) if unsupervised_reward else 256,
                                    probe_buffer_batch_size=int(unsupervised_reward.get("probe_buffer_batch_size", 16)) if unsupervised_reward else 16,
                                    lr=float(unsupervised_reward.get("probe_lr", 1e-3))
                                    if unsupervised_reward
                                    else 1e-3,
                                    aux_loss_weight=float(unsupervised_reward.get("probe_aux_loss_weight", 1.0))
                                    if unsupervised_reward
                                    else 1.0,
                                    class_balance_enable=self._cfg_to_bool(unsupervised_reward.get("probe_class_balance_enable", True), default=True)
                                    if unsupervised_reward
                                    else True,
                                    class_balance_power=float(unsupervised_reward.get("probe_class_balance_power", 0.5))
                                    if unsupervised_reward
                                    else 0.5,
                                    class_balance_min=float(unsupervised_reward.get("probe_class_balance_min", 0.25))
                                    if unsupervised_reward
                                    else 0.25,
                                    class_balance_max=float(unsupervised_reward.get("probe_class_balance_max", 4.0))
                                    if unsupervised_reward
                                    else 4.0,
                                )
                            for key, value in probe_train_metrics.items():
                                metrics[f"train/ensemble/{key}"] = value
                            pending_probe_train_batch = None

                    if self.config.get("ttrl", {}).get("enable", False):
                        from verl.trainer.ppo.ttrl_utils import apply_original_gt, compute_ttrl_metrics
                        batch = apply_original_gt(batch)
                        reward_tensor_original, reward_extra_infos_dict_original = compute_reward(batch, self.reward_fn)
                        batch.batch["token_level_scores_original"] = reward_tensor_original
                        # Compute ttrl metrics
                        ttrl_metrics = compute_ttrl_metrics(batch, self.config.ttrl.n_samples_per_prompt)
                        for key, value in ttrl_metrics.items():
                                metrics.update({f"train/{key}": value})

                    # Compute unsupervised reward metrics
                    if unsupervised_reward and unsupervised_reward.get("enable", False):
                        if unsupervised_reward.get("type", None) == "ensemble":
                            from verl.trainer.ppo.ttrl_utils import apply_original_gt, compute_ttrl_metrics
                            batch = apply_original_gt(batch)
                            reward_tensor_original, reward_extra_infos_dict_original = compute_reward(batch, self.reward_fn)
                            batch.batch["token_level_scores_original"] = reward_tensor_original
                            n_samples_per_prompt = self.config.actor_rollout_ref.rollout.n
                            ensemble_metrics = compute_ttrl_metrics(batch, n=n_samples_per_prompt)
                            for key, value in ensemble_metrics.items():
                                metrics.update({f"train/ensemble/{key}": value})

                        if unsupervised_reward.get("type", None) == "certainty":
                            from verl.trainer.ppo.ttrl_utils import compute_certainty_metrics
                            batch.batch["token_level_scores_original"] = reward_tensor
                            certainty_metrics = compute_certainty_metrics(batch, self.config.actor_rollout_ref.rollout.n)
                            for key, value in certainty_metrics.items():
                                metrics.update({f"train/certainty/{key}": value})
                        
                        if unsupervised_reward.get("type", None) == "external" and unsupervised_reward.get("estimator", None) == "self_verify":
                            from verl.trainer.ppo.ttrl_utils import compute_self_verify_metrics
                            batch.batch["token_level_scores_original"] = reward_tensor
                            self_verify_metrics = compute_self_verify_metrics(batch)
                            for key, value in self_verify_metrics.items():
                                metrics.update({f"train/self_verify/{key}": value})

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if self.global_steps%1 == 0 and rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                        if self.config.trainer.get("save_best_by_val", True):
                            preferred_best_metric = self.config.trainer.get("best_checkpoint_metric", None)
                            best_metric_key, best_metric_value = self._select_best_checkpoint_metric(
                                val_metrics,
                                preferred_key=preferred_best_metric,
                            )
                            if best_metric_key is not None and best_metric_value is not None:
                                previous_best = self.best_val_metric_value
                                is_new_best = float(best_metric_value) > float(previous_best)

                                metrics["val-aux/best_checkpoint/current_value"] = float(best_metric_value)
                                metrics["val-aux/best_checkpoint/is_new_best"] = 1.0 if is_new_best else 0.0

                                if is_new_best:
                                    self.best_val_metric_key = best_metric_key
                                    self.best_val_metric_value = float(best_metric_value)
                                    self.best_val_step = int(self.global_steps)
                                    print(
                                        f"New best validation metric at step {self.global_steps}: "
                                        f"{best_metric_key}={best_metric_value}. Saving checkpoint."
                                    )
                                    with marked_timer("save_best_checkpoint", timing_raw, color="green"):
                                        self._save_best_checkpoint()
                                    self._save_best_checkpoint_state()

                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()
                    if self.use_rm:
                        self.rm_wg.stop_profile()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

