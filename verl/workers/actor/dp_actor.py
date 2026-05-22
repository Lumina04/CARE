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
Single Process Actor
"""

import itertools
import json
import logging
import os
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_self_certainty_from_logits = (
            torch.compile(verl_F.self_certainty_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else verl_F.self_certainty_from_logits
        )

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False, calculate_self_certainty=False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            self_certainty: # (bs, response_len) or None
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            self_certainty = None
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                self_certainty_rmpad = None

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                    if calculate_self_certainty:
                        self_certainty_rmpad = self.compute_self_certainty_from_logits(logits_rmpad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if calculate_self_certainty and self_certainty_rmpad is not None:
                        self_certainty_rmpad = gather_outpus_and_unpad(
                            self_certainty_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_self_certainty and self_certainty_rmpad is not None:
                    full_self_certainty = pad_input(
                        hidden_states=self_certainty_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if calculate_self_certainty and self_certainty_rmpad is not None:
                    self_certainty = full_self_certainty.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                    if calculate_self_certainty:
                        self_certainty = verl_F.self_certainty_from_logits(logits)

            return entropy, log_probs, self_certainty

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    def _compute_unclipped_grad_norm(self, sync_across_ranks: bool = False) -> float:
        total_sq = torch.zeros(1, device=get_device_id(), dtype=torch.float32)
        for p in self.actor_module.parameters():
            if p.grad is None:
                continue
            grad = p.grad.detach()
            total_sq += torch.sum(grad.float() * grad.float())

        if sync_across_ranks and torch.distributed.is_initialized():
            torch.distributed.all_reduce(total_sq, op=torch.distributed.ReduceOp.SUM)

        return torch.sqrt(total_sq).item()

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False, calculate_self_certainty=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        def _get_micro_batches(data: DataProto) -> Tuple[list, list | None]:
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            batch = data.select(batch_keys=select_keys).batch
            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch

            if has_multi_modal_inputs:
                all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
                if use_dynamic_bsz:
                    max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                    rearranged_text_micro_batches, textual_indices = rearrange_micro_batches(
                        batch=batch, max_token_len=max_token_len
                    )

                    final_micro_batches_list = []
                    for i, text_mb_td in enumerate(rearranged_text_micro_batches):
                        current_original_indices = textual_indices[i]
                        current_mm_inputs_list = [all_multi_modal_inputs_list[idx] for idx in current_original_indices]

                        mb_dict = {k: v for k, v in text_mb_td.items()}
                        mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                        final_micro_batches_list.append(mb_dict)
                    return final_micro_batches_list, textual_indices
                else:
                    num_micro_batches = batch.batch_size[0] // micro_batch_size
                    micro_batches_dp = data.chunk(num_micro_batches)
                    return micro_batches_dp, None
            elif use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
                return micro_batches, indices
            else:
                micro_batches = batch.split(micro_batch_size)
                return micro_batches, None

        micro_batches, indices = _get_micro_batches(data)

        log_probs_lst = []
        entropy_lst = []
        self_certainty_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, self_certainty = self._forward_micro_batch(
                    micro_batch, temperature=temperature, calculate_entropy=calculate_entropy,
                    calculate_self_certainty=calculate_self_certainty
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
            if calculate_self_certainty and self_certainty is not None:
                self_certainty_lst.append(self_certainty)

        self_certaintys = None
        if calculate_self_certainty and self_certainty_lst:
            self_certaintys = torch.concat(self_certainty_lst, dim=0)
        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if calculate_entropy:
                entropys = entropys[revert_indices]
            if calculate_self_certainty and self_certaintys is not None and self_certaintys.numel() > 0:
                self_certaintys = self_certaintys[revert_indices]

        return log_probs, entropys, self_certaintys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_probe_stage1_features(self, data: DataProto) -> Dict[str, torch.Tensor]:
        """Return prompt/response hidden states for the stage-1 max-vote correctness probe."""
        self.actor_module.eval()

        batch_size = data.batch["input_ids"].size(0)
        micro_batch_size = data.meta_info.get("micro_batch_size", None)
        if micro_batch_size is None:
            micro_batch_size = batch_size
        micro_batch_size = max(int(micro_batch_size), 1)

        use_dynamic_bsz = bool(data.meta_info.get("use_dynamic_bsz", False))
        max_token_len = data.meta_info.get("max_token_len", None)

        def _get_micro_batches(data: DataProto) -> Tuple[list, list | None]:
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            batch = data.select(batch_keys=select_keys).batch
            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch

            if has_multi_modal_inputs:
                all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
                if use_dynamic_bsz:
                    if max_token_len is None:
                        raise ValueError("max_token_len must be provided when use_dynamic_bsz=True")
                    max_token_len_with_sp = int(max_token_len) * self.ulysses_sequence_parallel_size
                    rearranged_text_micro_batches, textual_indices = rearrange_micro_batches(
                        batch=batch, max_token_len=max_token_len_with_sp
                    )

                    final_micro_batches_list = []
                    for i, text_mb_td in enumerate(rearranged_text_micro_batches):
                        current_original_indices = textual_indices[i]
                        current_mm_inputs_list = [all_multi_modal_inputs_list[idx] for idx in current_original_indices]

                        mb_dict = {k: v for k, v in text_mb_td.items()}
                        mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                        final_micro_batches_list.append(mb_dict)
                    return final_micro_batches_list, textual_indices

                num_micro_batches = max(1, (batch.batch_size[0] + micro_batch_size - 1) // micro_batch_size)
                micro_batches_dp = data.chunk(num_micro_batches)
                return micro_batches_dp, None

            if use_dynamic_bsz:
                if max_token_len is None:
                    raise ValueError("max_token_len must be provided when use_dynamic_bsz=True")
                max_token_len_with_sp = int(max_token_len) * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len_with_sp)
                return micro_batches, indices

            micro_batches = batch.split(micro_batch_size)
            return micro_batches, None

        micro_batches, indices = _get_micro_batches(data)

        prompt_hidden_list = []
        resp_hidden_list = []
        temperature = float(data.meta_info.get("temperature", 1.0))
        decoder_module = getattr(self.actor_module, "model", None)
        if decoder_module is None:
            raise RuntimeError("actor_module.model is required to compute probe hidden states")

        def _extract_hidden_from_module_output(module_output: object) -> torch.Tensor:
            hidden = getattr(module_output, "last_hidden_state", None)
            if hidden is None and isinstance(module_output, (tuple, list)) and len(module_output) > 0:
                hidden = module_output[0]
            if hidden is None:
                raise RuntimeError("failed to capture decoder last_hidden_state while computing probe features")
            return hidden

        def _forward_and_capture_hidden(**forward_kwargs) -> torch.Tensor:
            hidden_holder: Dict[str, torch.Tensor] = {}

            def _capture_hook(_module, _inputs, output):
                hidden_holder["last_hidden_state"] = _extract_hidden_from_module_output(output)

            hook_handle = decoder_module.register_forward_hook(_capture_hook)
            try:
                extra_args = {"return_dict": True, "logits_to_keep": 1}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                self.actor_module(
                    **forward_kwargs,
                    use_cache=False,
                    **extra_args,
                )
            finally:
                hook_handle.remove()

            hidden = hidden_holder.get("last_hidden_state", None)
            if hidden is None:
                raise RuntimeError("decoder forward hook did not capture last_hidden_state for probe features")
            return hidden

        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            multi_modal_inputs = {}
            if "multi_modal_inputs" in micro_batch.keys():
                if "image_bound" in micro_batch["multi_modal_inputs"][0]:
                    for key in micro_batch["multi_modal_inputs"][0].keys():
                        multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
                else:
                    for key in micro_batch["multi_modal_inputs"][0].keys():
                        multi_modal_inputs[key] = torch.cat(
                            [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                        )

            with torch.no_grad():
                with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                    input_ids = micro_batch["input_ids"]
                    batch_size_mb, seqlen = input_ids.shape
                    attention_mask = micro_batch["attention_mask"]
                    position_ids = micro_batch["position_ids"]
                    response_length = micro_batch["responses"].size(-1)
                    if position_ids.dim() == 3:
                        position_ids = position_ids.transpose(0, 1)

                    if self.use_remove_padding:
                        input_ids_rmpad, token_indices, cu_seqlens, *_ = unpad_input(
                            input_ids.unsqueeze(-1), attention_mask
                        )
                        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                        if position_ids.dim() == 3:
                            position_ids_rmpad = (
                                index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), token_indices)
                                .transpose(0, 1)
                                .unsqueeze(1)
                            )
                        else:
                            position_ids_rmpad = index_first_axis(
                                rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), token_indices
                            ).transpose(0, 1)

                        if "image_bound" in multi_modal_inputs:
                            from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                            multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                                input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                            )

                        if self.use_ulysses_sp:
                            is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                            if is_vlm_model:
                                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                                    input_ids_rmpad,
                                    position_ids_rmpad=position_ids_rmpad,
                                    sp_size=self.ulysses_sequence_parallel_size,
                                )
                            else:
                                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                                    input_ids_rmpad,
                                    position_ids_rmpad=position_ids_rmpad,
                                    sp_size=self.ulysses_sequence_parallel_size,
                                )
                        else:
                            pad_size = 0

                        hidden_rmpad = _forward_and_capture_hidden(
                            input_ids=input_ids_rmpad,
                            attention_mask=None,
                            position_ids=position_ids_rmpad,
                            **multi_modal_inputs,
                        )
                        if hidden_rmpad.dim() == 3:
                            hidden_rmpad = hidden_rmpad.squeeze(0)
                        if hidden_rmpad.dim() != 2:
                            raise RuntimeError(
                                f"expected rank-2 hidden_rmpad after squeezing, got shape={tuple(hidden_rmpad.shape)}"
                            )

                        if self.use_ulysses_sp:
                            hidden_rmpad = gather_outpus_and_unpad(
                                hidden_rmpad,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )

                        full_hidden = pad_input(
                            hidden_states=hidden_rmpad,
                            indices=token_indices,
                            batch=batch_size_mb,
                            seqlen=seqlen,
                        )
                    else:
                        full_hidden = _forward_and_capture_hidden(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            **multi_modal_inputs,
                        )

                    full_hidden = full_hidden.float()

            prompt_part_len = int(input_ids.size(-1) - response_length)
            if prompt_part_len <= 0:
                raise RuntimeError(
                    f"invalid prompt_part_len={prompt_part_len} for input shape={tuple(input_ids.shape)} "
                    f"and response shape={tuple(micro_batch['responses'].shape)}"
                )

            prompt_mask = attention_mask[:, :prompt_part_len] > 0
            response_mask = attention_mask[:, prompt_part_len:] > 0
            if not torch.all(prompt_mask.any(dim=-1)):
                raise RuntimeError("failed to locate a valid prompt token for some samples while computing probe features")
            if not torch.all(response_mask.any(dim=-1)):
                raise RuntimeError("failed to locate a valid response token for some samples while computing probe features")

            prompt_positions = torch.arange(prompt_part_len, device=prompt_mask.device, dtype=torch.long).unsqueeze(0)
            prompt_last_idx = prompt_positions.masked_fill(~prompt_mask, -1).max(dim=-1).values.clamp(min=0)
            response_lengths = response_mask.long().sum(dim=-1)
            response_last_idx = (prompt_part_len + response_lengths - 1).clamp(min=0)

            prompt_gather_idx = prompt_last_idx.view(-1, 1, 1).expand(-1, 1, full_hidden.size(-1))
            response_gather_idx = response_last_idx.view(-1, 1, 1).expand(-1, 1, full_hidden.size(-1))
            prompt_hidden = full_hidden.gather(dim=1, index=prompt_gather_idx).squeeze(1)
            resp_hidden = full_hidden.gather(dim=1, index=response_gather_idx).squeeze(1)

            prompt_hidden_list.append(prompt_hidden.float())
            resp_hidden_list.append(resp_hidden.float())

        probe_prompt_hidden = torch.cat(prompt_hidden_list, dim=0)
        probe_resp_hidden = torch.cat(resp_hidden_list, dim=0)

        if use_dynamic_bsz:
            if indices is None:
                raise RuntimeError("indices is None while use_dynamic_bsz=True in compute_probe_stage1_features")
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == probe_prompt_hidden.size(0), f"{len(indices)} vs. {probe_prompt_hidden.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long, device=probe_prompt_hidden.device)
            probe_prompt_hidden = probe_prompt_hidden[revert_indices]
            probe_resp_hidden = probe_resp_hidden[revert_indices]

        return {
            "probe_prompt_hidden": probe_prompt_hidden,
            "probe_resp_hidden": probe_resp_hidden,
        }

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        # Optional debug mode: record per-sample grad norm for one/few steps.
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        debug_sample_grad = os.getenv("VERL_DEBUG_SAMPLE_GRAD_NORM", "0").lower() in {"1", "true", "yes", "y"}
        debug_all_ranks = os.getenv("VERL_DEBUG_SAMPLE_GRAD_NORM_ALL_RANKS", "0").lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        debug_step_filter = os.getenv("VERL_DEBUG_SAMPLE_GRAD_NORM_STEP", "")
        debug_file = os.getenv("VERL_DEBUG_SAMPLE_GRAD_NORM_FILE", "/tmp/verl_sample_grad_norm.jsonl")
        debug_use_scaled_loss = os.getenv("VERL_DEBUG_SAMPLE_GRAD_NORM_USE_TRAIN_LOSS", "0").lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        try:
            debug_max_samples = int(os.getenv("VERL_DEBUG_SAMPLE_GRAD_NORM_MAX_SAMPLES", "0"))
        except ValueError:
            debug_max_samples = 0

        current_global_step = data.meta_info.get("global_step", None)
        if debug_step_filter != "":
            if current_global_step is None:
                debug_sample_grad = False
            else:
                try:
                    debug_sample_grad = debug_sample_grad and (int(debug_step_filter) == int(current_global_step))
                except ValueError:
                    debug_sample_grad = False
        if (not debug_all_ranks) and rank != 0:
            debug_sample_grad = False
        if debug_all_ranks:
            base, ext = os.path.splitext(debug_file)
            if ext == "":
                ext = ".jsonl"
            debug_file = f"{base}.rank{rank}{ext}"

        debug_records = []
        debug_written = 0
        sample_uid_list = data.non_tensor_batch.get("uid", None)
        sample_extra_info_list = data.non_tensor_batch.get("extra_info", None)

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        mini_batches = list(dataloader)
        if has_multi_modal_inputs:
            mini_batch_global_indices = [None] * len(mini_batches)
        else:
            mini_batch_global_indices = []
            cursor = 0
            for mb in mini_batches:
                mbsz = mb.batch_size[0]
                mini_batch_global_indices.append(list(range(cursor, cursor + mbsz)))
                cursor += mbsz

        def _compute_policy_objective(micro_data):
            micro_batch_metrics = {}

            response_mask = micro_data["response_mask"]
            old_log_prob = micro_data["old_log_probs"]
            advantages = micro_data["advantages"]

            clip_ratio = self.config.clip_ratio
            clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
            clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
            clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
            entropy_coeff = self.config.entropy_coeff
            loss_agg_mode = self.config.loss_agg_mode

            calculate_entropy = entropy_coeff != 0
            entropy, log_prob, _ = self._forward_micro_batch(
                micro_batch=micro_data,
                temperature=temperature,
                calculate_entropy=calculate_entropy,
                calculate_self_certainty=False,
            )

            loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

            if self.config.policy_loss.loss_mode == "vanilla":
                pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    response_mask=response_mask,
                    cliprange=clip_ratio,
                    cliprange_low=clip_ratio_low,
                    cliprange_high=clip_ratio_high,
                    clip_ratio_c=clip_ratio_c,
                    loss_agg_mode=loss_agg_mode,
                )
            else:
                policy_loss_fn = get_policy_loss_fn(loss_mode)
                pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                    old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, self.config
                )

            if entropy_coeff != 0:
                entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                policy_loss = pg_loss - entropy_loss * entropy_coeff
            else:
                policy_loss = pg_loss

            if self.config.use_kl_loss:
                ref_log_prob = micro_data["ref_log_prob"]
                kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

            micro_batch_metrics.update(
                {
                    "actor/pg_loss": pg_loss.detach().item(),
                    "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                    "actor/ppo_kl": ppo_kl.detach().item(),
                    "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                }
            )
            return policy_loss, micro_batch_metrics

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(mini_batches):
                # split batch into micro_batches
                mini_batch = data
                mini_global_indices = mini_batch_global_indices[batch_idx]
                if has_multi_modal_inputs:
                    micro_batches = []
                    micro_global_indices = None
                    if self.config.use_dynamic_bsz:
                        all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
                        batch_tensordict_for_rearrange = data.batch

                        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                        rearranged_text_micro_batches_tds, textual_indices = rearrange_micro_batches(
                            batch=batch_tensordict_for_rearrange, max_token_len=max_token_len
                        )

                        for current_original_indices, text_mb_td in zip(
                            textual_indices, rearranged_text_micro_batches_tds
                        ):
                            current_mm_inputs_list = [
                                all_multi_modal_inputs_list[idx] for idx in current_original_indices
                            ]
                            mb_dict = {k: v for k, v in text_mb_td.items()}
                            mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                            micro_batches.append(mb_dict)
                        if mini_global_indices is not None:
                            micro_global_indices = [
                                [mini_global_indices[i] for i in local_indices] for local_indices in textual_indices
                            ]
                    else:
                        self.gradient_accumulation = (
                            self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                        )
                        num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                        micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                        micro_global_indices = None
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, local_micro_indices = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                    micro_global_indices = (
                        [[mini_global_indices[i] for i in local_idx] for local_idx in local_micro_indices]
                        if mini_global_indices is not None
                        else None
                    )
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                    micro_global_indices = None
                    if mini_global_indices is not None:
                        micro_global_indices = []
                        ptr = 0
                        for mb in micro_batches:
                            mbsz = mb.batch_size[0]
                            micro_global_indices.append(mini_global_indices[ptr : ptr + mbsz])
                            ptr += mbsz

                self.actor_optimizer.zero_grad()

                for micro_batch_idx, data in enumerate(micro_batches):

                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_device_id()), **data.non_tensor_batch}
                    elif isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, torch.Tensor):
                                data[k] = v.to(get_device_id())
                            elif k == "multi_modal_inputs" and v is not None:
                                data[k] = [
                                    {kk: vv.to(get_device_id()) for kk, vv in item_dict.items()} for item_dict in v
                                ]
                            else:
                                data[k] = v
                    else:
                        data = data.to(get_device_id())  # actor device is cpu when using offload

                    if debug_sample_grad and (debug_max_samples <= 0 or debug_written < debug_max_samples):
                        micro_bsz = data["responses"].shape[0]
                        current_global_indices = (
                            micro_global_indices[micro_batch_idx]
                            if micro_global_indices is not None and micro_batch_idx < len(micro_global_indices)
                            else [None] * micro_bsz
                        )
                        for sample_idx in range(micro_bsz):
                            if debug_max_samples > 0 and debug_written >= debug_max_samples:
                                break

                            sample_data = {}
                            for key, value in data.items():
                                if isinstance(value, torch.Tensor):
                                    sample_data[key] = value[sample_idx : sample_idx + 1]
                                elif key == "multi_modal_inputs" and value is not None:
                                    sample_data[key] = [value[sample_idx]]
                                else:
                                    sample_data[key] = value

                            self.actor_optimizer.zero_grad()
                            sample_policy_loss, _ = _compute_policy_objective(sample_data)
                            if debug_use_scaled_loss:
                                if self.config.use_dynamic_bsz:
                                    sample_loss = sample_policy_loss * (1.0 / self.config.ppo_mini_batch_size)
                                else:
                                    sample_loss = sample_policy_loss / self.gradient_accumulation
                            else:
                                sample_loss = sample_policy_loss
                            sample_loss.backward()
                            sample_grad_norm = self._compute_unclipped_grad_norm(
                                sync_across_ranks=debug_all_ranks
                            )

                            global_sample_idx = (
                                current_global_indices[sample_idx]
                                if sample_idx < len(current_global_indices)
                                else None
                            )
                            uid = None
                            if global_sample_idx is not None and sample_uid_list is not None:
                                uid = str(sample_uid_list[global_sample_idx])

                            extra_index = None
                            if global_sample_idx is not None and sample_extra_info_list is not None:
                                try:
                                    extra_info = sample_extra_info_list[global_sample_idx]
                                    if isinstance(extra_info, dict):
                                        extra_index = extra_info.get("index", None)
                                except Exception:
                                    extra_index = None

                            debug_records.append(
                                {
                                    "global_step": int(current_global_step) if current_global_step is not None else -1,
                                    "ppo_epoch": int(epoch),
                                    "mini_batch_idx": int(batch_idx),
                                    "micro_batch_idx": int(micro_batch_idx),
                                    "sample_idx_in_micro_batch": int(sample_idx),
                                    "global_sample_idx": (
                                        int(global_sample_idx) if global_sample_idx is not None else None
                                    ),
                                    "uid": uid,
                                    "extra_index": extra_index,
                                    "response_len": int(sample_data["response_mask"].sum().item()),
                                    "adv_mean": float(sample_data["advantages"].mean().detach().item()),
                                    "policy_loss": float(sample_policy_loss.detach().item()),
                                    "grad_norm": float(sample_grad_norm),
                                    "rank": int(rank),
                                }
                            )
                            debug_written += 1

                        self.actor_optimizer.zero_grad()

                    policy_loss, micro_batch_metrics = _compute_policy_objective(data)
                    if self.config.use_dynamic_bsz:
                        micro_bsz = data["responses"].shape[0]
                        loss = policy_loss * (micro_bsz / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)

        if debug_sample_grad and len(debug_records) > 0:
            debug_dir = os.path.dirname(debug_file)
            if debug_dir != "":
                os.makedirs(debug_dir, exist_ok=True)
            with open(debug_file, "a", encoding="utf-8") as f:
                for item in debug_records:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"[DEBUG] wrote {len(debug_records)} per-sample grad-norm records to {debug_file}")

        self.actor_optimizer.zero_grad()
        return metrics
