# Copyright 2025 TTRL Team (https://arxiv.org/abs/2504.16084)
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
from typing import Deque, Dict, List, Optional, Sequence, Tuple
from collections import Counter, defaultdict, deque
import re
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tensordict import TensorDict
from verl import DataProto
from verl.utils.reward_score.ttrl_math import extract_answer, simplify_expression_string, grade


_KK_DATA_SOURCE_MARKERS = ("kk_logic", "knights", "knaves")


def _is_kk_data_source(data_source) -> bool:
    if data_source is None:
        return False
    source = str(data_source).lower()
    return any(marker in source for marker in _KK_DATA_SOURCE_MARKERS)


def _get_item_data_source(data_item):
    non_tensor = getattr(data_item, "non_tensor_batch", {})
    if isinstance(non_tensor, dict):
        return non_tensor.get("data_source")
    try:
        if "data_source" in non_tensor:
            return non_tensor["data_source"]
    except Exception:
        pass
    return None


def _get_prompt_data_sources(batch, num_prompts: int) -> List[Optional[str]]:
    return [_get_item_data_source(batch[i]) for i in range(num_prompts)]


def _canonicalize_kk_value(value, ground_truth=None) -> Optional[str]:
    from verl.utils.reward_score.kk import canonicalize_answer, canonicalize_ground_truth

    if value is None:
        return None
    if isinstance(value, dict):
        return canonicalize_ground_truth(value)
    return canonicalize_answer(value, ground_truth=ground_truth)


def _normalize_ttrl_answer(answer: str, data_source=None, ground_truth=None) -> Optional[str]:
    if _is_kk_data_source(data_source):
        return _canonicalize_kk_value(answer, ground_truth=ground_truth)
    return simplify_expression_string(answer)


def _extract_ttrl_answer(generated_text: str, data_source=None, ground_truth=None) -> Optional[str]:
    if _is_kk_data_source(data_source):
        from verl.utils.reward_score.kk import extract_answer_block

        answer_text = extract_answer_block(generated_text)
        if answer_text is None:
            return None
        return _normalize_ttrl_answer(answer_text, data_source=data_source, ground_truth=ground_truth)

    model_answer = extract_answer(generated_text)
    if model_answer is None:
        return None
    return _normalize_ttrl_answer(model_answer, data_source=data_source)


_PROBE_TRAIN_BUFFER_STAGE1: Optional[Deque[Dict[str, np.ndarray]]] = None
_PROBE_TRAIN_BUFFER_STAGE2: Optional[Deque[Dict[str, np.ndarray]]] = None



def _make_activation(name: str) -> nn.Module:
    key = str(name).strip().lower()
    if key == 'gelu':
        return nn.GELU()
    if key == 'relu':
        return nn.ReLU()
    if key == 'tanh':
        return nn.Tanh()
    if key == 'elu':
        return nn.ELU()
    if key == 'mish':
        return nn.Mish()
    if key in {'silu', 'swish'}:
        return nn.SiLU()
    return nn.GELU()


class PromptFeatRespMLPBinaryClassifier(nn.Module):
    """offline-v2 stage-1 probe."""

    def __init__(
        self,
        hidden_input_dim: int,
        feat_input_dim: int,
        resp_input_dim: int,
        cluster_count_input_dim: int,
        hidden_dim: int = 256,
        mid_dim: int = 256,
        resp_mlp_hidden_dim: int = 1024,
        resp_mlp_out_dim: int = 1024,
        activation_name: str = 'gelu',
    ) -> None:
        super().__init__()
        self.group_size = max(1, int(cluster_count_input_dim))
        expected_feat_dim = 1 + 2 * self.group_size
        if int(feat_input_dim) != expected_feat_dim:
            raise RuntimeError(
                f"feature dim mismatch for offline-v2 probe: feat_input_dim={feat_input_dim}, expected={expected_feat_dim}"
            )

        self.resp_aux_dim = 1

        self.prompt_proj = nn.Sequential(
            nn.BatchNorm1d(int(hidden_input_dim) + 1 + self.group_size),
            nn.Linear(int(hidden_input_dim) + 1 + self.group_size, hidden_dim),
            _make_activation(activation_name),
        )

        self.resp_proj = nn.Sequential(
            nn.BatchNorm1d(int(resp_input_dim) + self.resp_aux_dim),
            nn.Linear(int(resp_input_dim) + self.resp_aux_dim, resp_mlp_hidden_dim),
            _make_activation(activation_name),
            nn.Linear(int(resp_mlp_hidden_dim), resp_mlp_out_dim),
            _make_activation(activation_name),
        )

        fused_main_dim = hidden_dim + resp_mlp_out_dim
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(fused_main_dim),
            nn.Linear(fused_main_dim, mid_dim),
            _make_activation(activation_name),
            nn.Linear(mid_dim, 2),
        )

        fused_aux_dim = hidden_dim + resp_mlp_out_dim
        self.aux_head = nn.Sequential(
            nn.BatchNorm1d(fused_aux_dim),
            nn.Linear(fused_aux_dim, mid_dim),
            _make_activation(activation_name),
            nn.Linear(mid_dim, 2),
        )

    def encode(
        self,
        hidden_x: torch.Tensor,
        feat_x: torch.Tensor,
        resp_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_feat, resp_len_norm = _split_feat_tensor(feat_x, self.group_size)
        prompt_in = torch.cat([hidden_x, prompt_feat], dim=-1)
        prompt_repr = self.prompt_proj(prompt_in)

        b, g, d = resp_x.shape
        if g != self.group_size:
            raise RuntimeError(f"resp group size mismatch: got {g}, expected {self.group_size}")

        resp_aux = resp_len_norm.unsqueeze(-1)
        resp_in = torch.cat([resp_x, resp_aux], dim=-1)
        flat = resp_in.reshape(b * g, d + self.resp_aux_dim)
        resp_repr = self.resp_proj(flat).reshape(b, g, -1)
        resp_pool = resp_repr.mean(dim=1)
        return prompt_repr, resp_repr, resp_pool

    def forward(
        self,
        hidden_x: torch.Tensor,
        feat_x: torch.Tensor,
        resp_x: torch.Tensor,
        return_aux: bool = False,
    ):
        prompt_repr, resp_repr, resp_pool = self.encode(hidden_x, feat_x, resp_x)
        fused = torch.cat([prompt_repr, resp_pool], dim=-1)
        logits = self.classifier(fused)
        if not return_aux:
            return logits

        prompt_expand = prompt_repr.unsqueeze(1).expand(-1, resp_repr.shape[1], -1)
        aux_in = torch.cat([prompt_expand, resp_repr], dim=-1)
        b, g, c = aux_in.shape
        aux_logits = self.aux_head(aux_in.reshape(b * g, c)).reshape(b, g, -1)
        return logits, aux_logits


class PromptFeatRespMLPBinaryStage2Classifier(nn.Module):
    """offline-v2 stage-2 probe: predicts the count of correct non-vote responses."""

    def __init__(
        self,
        hidden_input_dim: int,
        feat_input_dim: int,
        resp_input_dim: int,
        cluster_count_input_dim: int,
        hidden_dim: int = 256,
        mid_dim: int = 256,
        resp_mlp_hidden_dim: int = 1024,
        resp_mlp_out_dim: int = 1024,
        activation_name: str = 'gelu',
    ) -> None:
        super().__init__()
        self.group_size = max(1, int(cluster_count_input_dim))
        expected_feat_dim = 1 + 2 * self.group_size
        if int(feat_input_dim) != expected_feat_dim:
            raise RuntimeError(
                f"feature dim mismatch for offline-v2 probe: feat_input_dim={feat_input_dim}, expected={expected_feat_dim}"
            )

        self.num_classes = self.group_size + 1

        self.prompt_proj = nn.Sequential(
            nn.BatchNorm1d(int(hidden_input_dim) + 1 + self.group_size),
            nn.Linear(int(hidden_input_dim) + 1 + self.group_size, hidden_dim),
            _make_activation(activation_name),
        )

        self.resp_proj = nn.Sequential(
            nn.BatchNorm1d(int(resp_input_dim)),
            nn.Linear(int(resp_input_dim), resp_mlp_hidden_dim),
            _make_activation(activation_name),
            nn.Linear(int(resp_mlp_hidden_dim), resp_mlp_out_dim),
            _make_activation(activation_name),
        )

        fused_dim = hidden_dim + resp_mlp_out_dim
        in_dim = fused_dim
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, mid_dim),
            _make_activation(activation_name),
            nn.Linear(mid_dim, self.num_classes),
        )

    def encode(
        self,
        hidden_x: torch.Tensor,
        feat_x: torch.Tensor,
        resp_x: torch.Tensor,
        resp_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_feat, _ = _split_feat_tensor(feat_x, self.group_size)
        prompt_in = torch.cat([hidden_x, prompt_feat], dim=-1)
        prompt_repr = self.prompt_proj(prompt_in)

        b, g, d = resp_x.shape
        if g != self.group_size:
            raise RuntimeError(f"resp group size mismatch: got {g}, expected {self.group_size}")

        if resp_valid_mask is not None:
            if tuple(resp_valid_mask.shape) != (b, g):
                raise RuntimeError(
                    f"resp_valid_mask shape mismatch: got {tuple(resp_valid_mask.shape)}, expected {(b, g)}"
                )
            mask = resp_valid_mask.to(dtype=resp_x.dtype).unsqueeze(-1)
            resp_x = resp_x * mask
        else:
            mask = None

        flat = resp_x.reshape(b * g, d)
        resp_repr = self.resp_proj(flat).reshape(b, g, -1)

        if mask is not None:
            mask_r = mask.to(dtype=resp_repr.dtype)
            resp_repr = resp_repr * mask_r
            denom = torch.clamp(mask_r.sum(dim=1), min=1.0)
            resp_pool = resp_repr.sum(dim=1) / denom
        else:
            resp_pool = resp_repr.mean(dim=1)
        return prompt_repr, resp_pool

    def forward(
        self,
        hidden_x: torch.Tensor,
        feat_x: torch.Tensor,
        resp_x: torch.Tensor,
        resp_valid_mask: Optional[torch.Tensor] = None,
    ):
        prompt_repr, resp_pool = self.encode(
            hidden_x,
            feat_x,
            resp_x,
            resp_valid_mask=resp_valid_mask,
        )
        fused = torch.cat([prompt_repr, resp_pool], dim=-1)
        logits = self.classifier(fused)
        return logits


def _split_feat_tensor(feat_x: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if feat_x.ndim != 2:
        raise RuntimeError(f"feat_x must be 2D [B,F], got shape={tuple(feat_x.shape)}")
    g = int(group_size)
    expected_dim = 1 + 2 * g
    feat_dim = int(feat_x.shape[1])
    if feat_dim != expected_dim:
        raise RuntimeError(
            f"feat dim mismatch for offline-v2 split: got {feat_dim}, expected {expected_dim} (=1+2*group_size)"
        )

    prompt_feat = feat_x[:, :1 + g]
    resp_len_norm = torch.clamp(feat_x[:, 1 + g:1 + 2 * g], 0.0, 1.0)
    return prompt_feat, resp_len_norm


def _aux_response_ce_loss(
    aux_logits: torch.Tensor,
    resp_correct_t: torch.Tensor,
    resp_valid_mask_t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if aux_logits.ndim != 3 or aux_logits.shape[-1] != 2:
        raise RuntimeError(f"aux logits must be [B,G,2], got shape={tuple(aux_logits.shape)}")
    if resp_correct_t.shape != aux_logits.shape[:2]:
        raise RuntimeError(
            f"resp_correct shape mismatch: got {tuple(resp_correct_t.shape)}, expected {tuple(aux_logits.shape[:2])}"
        )

    if resp_valid_mask_t is not None:
        if resp_valid_mask_t.shape != aux_logits.shape[:2]:
            raise RuntimeError(
                f"resp_valid_mask shape mismatch: got {tuple(resp_valid_mask_t.shape)}, expected {tuple(aux_logits.shape[:2])}"
            )
        flat_mask = resp_valid_mask_t.reshape(-1).bool()
        if not bool(torch.any(flat_mask)):
            return aux_logits.sum() * 0.0
        flat_logits = aux_logits.reshape(-1, 2)[flat_mask]
        flat_target = resp_correct_t.reshape(-1).long()[flat_mask]
    else:
        flat_logits = aux_logits.reshape(-1, 2)
        flat_target = resp_correct_t.reshape(-1).long()
        if int(flat_target.numel()) <= 0:
            return aux_logits.sum() * 0.0
    return F.cross_entropy(flat_logits, flat_target)

def _binary_cls_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.int64).reshape(-1)
    true = np.asarray(true, dtype=np.int64).reshape(-1)
    if pred.size == 0 or true.size == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
        }

    tp = int(np.sum((pred == 1) & (true == 1)))
    fp = int(np.sum((pred == 1) & (true == 0)))
    tn = int(np.sum((pred == 0) & (true == 0)))
    fn = int(np.sum((pred == 0) & (true == 1)))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if (precision + recall) <= 0 else (2.0 * precision * recall / (precision + recall))
    acc = (tp + tn) / max(1, pred.size)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
    }

def _build_binary_class_weights(
    y_train: np.ndarray,
    power: float = 0.5,
    min_w: float = 0.25,
    max_w: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    y_train = np.asarray(y_train, dtype=np.int64).reshape(-1)
    counts = np.bincount(y_train, minlength=2).astype(np.float64)

    weights = np.zeros((2,), dtype=np.float64)
    nz = counts > 0
    if np.any(nz):
        weights[nz] = 1.0 / np.power(counts[nz], max(1e-8, float(power)))
        weights[nz] = weights[nz] / float(np.mean(weights[nz]))
    weights = np.clip(weights, float(min_w), float(max_w))
    weights[~nz] = 0.0
    return weights.astype(np.float32), counts.astype(np.int64)

def _get_probe_num_classes(probe: nn.Module, default: int = 2) -> int:
    module = probe.module if hasattr(probe, "module") else probe
    return int(getattr(module, "num_classes", default))


def _mask_logits_with_valid_classes(logits: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if valid_mask is None:
        return logits
    if tuple(valid_mask.shape) != tuple(logits.shape):
        raise RuntimeError(
            f"class valid mask shape mismatch: got {tuple(valid_mask.shape)}, expected {tuple(logits.shape)}"
        )
    valid_mask = valid_mask.to(device=logits.device).bool()
    return logits.masked_fill(~valid_mask, -1.0e9)


def _masked_multiclass_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise RuntimeError(f"multiclass logits must be [B,C], got shape={tuple(logits.shape)}")
    target = target.long().reshape(-1)
    if int(target.numel()) <= 0:
        return logits.sum() * 0.0
    if int(target.numel()) != int(logits.shape[0]):
        raise RuntimeError(f"target shape mismatch: got {tuple(target.shape)}, expected batch={int(logits.shape[0])}")
    if bool(torch.any((target < 0) | (target >= int(logits.shape[-1])))):
        raise RuntimeError("stage-2 count target is outside the logits class range")
    if valid_mask is not None:
        valid_mask = valid_mask.to(device=logits.device).bool()
        target_valid = valid_mask.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        if not bool(torch.all(target_valid)):
            raise RuntimeError("stage-2 count target must be included in its valid class mask")
    masked_logits = _mask_logits_with_valid_classes(logits, valid_mask)
    return F.cross_entropy(masked_logits, target, weight=weight)


def _build_multiclass_class_weights(
    y_train: np.ndarray,
    num_classes: int,
    power: float = 0.5,
    min_w: float = 0.25,
    max_w: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    y_train = np.asarray(y_train, dtype=np.int64).reshape(-1)
    num_classes = max(1, int(num_classes))
    counts = np.bincount(np.clip(y_train, 0, num_classes - 1), minlength=num_classes).astype(np.float64)

    weights = np.zeros((num_classes,), dtype=np.float64)
    nz = counts > 0
    if np.any(nz):
        weights[nz] = 1.0 / np.power(counts[nz], max(1e-8, float(power)))
        weights[nz] = weights[nz] / float(np.mean(weights[nz]))
    weights = np.clip(weights, float(min_w), float(max_w))
    weights[~nz] = 0.0
    return weights.astype(np.float32), counts.astype(np.int64)


def _count_cls_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.int64).reshape(-1)
    true = np.asarray(true, dtype=np.int64).reshape(-1)
    if pred.size == 0 or true.size == 0:
        return {
            "macro_recall": 0.0,
            "macro_recall_nonzero": 0.0,
        }

    classes = np.unique(true).astype(np.int64)
    per_class_recall: List[float] = []
    per_class_recall_nonzero: List[float] = []

    for cls in classes:
        cls_mask = true == int(cls)
        if not np.any(cls_mask):
            continue
        recall = float(np.mean(pred[cls_mask] == true[cls_mask]))
        per_class_recall.append(recall)
        if int(cls) > 0:
            per_class_recall_nonzero.append(recall)

    return {
        "macro_recall": float(np.mean(per_class_recall)) if per_class_recall else 0.0,
        "macro_recall_nonzero": float(np.mean(per_class_recall_nonzero)) if per_class_recall_nonzero else 0.0,
    }


def _answers_equivalent(lhs, rhs, data_source=None) -> bool:
    if lhs is None or rhs is None:
        return False
    if str(lhs).strip() == str(rhs).strip():
        return True
    if _is_kk_data_source(data_source):
        lhs_simple = _canonicalize_kk_value(lhs)
        rhs_simple = _canonicalize_kk_value(rhs)
        if lhs_simple is not None and lhs_simple == rhs_simple:
            return True
    try:
        if np.isclose(_compute_binary_reward(lhs, rhs, data_source=data_source), 1.0, atol=1e-6):
            return True
    except Exception:
        pass
    try:
        if np.isclose(_compute_binary_reward(rhs, lhs, data_source=data_source), 1.0, atol=1e-6):
            return True
    except Exception:
        pass
    return False


def _cluster_answer_indices(answers: Sequence[Optional[str]], data_source=None) -> List[Tuple[str, List[int]]]:
    clusters: List[Tuple[str, List[int]]] = []
    for idx, answer in enumerate(answers):
        if answer is None:
            continue
        matched = False
        for cluster_idx, (representative, indices) in enumerate(clusters):
            if _answers_equivalent(answer, representative, data_source=data_source):
                indices.append(int(idx))
                matched = True
                break
        if not matched:
            clusters.append((answer, [int(idx)]))
    return clusters


def _is_valid_xy(n: int, x: int, y: int) -> bool:
    n = int(n)
    x = int(x)
    y = int(y)
    return 0 <= x and 0 <= y and x + y <= n and x >= y


def _build_reward_vectors_from_counts(n: int, x: int, y: int) -> Tuple[np.ndarray, np.ndarray]:
    if not _is_valid_xy(n=n, x=x, y=y):
        raise ValueError(f"Invalid (x, y)=({x}, {y}) for n={n}; require 0<=x+y<=n and x>=y")
    pseudo_reward = np.zeros(int(n), dtype=float)
    true_reward = np.zeros(int(n), dtype=float)
    pseudo_reward[: int(x)] = 1.0
    true_reward[int(x) : int(x) + int(y)] = 1.0
    return pseudo_reward, true_reward


def _compute_adv_diff_norm_from_counts(n: int, x: int, y: int) -> float:
    pseudo_reward, true_reward = _build_reward_vectors_from_counts(n=n, x=x, y=y)
    pseudo_adv = _compute_grpo_advantage(pseudo_reward)
    true_adv = _compute_grpo_advantage(true_reward)
    return float(np.linalg.norm(pseudo_adv - true_adv, ord=2))


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = float(lr)


def _top_ratio_set(scores: np.ndarray, ratio: float) -> set[int]:
    arr = np.asarray(scores, dtype=float).reshape(-1)
    n = int(arr.size)
    if n <= 0:
        return set()
    ratio = float(ratio)
    if ratio <= 0.0:
        return set()
    ratio = min(1.0, ratio)
    k = max(1, int(np.ceil(n * ratio)))
    k = min(k, n)
    order = np.argsort(-arr)
    return set(order[:k].tolist())


def _top_ratio_set_with_ties(scores: np.ndarray, ratio: float) -> set[int]:
    arr = np.asarray(scores, dtype=float).reshape(-1)
    n = int(arr.size)
    if n <= 0:
        return set()
    ratio = float(ratio)
    if ratio <= 0.0:
        return set()
    ratio = min(1.0, ratio)
    k = max(1, int(np.ceil(n * ratio)))
    k = min(k, n)
    order = np.argsort(-arr)
    threshold = float(arr[order[k - 1]])
    return set(np.flatnonzero(arr >= threshold).tolist())


def _overlap_recall_like(pred_set: set[int], true_set: set[int]) -> float:
    if len(pred_set) == 0:
        return 0.0
    return float(len(pred_set.intersection(true_set)) / float(len(pred_set)))

def select_top_k_per_prompt(data, n_votes_per_prompt, n_samples_per_prompt):
    """
    Select the first k rollouts per prompt, used for TTRL downsampling.
    """
    assert len(data) % n_votes_per_prompt == 0, "data length must be divisible by n_votes_per_prompt"
    num_prompts = len(data) // n_votes_per_prompt

    selected_indices = []
    for i in range(num_prompts):
        start = i * n_votes_per_prompt
        selected_indices.extend(range(start, start + n_samples_per_prompt))

    return data[selected_indices]


# === Ground Truth Manipulation ===
def apply_original_gt(batch):
    """
    Apply the original ground truth to the batch.
    """
    for i in range(len(batch)):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["original_gt"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = original_gt

    return batch


def apply_ttrl_gt(batch, gen_batch_output, n, tokenizer):
    """
    Apply the majority vote ground truth to the batch.
    """
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"

    model_outputs = []  
    for i in range(num_prompts):
        start = i * n
        for j in range(n):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            model_outputs.append(response_str)

    data_sources = _get_prompt_data_sources(batch, num_prompts)
    ground_truths = [batch[i].non_tensor_batch["reward_model"]["ground_truth"] for i in range(num_prompts)]
    majority_gt_list, majority_ratio_list = _batch_majority_vote(
        model_outputs,
        n,
        data_sources=data_sources,
        ground_truths=ground_truths,
    )
    
    assert len(batch) == len(majority_gt_list), "batch length must be equal to the number of model outputs"
    
    majority_vote_correct_mask = []
    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt
        majority_vote_correct_mask.append(bool(_compute_binary_reward(majority_gt_list[i], original_gt, data_source=data_sources[i])))

    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    batch.non_tensor_batch["majority_vote_correct_mask"] = np.array(majority_vote_correct_mask, dtype=bool)
    return batch


def apply_random_ranked_gt(batch, gen_batch_output, n, tokenizer, top_pct=0.1):
    """
    Random-labeling baseline for ensemble reward:
    1) Use majority vote as pseudo label by default.
    2) Randomly select top_pct prompts.
    3) Keep original GT for selected prompts.
    """
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"

    top_pct = float(top_pct)
    top_pct = max(0.0, min(1.0, top_pct))

    model_outputs = []
    for i in range(num_prompts):
        start = i * n
        for j in range(n):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            model_outputs.append(response_str)

    data_sources = _get_prompt_data_sources(batch, num_prompts)
    ground_truths = [batch[i].non_tensor_batch["reward_model"]["ground_truth"] for i in range(num_prompts)]
    majority_gt_list, majority_ratio_list = _batch_majority_vote(
        model_outputs,
        n,
        data_sources=data_sources,
        ground_truths=ground_truths,
    )

    num_gt_prompts = int(np.ceil(num_prompts * top_pct)) if top_pct > 0 else 0
    if num_gt_prompts > 0:
        random_indices = np.random.permutation(num_prompts)[:num_gt_prompts]
        random_idx_set = set(random_indices.tolist())
    else:
        random_idx_set = set()

    use_ground_truth_mask = []
    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]

        data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

        use_ground_truth = i in random_idx_set
        use_ground_truth_mask.append(use_ground_truth)
        if use_ground_truth:
            data_item.non_tensor_batch["reward_model"]["ground_truth"] = original_gt
        else:
            data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt_list[i]

    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    batch.non_tensor_batch["use_ground_truth_mask"] = np.array(use_ground_truth_mask, dtype=bool)
    return batch


def apply_avg_prob_ranked_gt(batch, gen_batch_output, n, tokenizer, response_avg_probs, top_pct=0.2):
    """
    Average-probability active-labeling baseline for ensemble reward:
    1) Compute the mean token probability for every sampled response.
    2) Average those response probabilities per prompt.
    3) Keep original GT for the lowest-probability top_pct prompts; use majority pseudo labels otherwise.
    """
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"

    top_pct = float(top_pct)
    top_pct = max(0.0, min(1.0, top_pct))
    group_size = max(1, int(n))

    response_avg_probs = np.asarray(response_avg_probs, dtype=np.float64).reshape(-1)
    if response_avg_probs.size != len(gen_batch_output):
        raise ValueError(
            "response_avg_probs must be sample-level and aligned with gen_batch_output: "
            f"got {response_avg_probs.size}, expected {len(gen_batch_output)}"
        )
    response_avg_probs = np.nan_to_num(response_avg_probs, nan=0.0, posinf=0.0, neginf=0.0)

    majority_gt_list = []
    majority_ratio_list = []
    prompt_avg_prob_list = []

    for i in range(num_prompts):
        start = i * n
        data_source = _get_item_data_source(batch[i])
        ground_truth = batch[i].non_tensor_batch["reward_model"]["ground_truth"]
        prompt_answers = []
        for j in range(n):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            model_answer = _extract_ttrl_answer(response_str, data_source=data_source, ground_truth=ground_truth)
            prompt_answers.append(model_answer)

        clusters = _cluster_answer_indices(prompt_answers, data_source=data_source)
        sorted_clusters = sorted(
            clusters,
            key=lambda item: (-len(item[1]), int(item[1][0]) if item[1] else group_size),
        )

        if sorted_clusters:
            majority_gt, majority_indices = sorted_clusters[0]
            majority_ratio = float(len(majority_indices)) / float(group_size)
        else:
            majority_gt = "None"
            majority_ratio = 0.0

        prompt_response_probs = response_avg_probs[start : start + group_size]
        prompt_avg_prob = float(np.mean(prompt_response_probs)) if prompt_response_probs.size > 0 else 0.0

        majority_gt_list.append(majority_gt)
        majority_ratio_list.append(majority_ratio)
        prompt_avg_prob_list.append(prompt_avg_prob)

    num_gt_prompts = int(np.ceil(num_prompts * top_pct)) if top_pct > 0 else 0
    num_gt_prompts = min(num_gt_prompts, num_prompts)
    if num_gt_prompts > 0:
        ranked_indices = np.argsort(np.array(prompt_avg_prob_list, dtype=float), kind="mergesort")[:num_gt_prompts]
        selected_idx_set = set(ranked_indices.tolist())
    else:
        selected_idx_set = set()

    use_ground_truth_mask = []
    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]

        data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

        use_ground_truth = i in selected_idx_set
        use_ground_truth_mask.append(use_ground_truth)
        if use_ground_truth:
            data_item.non_tensor_batch["reward_model"]["ground_truth"] = original_gt
        else:
            data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt_list[i]

    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    batch.non_tensor_batch["avg_probability_list"] = np.array(prompt_avg_prob_list, dtype=float)
    batch.non_tensor_batch["use_ground_truth_mask"] = np.array(use_ground_truth_mask, dtype=bool)
    return batch


def apply_entropy_ranked_gt(batch, gen_batch_output, n, tokenizer, top_pct=0.2):
    """
    Entropy-ranked active-labeling baseline for ensemble reward:
    1) Cluster extracted answers with the same equivalence rule as the probe scorer.
    2) Rank prompts by the entropy of their answer-cluster distribution.
    3) Keep original GT for top_pct prompts; use majority pseudo labels otherwise.
    """
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"

    top_pct = float(top_pct)
    top_pct = max(0.0, min(1.0, top_pct))
    group_size = max(1, int(n))

    majority_gt_list = []
    majority_ratio_list = []
    entropy_list = []

    for i in range(num_prompts):
        start = i * n
        data_source = _get_item_data_source(batch[i])
        ground_truth = batch[i].non_tensor_batch["reward_model"]["ground_truth"]
        prompt_answers = []
        for j in range(n):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            model_answer = _extract_ttrl_answer(response_str, data_source=data_source, ground_truth=ground_truth)
            prompt_answers.append(model_answer)

        clusters = _cluster_answer_indices(prompt_answers, data_source=data_source)
        sorted_clusters = sorted(
            clusters,
            key=lambda item: (-len(item[1]), int(item[1][0]) if item[1] else group_size),
        )

        if sorted_clusters:
            majority_gt, majority_indices = sorted_clusters[0]
            majority_ratio = float(len(majority_indices)) / float(group_size)
            cluster_sizes = np.asarray([len(indices) for _, indices in sorted_clusters], dtype=np.float64)
            probs = cluster_sizes / float(np.sum(cluster_sizes))
            entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
        else:
            majority_gt = "None"
            majority_ratio = 0.0
            entropy = 0.0

        majority_gt_list.append(majority_gt)
        majority_ratio_list.append(majority_ratio)
        entropy_list.append(entropy)

    num_gt_prompts = int(np.ceil(num_prompts * top_pct)) if top_pct > 0 else 0
    num_gt_prompts = min(num_gt_prompts, num_prompts)
    if num_gt_prompts > 0:
        ranked_indices = np.argsort(-np.array(entropy_list, dtype=float), kind="mergesort")[:num_gt_prompts]
        selected_idx_set = set(ranked_indices.tolist())
    else:
        selected_idx_set = set()

    use_ground_truth_mask = []
    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]

        data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

        use_ground_truth = i in selected_idx_set
        use_ground_truth_mask.append(use_ground_truth)
        if use_ground_truth:
            data_item.non_tensor_batch["reward_model"]["ground_truth"] = original_gt
        else:
            data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt_list[i]

    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    batch.non_tensor_batch["answer_cluster_entropy_list"] = np.array(entropy_list, dtype=float)
    batch.non_tensor_batch["use_ground_truth_mask"] = np.array(use_ground_truth_mask, dtype=bool)
    return batch


def apply_grpo_adv_diff_top_gt(batch, gen_batch_output, n, tokenizer, top_pct=0.1):
    """
    GRPO advantage-difference top-p active-labeling rule for ensemble reward:
    1) Use majority vote as pseudo label by default.
    2) For each prompt, compute pseudo-label and GT reward vectors over n responses.
    3) Convert both reward vectors into GRPO advantages: (r - mean(r)) / std(r).
    4) Compute the L2 norm of the advantage-difference vector.
    5) Rank prompts by this norm and keep original GT for top_pct prompts.
    """
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"

    top_pct = float(top_pct)
    top_pct = max(0.0, min(1.0, top_pct))

    prompt_model_answers = []
    majority_gt_list = []
    majority_ratio_list = []
    for i in range(num_prompts):
        start = i * n
        data_source = _get_item_data_source(batch[i])
        ground_truth = batch[i].non_tensor_batch["reward_model"]["ground_truth"]
        prompt_answers = []
        for j in range(n):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            model_answer = _extract_ttrl_answer(response_str, data_source=data_source, ground_truth=ground_truth)
            prompt_answers.append(model_answer)

        prompt_model_answers.append(prompt_answers)
        valid_answers = [answer for answer in prompt_answers if answer is not None]
        if len(valid_answers) == 0:
            majority_gt = "None"
            majority_ratio = 0.0
        else:
            counter = Counter(valid_answers)
            majority_gt, majority_count = counter.most_common(1)[0]
            majority_ratio = majority_count / len(prompt_answers)
        majority_gt_list.append(majority_gt)
        majority_ratio_list.append(majority_ratio)

    original_gt_list = []
    adv_diff_norm_list = []
    for i in range(num_prompts):
        data_source = _get_item_data_source(batch[i])
        original_gt = batch[i].non_tensor_batch["reward_model"]["ground_truth"]
        original_gt_list.append(original_gt)

        pseudo_reward_vec = np.array(
            [_compute_binary_reward(model_answer, majority_gt_list[i], data_source=data_source) for model_answer in prompt_model_answers[i]],
            dtype=float,
        )
        gt_reward_vec = np.array(
            [_compute_binary_reward(model_answer, original_gt, data_source=data_source) for model_answer in prompt_model_answers[i]],
            dtype=float,
        )

        pseudo_adv = _compute_grpo_advantage(pseudo_reward_vec)
        gt_adv = _compute_grpo_advantage(gt_reward_vec)
        adv_diff = gt_adv - pseudo_adv
        adv_diff_norm_list.append(float(np.linalg.norm(adv_diff, ord=2)))

    num_gt_prompts = int(np.ceil(num_prompts * top_pct)) if top_pct > 0 else 0
    num_gt_prompts = min(num_gt_prompts, num_prompts)
    if num_gt_prompts > 0:
        ranked_indices = np.argsort(-np.array(adv_diff_norm_list, dtype=float))[:num_gt_prompts]
        selected_idx_set = set(ranked_indices.tolist())
    else:
        selected_idx_set = set()

    use_ground_truth_mask = []
    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = original_gt_list[i]

        data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

        use_ground_truth = i in selected_idx_set
        use_ground_truth_mask.append(use_ground_truth)

        if use_ground_truth:
            data_item.non_tensor_batch["reward_model"]["ground_truth"] = original_gt
        else:
            data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt_list[i]

    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    batch.non_tensor_batch["adv_diff_norm_list"] = np.array(adv_diff_norm_list, dtype=float)
    batch.non_tensor_batch["use_ground_truth_mask"] = np.array(use_ground_truth_mask, dtype=bool)
    return batch

def apply_grpo_adv_diff_probe_setscorer_gt(
    batch,
    gen_batch_output,
    n,
    tokenizer,
    prompt_hidden: torch.Tensor,
    resp_last_hidden: torch.Tensor,
    probe,
    top_pct: float = 0.2,
    probe_device: str = "cpu",
    global_step: int = 0,
    prob_gate: float = 0.1,
):
    """Cascade probe scoring for active labeling with a count-class stage-2 probe."""
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"

    if not isinstance(probe, dict) or ("stage1" not in probe) or ("stage2" not in probe):
        raise RuntimeError("probe must be a dict with keys {'stage1','stage2'} for offline-v2 dual probes")
    stage1_probe = probe["stage1"]
    stage2_probe = probe["stage2"]

    top_pct = float(top_pct)
    top_pct = max(0.0, min(1.0, top_pct))
    group_size = max(1, int(n))
    stage2_num_classes = group_size + 1
    total_samples = int(len(gen_batch_output))

    if not isinstance(prompt_hidden, torch.Tensor):
        prompt_hidden = torch.as_tensor(prompt_hidden)
    if not isinstance(resp_last_hidden, torch.Tensor):
        resp_last_hidden = torch.as_tensor(resp_last_hidden)

    prompt_hidden = prompt_hidden.detach().to(dtype=torch.float32, device="cpu")
    resp_last_hidden = resp_last_hidden.detach().to(dtype=torch.float32, device="cpu")

    if prompt_hidden.dim() != 2 or prompt_hidden.size(0) != total_samples:
        raise ValueError(
            f"prompt_hidden must be rank-2 with first dim={total_samples}, got {tuple(prompt_hidden.shape)}"
        )
    if resp_last_hidden.dim() != 2 or resp_last_hidden.size(0) != total_samples:
        raise ValueError(
            f"resp_last_hidden must be rank-2 with first dim={total_samples}, got {tuple(resp_last_hidden.shape)}"
        )

    feat_dim = 1 + 2 * group_size
    prompt_hidden_np = np.zeros((num_prompts, int(prompt_hidden.size(-1))), dtype=np.float32)
    prompt_feat_np = np.zeros((num_prompts, feat_dim), dtype=np.float32)
    resp_hidden_np = np.zeros((num_prompts, group_size, int(resp_last_hidden.size(-1))), dtype=np.float32)
    resp_valid_mask_stage2_np = np.zeros((num_prompts, group_size), dtype=np.bool_)
    resp_correct_np = np.zeros((num_prompts, group_size), dtype=np.int64)
    stage2_class_valid_mask_np = np.zeros((num_prompts, stage2_num_classes), dtype=np.bool_)
    stage2_class_valid_mask_np[:, 0] = True
    stage2_cost_np = np.zeros((num_prompts, stage2_num_classes), dtype=np.float32)

    majority_gt_list: List[str] = []
    majority_ratio_list: List[float] = []
    true_vote_correct_np = np.zeros((num_prompts,), dtype=np.int64)
    oracle_correct_count_np = np.zeros((num_prompts,), dtype=np.int64)
    true_gap_np = np.zeros((num_prompts,), dtype=np.float32)
    gt_reward_all_zero_np = np.ones((num_prompts,), dtype=np.bool_)

    for i in range(num_prompts):
        start = i * n
        end = start + n
        prompt_hidden_np[i] = np.array(prompt_hidden[start], dtype=np.float32, copy=True)
        resp_hidden_np[i] = np.array(resp_last_hidden[start:end], dtype=np.float32, copy=True)

        data_source = _get_item_data_source(batch[i])
        original_gt = batch[i].non_tensor_batch["reward_model"]["ground_truth"]

        prompt_answers: List[Optional[str]] = []
        prompt_resp_lengths: List[float] = []
        for j in range(n):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            prompt_resp_lengths.append(float(valid_response_length))
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            model_answer = _extract_ttrl_answer(response_str, data_source=data_source, ground_truth=original_gt)
            if model_answer is not None:
                resp_valid_mask_stage2_np[i, j] = True
                resp_correct_np[i, j] = int(np.isclose(_compute_binary_reward(model_answer, original_gt, data_source=data_source), 1.0, atol=1e-6))
            prompt_answers.append(model_answer)

        valid_answers = [answer for answer in prompt_answers if answer is not None]
        valid_ratio = float(len(valid_answers)) / float(max(1, group_size))
        prompt_feat_np[i, 0] = np.float32(valid_ratio)
        prompt_feat_np[i, 1 + group_size:1 + 2 * group_size] = np.clip(
            np.asarray(prompt_resp_lengths, dtype=np.float32) / 4000.0,
            0.0,
            1.0,
        )

        majority_gt = "None"
        majority_ratio = 0.0
        vote_is_correct = False
        voted_count = 0
        correct_count_label = 0
        true_gap = 0.0

        clusters = _cluster_answer_indices(prompt_answers, data_source=data_source)
        sorted_clusters = sorted(
            clusters,
            key=lambda item: (-len(item[1]), int(item[1][0]) if item[1] else group_size),
        )

        if sorted_clusters:
            voted_answer, voted_indices = sorted_clusters[0]
            voted_index_set = set(int(idx) for idx in voted_indices)
            for idx in voted_indices:
                resp_valid_mask_stage2_np[i, int(idx)] = False
            voted_count = int(len(voted_indices))
            majority_gt = voted_answer
            majority_ratio = float(voted_count) / float(max(1, group_size))

            cluster_sizes = [int(len(indices)) for _, indices in sorted_clusters[:group_size]]
            cluster_count_row = np.zeros((group_size,), dtype=np.float32)
            if cluster_sizes:
                cluster_count_row[: len(cluster_sizes)] = (
                    np.asarray(cluster_sizes, dtype=np.float32) / float(group_size)
                )
            prompt_feat_np[i, 1:1 + group_size] = cluster_count_row

            non_vote_clusters = [
                (answer, indices) for answer, indices in sorted_clusters if not set(int(idx) for idx in indices).issubset(voted_index_set)
            ]
            for _, indices in non_vote_clusters:
                count_class = int(len(indices))
                if 0 <= count_class < stage2_num_classes:
                    stage2_class_valid_mask_np[i, count_class] = True

            voted_scores = np.asarray([float(resp_correct_np[i, idx]) for idx in voted_indices], dtype=np.float32)
            vote_is_correct = bool(np.any(np.isclose(voted_scores, 1.0, atol=1e-6)))

            if not vote_is_correct:
                correct_count = 0
                for _, indices in non_vote_clusters:
                    if any(np.isclose(_compute_binary_reward(prompt_answers[int(idx)], original_gt, data_source=data_source), 1.0, atol=1e-6) for idx in indices):
                        correct_count += int(len(indices))
                correct_count_label = int(min(max(correct_count, 0), group_size))

            for count_class in np.flatnonzero(stage2_class_valid_mask_np[i]).astype(np.int64).tolist():
                if _is_valid_xy(n=group_size, x=voted_count, y=int(count_class)):
                    stage2_cost_np[i, int(count_class)] = np.float32(
                        _compute_adv_diff_norm_from_counts(n=group_size, x=voted_count, y=int(count_class))
                    )
            gt_reward_vec = np.array(
                [_compute_binary_reward(model_answer, original_gt, data_source=data_source) for model_answer in prompt_answers],
                dtype=float,
            )
            gt_reward_all_zero_np[i] = bool(np.all(np.isclose(gt_reward_vec, 0.0, atol=1e-6)))
            if vote_is_correct:
                true_gap = 0.0
            else:
                true_gap = float(stage2_cost_np[i, correct_count_label])

        majority_gt_list.append(majority_gt)
        majority_ratio_list.append(majority_ratio)
        true_vote_correct_np[i] = 1 if vote_is_correct else 0
        oracle_correct_count_np[i] = int(correct_count_label)
        true_gap_np[i] = float(true_gap)

    try:
        probe_device_obj = torch.device(probe_device)
    except Exception:
        probe_device_obj = torch.device("cpu")

    hidden_t = torch.from_numpy(prompt_hidden_np).to(probe_device_obj)
    feat_t = torch.from_numpy(prompt_feat_np).to(probe_device_obj)
    resp_t = torch.from_numpy(resp_hidden_np).to(probe_device_obj)
    resp_valid_mask_stage2_t = torch.from_numpy(resp_valid_mask_stage2_np).to(probe_device_obj)
    stage2_class_valid_mask_t = torch.from_numpy(stage2_class_valid_mask_np).to(probe_device_obj)

    stage1_probe.eval()
    stage2_probe.eval()
    with torch.no_grad():
        stage1_logits = stage1_probe(hidden_x=hidden_t, feat_x=feat_t, resp_x=resp_t)
        stage1_probs = torch.softmax(stage1_logits, dim=-1).detach().to("cpu", dtype=torch.float32).numpy()
        stage2_logits = stage2_probe(
            hidden_x=hidden_t,
            feat_x=feat_t,
            resp_x=resp_t,
            resp_valid_mask=resp_valid_mask_stage2_t,
        )
        stage2_logits = _mask_logits_with_valid_classes(stage2_logits, stage2_class_valid_mask_t)
        stage2_probs = torch.softmax(stage2_logits, dim=-1).detach().to("cpu", dtype=torch.float32).numpy()

    sorted_indices = np.argsort(stage1_probs[:,1])[::-1]
    probe_gate_warmup_steps = 20
    try:
        current_global_step = int(global_step)
    except Exception:
        current_global_step = 0
    in_probe_gate_warmup = current_global_step <= probe_gate_warmup_steps
    effective_prob_gate = 0.0 if in_probe_gate_warmup else prob_gate
    k = int(np.ceil(num_prompts * effective_prob_gate)) if effective_prob_gate > 0 else 0
    k = min(max(k, 0), num_prompts)
    stage1_pred_labels = np.zeros(num_prompts, dtype=np.int64)
    if k > 0:
        stage1_pred_labels[sorted_indices[:k]] = 1
    selection_scores = np.sum(stage2_probs * stage2_cost_np, axis=1).astype(np.float32, copy=False)

    cls_metrics = _binary_cls_metrics(stage1_pred_labels, true_vote_correct_np)

    stage1_pred1_mask = stage1_pred_labels == 1
    stage1_pred1_total_count = int(np.sum(stage1_pred1_mask))
    stage1_pred1_true1_count = int(np.sum(stage1_pred1_mask & (true_vote_correct_np == 1)))
    stage1_pred1_wrong_count = int(stage1_pred1_total_count - stage1_pred1_true1_count)
    stage1_pred1_true1_ratio = float(stage1_pred1_true1_count / max(1, stage1_pred1_total_count))
    stage2_eval_mask = stage1_pred_labels == 0
    stage2_pred_raw = np.argmax(stage2_probs, axis=1).astype(np.int64)
    stage2_count_metrics = _count_cls_metrics(
        stage2_pred_raw[stage2_eval_mask],
        oracle_correct_count_np[stage2_eval_mask],
    )

    num_gt_prompts = int(np.ceil(num_prompts * top_pct)) if top_pct > 0 else 0
    num_gt_prompts = min(num_gt_prompts, num_prompts)
    selected_indices: List[int] = []
    rank_scores = np.array(selection_scores, dtype=np.float32, copy=True)
    rank_scores[stage1_pred_labels == 1] = -1.0
    selected_indices = [
        int(x) for x in np.argsort(-rank_scores, kind="mergesort")[:num_gt_prompts].tolist()
    ]
    selected_idx_set = set(selected_indices)
    selected_indices_np = np.asarray(selected_indices, dtype=np.int64)
    if selected_indices_np.size > 0:
        selected_gt_reward_all_zero_count = int(np.sum(gt_reward_all_zero_np[selected_indices_np]))
        selected_gt_reward_all_zero_ratio = float(
            selected_gt_reward_all_zero_count / max(1, int(selected_indices_np.size))
        )
        selected_vote_correct_ratio = float(np.mean(true_vote_correct_np[selected_indices_np].astype(np.float32)))
    else:
        selected_gt_reward_all_zero_count = 0
        selected_gt_reward_all_zero_ratio = 0.0
        selected_vote_correct_ratio = 0.0

    top_pct_100 = float(top_pct) * 100.0
    if abs(top_pct_100 - round(top_pct_100)) < 1e-8:
        top_pct_label = str(int(round(top_pct_100)))
    else:
        top_pct_label = str(top_pct_100).rstrip("0").rstrip(".").replace(".", "p")

    adv_diff_norm_list = np.array(selection_scores, dtype=np.float32, copy=True)
    adv_diff_norm_list[stage1_pred_labels == 0] = 0.0

    rank_scores_for_metric = np.array(selection_scores, dtype=np.float32, copy=True)
    rank_scores_for_metric[stage1_pred_labels == 1] = -1.0
    top5_pred_set = _top_ratio_set(rank_scores_for_metric, 0.05)
    top10_pred_set = _top_ratio_set(rank_scores_for_metric, 0.10)
    topp_pred_set = _top_ratio_set(rank_scores_for_metric, top_pct)
    top5_true_set = _top_ratio_set_with_ties(true_gap_np, 0.05)
    top10_true_set = _top_ratio_set_with_ties(true_gap_np, 0.10)
    topp_true_set = _top_ratio_set_with_ties(true_gap_np, top_pct)

    recall_top5 = _overlap_recall_like(top5_pred_set, top5_true_set)
    recall_top10 = _overlap_recall_like(top10_pred_set, top10_true_set)
    recall_topp = _overlap_recall_like(topp_pred_set, topp_true_set)

    use_ground_truth_mask = np.zeros((num_prompts,), dtype=bool)
    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]

        data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

        use_ground_truth = i in selected_idx_set
        use_ground_truth_mask[i] = use_ground_truth
        if use_ground_truth:
            data_item.non_tensor_batch["reward_model"]["ground_truth"] = original_gt
        else:
            data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt_list[i]

    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    batch.non_tensor_batch["use_ground_truth_mask"] = use_ground_truth_mask
    batch.non_tensor_batch["adv_diff_norm_list"] = adv_diff_norm_list.astype(float)

    probe_metrics: Dict[str, float] = {
        "probe_cls_precision": float(cls_metrics["precision"]),
        "probe_cls_recall": float(cls_metrics["recall"]),
        "probe_cls_f1": float(cls_metrics["f1"]),
        "probe_cls_accuracy": float(cls_metrics["accuracy"]),
        "probe_stage1_pred1_total_count": float(stage1_pred1_total_count),
        "probe_stage1_pred1_true1_count": float(stage1_pred1_true1_count),
        "probe_stage1_pred1_wrong_count": float(stage1_pred1_wrong_count),
        "probe_stage1_pred1_true1_ratio": float(stage1_pred1_true1_ratio),
        "probe_stage2_count_macro_recall": float(stage2_count_metrics["macro_recall"]),
        "probe_stage2_count_macro_recall_nonzero": float(stage2_count_metrics["macro_recall_nonzero"]),
        "probe_stage2_expected_cost_mean": float(np.mean(selection_scores)) if selection_scores.size > 0 else 0.0,
        "probe_recall_predtop_true_top5": float(recall_top5),
        "probe_recall_predtop_true_top10": float(recall_top10),
        f"probe_recall_predtop_true_top{top_pct_label}": float(recall_topp),
        "probe_selected_gt_reward_all_zero_ratio": float(selected_gt_reward_all_zero_ratio),
        "probe_selected_gt_reward_all_zero_count": float(selected_gt_reward_all_zero_count),
        "probe_selected_gt_count": float(selected_indices_np.size),
        "probe_selected_vote_correct_ratio": float(selected_vote_correct_ratio),
    }

    probe_train_batch: Optional[Dict[str, np.ndarray]] = None
    if num_prompts > 0 and num_gt_prompts > 0:
        train_indices = np.array(selected_indices, dtype=np.int64)
        stage1_label_train = np.array(true_vote_correct_np[train_indices], dtype=np.int64, copy=True)
        probe_train_batch = {
            "hidden": np.array(prompt_hidden_np[train_indices], dtype=np.float32, copy=True),
            "feat": np.array(prompt_feat_np[train_indices], dtype=np.float32, copy=True),
            "resp": np.array(resp_hidden_np[train_indices], dtype=np.float32, copy=True),
            "resp_valid_mask_stage2": np.array(resp_valid_mask_stage2_np[train_indices], dtype=np.bool_, copy=True),
            "stage2_class_valid_mask": np.array(stage2_class_valid_mask_np[train_indices], dtype=np.bool_, copy=True),
            "resp_correct": np.array(resp_correct_np[train_indices], dtype=np.int64, copy=True),
            "stage1_label": stage1_label_train,
            "stage2_label": np.array(oracle_correct_count_np[train_indices], dtype=np.int64, copy=True),
        }

    return batch, probe_metrics, probe_train_batch

def train_grpo_adv_diff_probe_setscorer(
    probe,
    probe_optimizer,
    probe_train_batch: Optional[Dict[str, np.ndarray]],
    probe_device: str = "cpu",
    train_steps: int = 1,
    probe_grad_clip_norm: float = 1.0,
    probe_buffer_size: int = 0,
    probe_buffer_batch_size: int = 0,
    lr: float = 1e-3,
    aux_loss_weight: float = 1.0,
    class_balance_enable: bool = True,
    class_balance_power: float = 0.5,
    class_balance_min: float = 0.25,
    class_balance_max: float = 4.0,
) -> Dict[str, float]:
    """Train offline-v2 aligned dual probes with a count-class stage-2 objective."""
    metrics: Dict[str, float] = {
        "probe_loss": 0.0,
        "probe_stage1_loss": 0.0,
        "probe_stage2_loss": 0.0,
        "probe_stage1_aux_loss": 0.0,
        "probe_buffer_size": 0.0,
        "probe_buffer_size_stage1": 0.0,
        "probe_buffer_size_stage2": 0.0,
        "probe_buffer_train_size": 0.0,
        "probe_buffer_train_size_stage1": 0.0,
        "probe_buffer_train_size_stage2": 0.0,
        "probe_train_samples": 0.0,
        "probe_stage2_train_samples": 0.0,
        "probe_lr": 0.0,
    }

    if (
        not isinstance(probe, dict)
        or "stage1" not in probe
        or "stage2" not in probe
        or not isinstance(probe_optimizer, dict)
        or "stage1" not in probe_optimizer
        or "stage2" not in probe_optimizer
    ):
        return metrics

    stage1_probe = probe["stage1"]
    stage2_probe = probe["stage2"]
    stage1_optimizer = probe_optimizer["stage1"]
    stage2_optimizer = probe_optimizer["stage2"]
    if stage1_probe is None or stage2_probe is None or stage1_optimizer is None or stage2_optimizer is None:
        return metrics

    try:
        probe_device_obj = torch.device(probe_device)
    except Exception:
        probe_device_obj = torch.device("cpu")

    probe_buffer_size = max(0, int(probe_buffer_size))
    probe_buffer_batch_size = max(0, int(probe_buffer_batch_size))
    aux_loss_weight = float(aux_loss_weight)
    class_balance_enable = bool(class_balance_enable)
    stage2_num_classes = _get_probe_num_classes(stage2_probe, default=2)

    step_lr = float(lr)
    _set_optimizer_lr(stage1_optimizer, step_lr)
    _set_optimizer_lr(stage2_optimizer, step_lr)
    metrics["probe_lr"] = float(step_lr)

    hidden_np = np.zeros((0,), dtype=np.float32)
    feat_np = np.zeros((0,), dtype=np.float32)
    resp_np = np.zeros((0,), dtype=np.float32)
    resp_valid_mask_stage2_np = np.zeros((0,), dtype=np.bool_)
    stage2_class_valid_mask_np = np.zeros((0, stage2_num_classes), dtype=np.bool_)
    resp_correct_np = np.zeros((0,), dtype=np.int64)
    stage1_label_np = np.zeros((0,), dtype=np.int64)
    stage2_label_np = np.zeros((0,), dtype=np.int64)
    current_count = 0

    if probe_train_batch is not None:
        hidden_np = np.asarray(probe_train_batch.get("hidden", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        feat_np = np.asarray(probe_train_batch.get("feat", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        resp_np = np.asarray(probe_train_batch.get("resp", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        resp_valid_mask_stage2_np = np.asarray(
            probe_train_batch["resp_valid_mask_stage2"],
            dtype=np.bool_,
        )
        stage2_class_valid_mask_np = np.asarray(
            probe_train_batch.get("stage2_class_valid_mask", np.zeros((0, stage2_num_classes), dtype=np.bool_)),
            dtype=np.bool_,
        )
        resp_correct_np = np.asarray(
            probe_train_batch.get("resp_correct", np.zeros((0,), dtype=np.int64)),
            dtype=np.int64,
        )
        stage1_label_np = np.asarray(
            probe_train_batch.get("stage1_label", np.zeros((0,), dtype=np.int64)),
            dtype=np.int64,
        ).reshape(-1)
        stage2_label_np = np.asarray(
            probe_train_batch.get("stage2_label", np.zeros((0,), dtype=np.int64)),
            dtype=np.int64,
        ).reshape(-1)
        if hidden_np.ndim == 1 and hidden_np.size > 0:
            hidden_np = hidden_np.reshape(1, -1)
        if feat_np.ndim == 1 and feat_np.size > 0:
            feat_np = feat_np.reshape(1, -1)
        if resp_np.ndim == 2 and resp_np.size > 0:
            resp_np = resp_np.reshape(1, resp_np.shape[0], resp_np.shape[1])
        if resp_valid_mask_stage2_np.ndim == 1 and resp_valid_mask_stage2_np.size > 0:
            resp_valid_mask_stage2_np = resp_valid_mask_stage2_np.reshape(1, -1)
        if stage2_class_valid_mask_np.ndim == 1 and stage2_class_valid_mask_np.size > 0:
            stage2_class_valid_mask_np = stage2_class_valid_mask_np.reshape(1, -1)
        if resp_correct_np.ndim == 1 and resp_correct_np.size > 0:
            resp_correct_np = resp_correct_np.reshape(1, -1)

        current_count = int(hidden_np.shape[0]) if hidden_np.ndim >= 2 else 0
        if current_count > 0:
            if stage2_class_valid_mask_np.size == 0:
                stage2_class_valid_mask_np = np.ones((current_count, stage2_num_classes), dtype=np.bool_)
            if stage2_class_valid_mask_np.shape[-1] != stage2_num_classes:
                fixed_mask = np.zeros((current_count, stage2_num_classes), dtype=np.bool_)
                width = min(stage2_num_classes, int(stage2_class_valid_mask_np.shape[-1]))
                fixed_mask[:, :width] = stage2_class_valid_mask_np[:, :width]
                stage2_class_valid_mask_np = fixed_mask
            target_rows = np.arange(current_count, dtype=np.int64)
            safe_targets = np.clip(stage2_label_np, 0, stage2_num_classes - 1)
            stage2_class_valid_mask_np[target_rows, safe_targets] = True
            if (
                feat_np.shape[0] != current_count
                or resp_np.shape[0] != current_count
                or resp_valid_mask_stage2_np.shape[0] != current_count
                or stage2_class_valid_mask_np.shape[0] != current_count
                or resp_correct_np.shape[0] != current_count
                or stage1_label_np.shape[0] != current_count
                or stage2_label_np.shape[0] != current_count
            ):
                raise ValueError("probe_train_batch arrays must share the same first dimension")

    current_samples = []
    if current_count > 0:
        current_samples = [
            {
                "hidden": np.array(hidden_np[j], dtype=np.float32, copy=True),
                "feat": np.array(feat_np[j], dtype=np.float32, copy=True),
                "resp": np.array(resp_np[j], dtype=np.float32, copy=True),
                "resp_valid_mask_stage2": np.array(resp_valid_mask_stage2_np[j], dtype=np.bool_, copy=True),
                "stage2_class_valid_mask": np.array(stage2_class_valid_mask_np[j], dtype=np.bool_, copy=True),
                "resp_correct": np.array(resp_correct_np[j], dtype=np.int64, copy=True),
                "stage1_label": int(stage1_label_np[j]),
                "stage2_label": int(stage2_label_np[j]),
            }
            for j in range(current_count)
        ]

    current_stage1_samples = current_samples
    current_stage2_samples = current_samples

    buffer_stage1 = None
    buffer_stage2 = None
    history_size_stage1 = 0
    history_size_stage2 = 0
    history_train_size_stage1 = 0
    history_train_size_stage2 = 0

    if probe_buffer_size > 0:
        global _PROBE_TRAIN_BUFFER_STAGE1, _PROBE_TRAIN_BUFFER_STAGE2
        if _PROBE_TRAIN_BUFFER_STAGE1 is None:
            _PROBE_TRAIN_BUFFER_STAGE1 = deque(maxlen=probe_buffer_size)
        if _PROBE_TRAIN_BUFFER_STAGE2 is None:
            _PROBE_TRAIN_BUFFER_STAGE2 = deque(maxlen=probe_buffer_size)

        buffer_stage1 = _PROBE_TRAIN_BUFFER_STAGE1
        buffer_stage2 = _PROBE_TRAIN_BUFFER_STAGE2
        history_size_stage1 = len(buffer_stage1)
        history_size_stage2 = len(buffer_stage2)

        if probe_buffer_batch_size > 0:
            history_train_size_stage1 = int(min(history_size_stage1, probe_buffer_batch_size))
            history_train_size_stage2 = int(min(history_size_stage2, probe_buffer_batch_size))
        else:
            history_train_size_stage1 = int(history_size_stage1)
            history_train_size_stage2 = int(history_size_stage2)

        metrics["probe_buffer_size_stage1"] = float(history_size_stage1)
        metrics["probe_buffer_size_stage2"] = float(history_size_stage2)
        metrics["probe_buffer_size"] = float(max(history_size_stage1, history_size_stage2))
        metrics["probe_buffer_train_size_stage1"] = float(history_train_size_stage1)
        metrics["probe_buffer_train_size_stage2"] = float(history_train_size_stage2)
        metrics["probe_buffer_train_size"] = float(max(history_train_size_stage1, history_train_size_stage2))

    if (
        len(current_stage1_samples) == 0
        and len(current_stage2_samples) == 0
        and history_train_size_stage1 <= 0
        and history_train_size_stage2 <= 0
    ):
        return metrics

    def _sample_history(
        buffer_obj,
        history_n: int,
        stratify_by_stage2_label: bool = False,
    ) -> List[Dict[str, np.ndarray]]:
        if buffer_obj is None or history_n <= 0:
            return []
        size_now = len(buffer_obj)
        if size_now <= 0:
            return []
        history_n = int(min(int(history_n), size_now))
        if not stratify_by_stage2_label or history_n >= size_now:
            picked = np.arange(size_now) if history_n >= size_now else np.random.choice(size_now, size=history_n, replace=False)
        else:
            label_to_indices: Dict[int, List[int]] = defaultdict(list)
            for idx, sample in enumerate(buffer_obj):
                label = int(sample.get("stage2_label", 0))
                label = max(0, min(stage2_num_classes - 1, label))
                label_to_indices[label].append(int(idx))
            buckets = []
            for label in sorted(label_to_indices.keys()):
                bucket = np.asarray(label_to_indices[label], dtype=np.int64)
                if bucket.size > 0:
                    bucket = np.random.permutation(bucket)
                    buckets.append(bucket.tolist())
            picked_list: List[int] = []
            if buckets:
                base = max(1, history_n // len(buckets))
                leftovers: List[int] = []
                for bucket in buckets:
                    take = min(base, len(bucket), history_n - len(picked_list))
                    if take > 0:
                        picked_list.extend(bucket[:take])
                    if take < len(bucket):
                        leftovers.extend(bucket[take:])
                    if len(picked_list) >= history_n:
                        break
                if len(picked_list) < history_n and leftovers:
                    leftovers = np.random.permutation(np.asarray(leftovers, dtype=np.int64)).tolist()
                    need = history_n - len(picked_list)
                    picked_list.extend([int(x) for x in leftovers[:need]])
            if len(picked_list) < history_n:
                already = set(int(x) for x in picked_list)
                rest = [idx for idx in range(size_now) if idx not in already]
                if rest:
                    rest = np.random.permutation(np.asarray(rest, dtype=np.int64)).tolist()
                    picked_list.extend([int(x) for x in rest[: history_n - len(picked_list)]])
            picked = np.asarray(picked_list[:history_n], dtype=np.int64)
        out: List[Dict[str, np.ndarray]] = []
        for k in np.asarray(picked, dtype=np.int64).tolist():
            s = buffer_obj[int(k)]
            class_mask = np.array(
                s.get("stage2_class_valid_mask", np.ones((stage2_num_classes,), dtype=np.bool_)),
                dtype=np.bool_,
                copy=True,
            )
            if class_mask.shape[0] != stage2_num_classes:
                fixed_mask = np.zeros((stage2_num_classes,), dtype=np.bool_)
                width = min(stage2_num_classes, int(class_mask.shape[0]))
                fixed_mask[:width] = class_mask[:width]
                class_mask = fixed_mask
            label = int(s["stage2_label"])
            if 0 <= label < stage2_num_classes:
                class_mask[label] = True
            out.append(
                {
                    "hidden": np.array(s["hidden"], dtype=np.float32, copy=True),
                    "feat": np.array(s["feat"], dtype=np.float32, copy=True),
                    "resp": np.array(s["resp"], dtype=np.float32, copy=True),
                    "resp_valid_mask_stage2": np.array(s["resp_valid_mask_stage2"], dtype=np.bool_, copy=True),
                    "stage2_class_valid_mask": class_mask,
                    "resp_correct": np.array(s["resp_correct"], dtype=np.int64, copy=True),
                    "stage1_label": int(s["stage1_label"]),
                    "stage2_label": label,
                }
            )
        return out

    train_steps = max(int(train_steps), 1)
    stage1_probe.train()
    stage2_probe.train()

    stage1_losses: List[float] = []
    stage2_losses: List[float] = []
    stage1_aux_losses: List[float] = []
    last_stage1_sample_n = 0
    last_stage2_sample_n = 0

    for _ in range(train_steps):
        stage1_samples = [
            {
                "hidden": np.array(s["hidden"], dtype=np.float32, copy=True),
                "feat": np.array(s["feat"], dtype=np.float32, copy=True),
                "resp": np.array(s["resp"], dtype=np.float32, copy=True),
                "resp_valid_mask_stage2": np.array(s["resp_valid_mask_stage2"], dtype=np.bool_, copy=True),
                "stage2_class_valid_mask": np.array(s["stage2_class_valid_mask"], dtype=np.bool_, copy=True),
                "resp_correct": np.array(s["resp_correct"], dtype=np.int64, copy=True),
                "stage1_label": int(s["stage1_label"]),
                "stage2_label": int(s["stage2_label"]),
            }
            for s in current_stage1_samples
        ]
        stage2_samples = [
            {
                "hidden": np.array(s["hidden"], dtype=np.float32, copy=True),
                "feat": np.array(s["feat"], dtype=np.float32, copy=True),
                "resp": np.array(s["resp"], dtype=np.float32, copy=True),
                "resp_valid_mask_stage2": np.array(s["resp_valid_mask_stage2"], dtype=np.bool_, copy=True),
                "stage2_class_valid_mask": np.array(s["stage2_class_valid_mask"], dtype=np.bool_, copy=True),
                "resp_correct": np.array(s["resp_correct"], dtype=np.int64, copy=True),
                "stage1_label": int(s["stage1_label"]),
                "stage2_label": int(s["stage2_label"]),
            }
            for s in current_stage2_samples
        ]

        stage1_samples.extend(_sample_history(buffer_stage1, history_train_size_stage1))
        stage2_samples.extend(
            _sample_history(buffer_stage2, history_train_size_stage2, stratify_by_stage2_label=True)
        )

        if stage1_samples:
            hidden_t = torch.from_numpy(np.stack([s["hidden"] for s in stage1_samples], axis=0)).to(probe_device_obj)
            feat_t = torch.from_numpy(np.stack([s["feat"] for s in stage1_samples], axis=0)).to(probe_device_obj)
            resp_t = torch.from_numpy(np.stack([s["resp"] for s in stage1_samples], axis=0)).to(probe_device_obj)
            resp_correct_t = torch.from_numpy(np.stack([s["resp_correct"] for s in stage1_samples], axis=0)).to(probe_device_obj)

            stage1_label_np = np.asarray([s["stage1_label"] for s in stage1_samples], dtype=np.int64)
            stage1_label_t = torch.from_numpy(stage1_label_np).to(probe_device_obj)

            stage1_weight_t = None
            if class_balance_enable:
                stage1_weights_np, _ = _build_binary_class_weights(
                    y_train=stage1_label_np,
                    power=class_balance_power,
                    min_w=class_balance_min,
                    max_w=class_balance_max,
                )
                stage1_weight_t = torch.from_numpy(stage1_weights_np).to(probe_device_obj)

            stage1_optimizer.zero_grad(set_to_none=True)
            stage1_logits, stage1_aux_logits = stage1_probe(
                hidden_x=hidden_t,
                feat_x=feat_t,
                resp_x=resp_t,
                return_aux=True,
            )
            stage1_main_loss = F.cross_entropy(stage1_logits, stage1_label_t, weight=stage1_weight_t)
            stage1_aux_loss = _aux_response_ce_loss(stage1_aux_logits, resp_correct_t)
            stage1_loss = stage1_main_loss + float(aux_loss_weight) * stage1_aux_loss
            stage1_loss.backward()
            if probe_grad_clip_norm is not None and float(probe_grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(stage1_probe.parameters(), max_norm=float(probe_grad_clip_norm))
            stage1_optimizer.step()

            stage1_losses.append(float(stage1_loss.item()))
            stage1_aux_losses.append(float(stage1_aux_loss.item()))
            last_stage1_sample_n = int(stage1_label_np.size)

        if stage2_samples:
            hidden_t = torch.from_numpy(np.stack([s["hidden"] for s in stage2_samples], axis=0)).to(probe_device_obj)
            feat_t = torch.from_numpy(np.stack([s["feat"] for s in stage2_samples], axis=0)).to(probe_device_obj)
            resp_t = torch.from_numpy(np.stack([s["resp"] for s in stage2_samples], axis=0)).to(probe_device_obj)
            resp_valid_mask_stage2_t = torch.from_numpy(
                np.stack([s["resp_valid_mask_stage2"] for s in stage2_samples], axis=0)
            ).to(probe_device_obj)
            stage2_class_valid_mask_t = torch.from_numpy(
                np.stack([s["stage2_class_valid_mask"] for s in stage2_samples], axis=0)
            ).to(probe_device_obj)
            stage2_label_np = np.asarray([s["stage2_label"] for s in stage2_samples], dtype=np.int64)
            stage2_label_t = torch.from_numpy(stage2_label_np).to(probe_device_obj)

            stage2_weight_t = None
            if class_balance_enable:
                stage2_weights_np, _ = _build_multiclass_class_weights(
                    y_train=stage2_label_np,
                    num_classes=stage2_num_classes,
                    power=class_balance_power,
                    min_w=class_balance_min,
                    max_w=class_balance_max,
                )
                stage2_weight_t = torch.from_numpy(stage2_weights_np).to(probe_device_obj)

            stage2_optimizer.zero_grad(set_to_none=True)
            stage2_logits = stage2_probe(
                hidden_x=hidden_t,
                feat_x=feat_t,
                resp_x=resp_t,
                resp_valid_mask=resp_valid_mask_stage2_t,
            )
            stage2_main_loss = _masked_multiclass_ce_loss(
                stage2_logits,
                stage2_label_t,
                valid_mask=stage2_class_valid_mask_t,
                weight=stage2_weight_t,
            )
            stage2_loss = stage2_main_loss
            stage2_loss.backward()
            if probe_grad_clip_norm is not None and float(probe_grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(stage2_probe.parameters(), max_norm=float(probe_grad_clip_norm))
            stage2_optimizer.step()

            stage2_losses.append(float(stage2_loss.item()))
            last_stage2_sample_n = int(stage2_label_np.size)

    if buffer_stage1 is not None:
        for s in current_stage1_samples:
            buffer_stage1.append(
                {
                    "hidden": np.array(s["hidden"], dtype=np.float32, copy=True),
                    "feat": np.array(s["feat"], dtype=np.float32, copy=True),
                    "resp": np.array(s["resp"], dtype=np.float32, copy=True),
                    "resp_valid_mask_stage2": np.array(s["resp_valid_mask_stage2"], dtype=np.bool_, copy=True),
                    "stage2_class_valid_mask": np.array(s["stage2_class_valid_mask"], dtype=np.bool_, copy=True),
                    "resp_correct": np.array(s["resp_correct"], dtype=np.int64, copy=True),
                    "stage1_label": int(s["stage1_label"]),
                    "stage2_label": int(s["stage2_label"]),
                }
            )
    if buffer_stage2 is not None:
        for s in current_stage2_samples:
            buffer_stage2.append(
                {
                    "hidden": np.array(s["hidden"], dtype=np.float32, copy=True),
                    "feat": np.array(s["feat"], dtype=np.float32, copy=True),
                    "resp": np.array(s["resp"], dtype=np.float32, copy=True),
                    "resp_valid_mask_stage2": np.array(s["resp_valid_mask_stage2"], dtype=np.bool_, copy=True),
                    "stage2_class_valid_mask": np.array(s["stage2_class_valid_mask"], dtype=np.bool_, copy=True),
                    "resp_correct": np.array(s["resp_correct"], dtype=np.int64, copy=True),
                    "stage1_label": int(s["stage1_label"]),
                    "stage2_label": int(s["stage2_label"]),
                }
            )

    if buffer_stage1 is not None:
        metrics["probe_buffer_size_stage1"] = float(len(buffer_stage1))
    if buffer_stage2 is not None:
        metrics["probe_buffer_size_stage2"] = float(len(buffer_stage2))
    metrics["probe_buffer_size"] = float(max(metrics["probe_buffer_size_stage1"], metrics["probe_buffer_size_stage2"]))

    metrics["probe_train_samples"] = float(last_stage1_sample_n)
    metrics["probe_stage2_train_samples"] = float(last_stage2_sample_n)

    stage1_loss_mean = float(np.mean(np.asarray(stage1_losses, dtype=np.float32))) if stage1_losses else 0.0
    stage2_loss_mean = float(np.mean(np.asarray(stage2_losses, dtype=np.float32))) if stage2_losses else 0.0
    stage1_aux_mean = float(np.mean(np.asarray(stage1_aux_losses, dtype=np.float32))) if stage1_aux_losses else 0.0

    metrics["probe_stage1_loss"] = stage1_loss_mean
    metrics["probe_stage2_loss"] = stage2_loss_mean
    metrics["probe_stage1_aux_loss"] = stage1_aux_mean
    metrics["probe_loss"] = stage1_loss_mean + stage2_loss_mean
    return metrics

def _compute_binary_reward(model_answer, gt_answer, data_source=None) -> float:
    if model_answer is None:
        return 0.0

    if _is_kk_data_source(data_source):
        gt_candidates = gt_answer if isinstance(gt_answer, (list, tuple)) else [gt_answer]
        for candidate in gt_candidates:
            model_simple = _canonicalize_kk_value(model_answer, ground_truth=candidate)
            candidate_simple = _canonicalize_kk_value(candidate)
            if model_simple is not None and candidate_simple is not None and model_simple == candidate_simple:
                return 1.0
        return 0.0

    gt_candidates = gt_answer if isinstance(gt_answer, (list, tuple)) else [gt_answer]
    for candidate in gt_candidates:
        candidate_text = str(candidate)
        try:
            if grade(model_answer, candidate_text):
                return 1.0
        except Exception:
            # Preserve the original math fallback: exact-match only when the math grader raises.
            if str(model_answer).strip() == candidate_text.strip():
                return 1.0
    return 0.0

def _compute_grpo_advantage(reward_vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    reward_vec = np.asarray(reward_vec, dtype=float)
    reward_std = float(np.std(reward_vec))
    if reward_std < eps:
        return np.zeros_like(reward_vec, dtype=float)
    reward_mean = float(np.mean(reward_vec))
    return (reward_vec - reward_mean) / reward_std


def _batch_majority_vote(
    model_outputs: List[str],
    n: int,
    data_sources: Optional[Sequence[Optional[str]]] = None,
    ground_truths: Optional[Sequence[object]] = None,
) -> tuple[List[str], List[float]]:
    """
    Used to generate the ground truth for TTRL.
    Input:
        model_outputs: list of str
        n: int
    Output:
        majority_gt_list: list of str
        majority_ratio_list: list of float
    """
    majority_gt_list = []
    majority_ratio_list = []
    assert len(model_outputs) % n == 0
    n_prompts = len(model_outputs) // n
    for i in range(n_prompts):
        prompt_outputs = model_outputs[i * n:(i + 1) * n]
        data_source = data_sources[i] if data_sources is not None and i < len(data_sources) else None
        ground_truth = ground_truths[i] if ground_truths is not None and i < len(ground_truths) else None
        prompt_majority_gt, prompt_majority_ratio = _majority_vote(prompt_outputs, data_source=data_source, ground_truth=ground_truth)
        majority_gt_list.append(prompt_majority_gt)
        majority_ratio_list.append(prompt_majority_ratio)
        
    return majority_gt_list, majority_ratio_list


def _majority_vote(model_outputs: List[str], data_source=None, ground_truth=None) -> tuple[str, float]:
    assert len(model_outputs) > 0
    model_answers = [
        _extract_ttrl_answer(generated_text, data_source=data_source, ground_truth=ground_truth)
        for generated_text in model_outputs
    ]
    model_answers = [answer for answer in model_answers if answer is not None]
    if len(model_answers) == 0:
        return "None", 0.0

    clusters = _cluster_answer_indices(model_answers, data_source=data_source)
    if not clusters:
        return "None", 0.0
    clusters = sorted(clusters, key=lambda item: (-len(item[1]), int(item[1][0]) if item[1] else len(model_outputs)))
    majority_answer, majority_indices = clusters[0]
    majority_ratio = len(majority_indices) / len(model_outputs)

    return majority_answer, majority_ratio


def _to_prompt_level_non_tensor_array(values, num_prompts, n):
    arr = np.asarray(values)
    if arr.size == num_prompts:
        return arr
    if arr.size == num_prompts * n:
        return arr.reshape(num_prompts, n)[:, 0]
    return arr[:num_prompts]


# === Metrics Computation ===
def compute_ttrl_metrics(batch, n):
    """
    Compute the TTRL metrics.
    """
    assert len(batch) % n == 0, "batch length must be divisible by n"
    num_prompts = len(batch) // n

    # Sort the batch by the ID
    idx = sorted(range(len(batch)), key=lambda x: batch[x].non_tensor_batch["extra_info"]["index"])

    majority_reward = []
    gt_reward = []
    majority_label = []
    gt_label = []
    data_source = []

    for i in range(len(batch)):
        data_item = batch[idx[i]]
        item_data_source = _get_item_data_source(data_item)
        majority_gt_value = data_item.non_tensor_batch["reward_model"]["majority_gt"]
        original_gt_value = data_item.non_tensor_batch["reward_model"]["original_gt"]
        if _is_kk_data_source(item_data_source):
            majority_gt_value = _canonicalize_kk_value(majority_gt_value, ground_truth=original_gt_value)
            original_gt_value = _canonicalize_kk_value(original_gt_value)
        majority_reward.append(data_item.batch["token_level_scores"].sum().item())
        gt_reward.append(data_item.batch["token_level_scores_original"].sum().item())
        majority_label.append(majority_gt_value)
        gt_label.append(original_gt_value)
        data_source.append(item_data_source)

    ttrl_metrics = _batch_compute_ttrl_metrics(majority_reward, gt_reward, majority_label, gt_label, n=n, data_source=data_source)
    majority_ratio_list = _to_prompt_level_non_tensor_array(batch.non_tensor_batch["majority_ratio_list"], num_prompts, n)
    majority_ratio = float(np.mean(majority_ratio_list))
    ttrl_metrics["majority_ratio"] = majority_ratio
    vote_count_list = np.rint(np.clip(majority_ratio_list.astype(float), 0.0, 1.0) * n).astype(int)

    if "use_ground_truth_mask" in batch.non_tensor_batch:
        use_ground_truth_mask = _to_prompt_level_non_tensor_array(batch.non_tensor_batch["use_ground_truth_mask"], num_prompts, n)
        ttrl_metrics["use_ground_truth_ratio"] = float(np.mean(use_ground_truth_mask.astype(float)))

    return ttrl_metrics


def _batch_compute_ttrl_metrics(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: List[str],
    gt_label: List[str],
    n: int,
    data_source: Optional[Sequence[Optional[str]]] = None,
):
    """
    Compute the TTRL metrics for batch inputs.
    """
    assert len(majority_reward) == len(gt_reward) == len(majority_label) == len(gt_label)
    assert len(majority_reward) % n == 0
    n_prompts = len(majority_reward) // n
    ttrl_metrics = []
    for i in range(n_prompts):
        prompt_majority_reward = majority_reward[i * n:(i + 1) * n]
        prompt_gt_reward = gt_reward[i * n:(i + 1) * n]
        prompt_majority_label = majority_label[i * n:(i + 1) * n]
        prompt_gt_label = gt_label[i * n:(i + 1) * n]

        assert Counter(prompt_majority_label).most_common(1)[0][1] == n
        assert Counter(prompt_gt_label).most_common(1)[0][1] == n

        prompt_majority_label = prompt_majority_label[0]
        prompt_gt_label = prompt_gt_label[0]

        prompt_data_source = data_source[i * n] if data_source is not None and i * n < len(data_source) else None
        ttrl_metric = _prompt_compute_ttrl_metrics(prompt_majority_reward, prompt_gt_reward, prompt_majority_label, prompt_gt_label, data_source=prompt_data_source)
        ttrl_metrics.append(ttrl_metric)

    # Compute the average metrics
    ttrl_metrics = {k: sum(d[k] for d in ttrl_metrics) / len(ttrl_metrics) for k in ttrl_metrics[0]}

    return ttrl_metrics

def _prompt_compute_ttrl_metrics(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: str,
    gt_label: str,
    data_source=None,
    ):    
    assert len(majority_reward) == len(gt_reward)

    if _is_kk_data_source(data_source):
        hit_rate = _compute_binary_reward(majority_label, gt_label, data_source=data_source)
    else:
        hit_rate = 1.0 if grade(majority_label, gt_label) else 0.0
    rewards_hit_rate = 0
    for estimate_reward, true_reward in zip(majority_reward, gt_reward):
        if estimate_reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(majority_reward)
    
    ttrl_metric = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "majority_voting_reward": sum(majority_reward) / len(majority_reward),
        "ground_truth_reward": sum(gt_reward) / len(gt_reward),
        f"pass@{len(majority_reward)}": 1.0 if sum(gt_reward) >= 1 else 0.0,
    }
    return ttrl_metric


# =============================================================================
# Unsupervised RLVR Extensions
# =============================================================================


def apply_hybrid_gt(batch, reward_extra_infos_dict, gen_batch_output, n, tokenizer):
    """
    Apply the hybrid ground truth to the batch.
    Only replaces GT with majority vote when no response in the group is correct
    (according to the reward function), but at least one response has correct format.
    """
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"

    # Get the majority vote ground truth
    model_outputs = []
    for i in range(num_prompts):
        start = i * n
        for j in range(n):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            model_outputs.append(response_str)

    data_sources = _get_prompt_data_sources(batch, num_prompts)
    ground_truths = [batch[i].non_tensor_batch["reward_model"]["ground_truth"] for i in range(num_prompts)]
    majority_gt_list, majority_ratio_list = _batch_majority_vote(
        model_outputs,
        n,
        data_sources=data_sources,
        ground_truths=ground_truths,
    )

    assert len(batch) == len(majority_gt_list), "batch length must be equal to the number of model outputs"

    # Apply the hybrid ground truth
    for i in range(num_prompts):
        data_item = batch[i]

        if "acc" in reward_extra_infos_dict:
            acc_list = reward_extra_infos_dict["acc"][n * i : n * (i + 1)]
        else:
            acc_list = reward_extra_infos_dict["score"][n * i : n * (i + 1)]
        if "format" in reward_extra_infos_dict:
            format_list = reward_extra_infos_dict["format"][n * i : n * (i + 1)]
        else:
            format_list = [1.0] * n

        if True in acc_list:
            continue
        elif sum(format_list) > 0:
            original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt_list[i]
            data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt_list[i]
            data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    return batch


# === Certainty-Based Reward ===
def compute_certainty_metrics(batch, n):
    """
    Calculates the point-biserial correlation and a custom label accuracy metric
    for certainty-based unsupervised rewards.

    Args:
        batch: An object containing the model's outputs with 'non_tensor_batch'
               containing "pseudo_score" and "score".
        n (int): The number of samples per prompt.
    """
    from scipy.stats import pointbiserialr

    proxy_rewards = np.array(batch.non_tensor_batch["pseudo_score"])
    ground_truth_rewards = np.array(batch.non_tensor_batch["score"])

    # 1. Calculate the point-biserial correlation coefficient
    if len(np.unique(ground_truth_rewards)) > 1:
        corr, _ = pointbiserialr(ground_truth_rewards, proxy_rewards)
    else:
        corr = float(0)

    # 2. Reshape the data to (number_of_prompts, n)
    assert len(proxy_rewards) % n == 0, "proxy_rewards length must be divisible by n"
    num_prompts = len(proxy_rewards) // n
    proxy_rewards_reshaped = proxy_rewards.reshape(num_prompts, n)
    ground_truth_rewards_reshaped = ground_truth_rewards.reshape(num_prompts, n)

    # 3. For each prompt, find the index of the response with the largest proxy reward.
    selected_indices = np.argmax(proxy_rewards_reshaped, axis=1)

    # 4. Select the ground truth reward of the chosen response for each prompt.
    rewards_of_selected_responses = ground_truth_rewards_reshaped[np.arange(num_prompts), selected_indices]

    # 5. Calculate the proportion (reward accuracy).
    correctly_identified_count = np.sum(rewards_of_selected_responses)
    label_acc = correctly_identified_count / num_prompts

    return {
        "point_biserial_correlation": corr,
        "pseudo_label_acc": label_acc,
        "train_acc": batch.batch["token_level_scores_original"].sum(dim=-1).float().mean().item(),
    }


def compute_certainty_reward(data, reward_type):
    """
    Compute reward based on model's certainty metrics.

    Args:
        data: DataProto containing model outputs with entropys, self_certaintys, old_log_probs.
        reward_type: One of "self_certainty", "token_level_entropy",
                     "trajectory_level_entropy", "probability".

    Returns:
        reward_tensor, reward_extra_info dict
    """
    reward_extra_info = defaultdict(list)
    response_mask = data.batch["response_mask"]
    from verl.utils.torch_functional import masked_mean, masked_sum

    if reward_type == "self_certainty":
        token_scores = data.batch.get("self_certaintys", torch.zeros_like(response_mask, device=response_mask.device))
        scores = masked_mean(token_scores, response_mask, axis=-1)
    elif reward_type == "token_level_entropy":
        token_scores = -data.batch.get("entropys", torch.zeros_like(response_mask, device=response_mask.device))
        scores = masked_mean(token_scores, response_mask, axis=-1)
    elif reward_type == "trajectory_level_entropy":
        token_scores = data.batch.get("old_log_probs", torch.zeros_like(response_mask, device=response_mask.device))
        scores = masked_mean(token_scores, response_mask, axis=-1)
    elif reward_type == "probability":
        log_probs = data.batch.get("old_log_probs", torch.zeros_like(response_mask, device=response_mask.device))
        sentence_scores = masked_sum(log_probs, response_mask, axis=-1)  # (batch_size,)
        scores = torch.exp(sentence_scores)  # Convert log probabilities to probabilities
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")

    reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
    response_lengths = response_mask.sum(dim=-1).long()
    eos_indices = response_lengths - 1
    reward_tensor.scatter_(-1, eos_indices.unsqueeze(-1), scores.unsqueeze(-1))

    # save the pseudo scores for later calculation
    reward_extra_info['pseudo_score'] = scores

    return reward_tensor, reward_extra_info


# === Self-Verify ===
def compute_self_verify_metrics(batch):
    """Compute metrics for self-verify based reward."""
    proxy_rewards = np.array(batch.non_tensor_batch["verification_score"])
    ground_truth_rewards = np.array(batch.non_tensor_batch["score"])

    rewards_hit_rate = 0
    for estimate_reward, true_reward in zip(proxy_rewards, ground_truth_rewards):
        if estimate_reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(proxy_rewards)

    return {
        "reward_accuracy": rewards_hit_rate,
        "self_verify_reward": sum(proxy_rewards) / len(proxy_rewards),
        "ground_truth_reward": sum(ground_truth_rewards) / len(ground_truth_rewards),
    }


def apply_self_verify(batch, tokenizer, actor_rollout_wg, verify_prompt=None):
    """
    Apply self-verify ground truth to the batch using actor model for verification.
    Returns:
        reward_tensor, reward_extra_infos_dict
    """
    verify_prompt = '''You are given a question and its proposed solution. Your task is to EVALUATE whether the solution is correct.

Follow these steps carefully:
1. The expression in the solution only contains numbers that appear in the question.
2. Every number that appears in the question is used exactly once in the solution.
3. The solution is a valid arithmetic expression (not an equation).
4. The solution evaluates to the target value specified in the question.
5. At the end, output ONLY one of the following with your explanation:
- \\boxed{{True}}  (if the solution is correct)
- \\boxed{{False}} (if the solution is incorrect)

Question:
[{}]

Solution:
[{}]

Result:    
'''
    reward_tensor, reward_extra_infos_dict = _compute_self_verify_rewards(
        batch, tokenizer, actor_rollout_wg, verify_prompt, num_examine=5, reward_fn_key="data_source"
    )
    return reward_tensor, reward_extra_infos_dict


def _compute_self_verify_rewards(data, tokenizer, actor_rollout_wg, verify_prompt, num_examine=5, reward_fn_key="data_source"):
    """
    Compute self-verify rewards using actor model for verification.

    Args:
        data: DataProto containing prompts and responses
        tokenizer: Tokenizer for text processing
        actor_rollout_wg: Actor rollout worker group for self-verification
        verify_prompt: Template string for verification prompts
        num_examine: Number of samples to print for debugging
        reward_fn_key: Key for accessing data source

    Returns:
        tuple: (reward_tensor, reward_extra_infos_dict)
    """
    # If there is rm score, we directly return rm score
    if "rm_scores" in data.batch.keys():
        return {"reward_tensor": data.batch["rm_scores"]}, {}

    reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
    reward_extra_info = defaultdict(list)

    # Collect questions and solutions for batch processing
    questions = []
    solutions = []
    item_indices = []

    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem

        prompt_ids = data_item.batch["prompts"]
        response_ids = data_item.batch["responses"]
        # decode
        prompt_str = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        response_str = tokenizer.decode(response_ids, skip_special_tokens=True)

        # Use ground truth as question if available, otherwise use prompt
        question = prompt_str[(prompt_str.find('user\n') + len('user\n')):prompt_str.find('assistant\n')].strip()
        questions.append(question)
        solutions.append(response_str)
        item_indices.append(i)

    # Batch verification using actor
    if questions:
        verification_batch = _create_verification_batch(questions, solutions, tokenizer, verify_prompt)
        verification_batch.meta_info = {
            "kwargs": {
                "max_tokens": 4096,
                "n": 1,
                "temperature": 0.5,
            }
        }
        # Generate verification responses using actor
        verification_output = actor_rollout_wg.generate_sequences(verification_batch)

        # Parse verification responses
        verification_responses = verification_output.batch["responses"]
        for i, (item_idx, verification_response_ids) in enumerate(zip(item_indices, verification_responses)):
            # Decode verification response
            valid_verification_length = verification_output.batch["attention_mask"][i].sum()
            valid_verification_ids = verification_response_ids[:valid_verification_length]
            verification_text = tokenizer.decode(valid_verification_ids, skip_special_tokens=True)

            # Parse score
            score = _parse_verification_response(verification_text)

            # Get original data item for logging
            data_item = data[item_idx]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()

            # Store reward
            reward_tensor[item_idx, valid_response_length - 1] = score

            # Store extra info
            reward_extra_info["verification_response"].append(verification_text)
            reward_extra_info["verification_score"].append(score)

    return reward_tensor, reward_extra_info


def _create_verification_batch(questions, solutions, tokenizer, prompt):
    """Create a DataProto batch of verification prompts."""
    PROMPT = prompt

    verification_prompts = []
    for q, s in zip(questions, solutions):
        q_escaped = str(q).replace("{", "{{").replace("}", "}}")
        s_escaped = str(s).replace("{", "{{").replace("}", "}}")
        message = [{"role": "user", "content": PROMPT.format(q_escaped, s_escaped)}]
        verification_prompts.append(message)

    # First format the chat templates into strings
    formatted_prompts = []
    for msg in verification_prompts:
        formatted = tokenizer.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted_prompts.append(formatted)
    # Then tokenize the formatted strings
    tokenizer.padding_side = "left"
    tokenized = tokenizer(
        formatted_prompts,
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="pt",
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # Construct position_ids
    position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

    batch = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=len(verification_prompts),
    )

    return DataProto(batch=batch)


def _parse_verification_response(response):
    """
    Parse the verification response to get a score.

    Args:
        response: Verification response text

    Returns:
        float: 1.0 if verified as correct, 0.0 otherwise
    """
    response = response.strip().lower()

    # Look for boxed answers
    if "\\boxed{true}" in response or "\\boxed{true" in response:
        return 1.0
    elif "\\boxed{false}" in response or "\\boxed{false" in response:
        return 0.0
    # Fallback to keyword matching
    elif "true" in response and "false" not in response:
        return 1.0
    elif "false" in response and "true" not in response:
        return 0.0
    else:
        return 0.0
