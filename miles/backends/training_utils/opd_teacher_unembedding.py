"""Local replica of the disaggregated OPD teacher's final norm + unembedding.

The teacher mesh only ever sends hidden states (see `get_hidden_states` in
`logit_processors.py` and `send_teacher_hidden_states` in `megatron_utils/actor.py`);
this module lets the student mesh reconstruct full-vocab teacher log-probs locally
from those hidden states, using a one-time (not per-step) read of just two tensors
out of the teacher's checkpoint.
"""

from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed.checkpoint as dist_cp
import torch.nn.functional as F

_OUTPUT_LAYER_KEY = "module.module.output_layer.weight"
_FINAL_NORM_KEY = "module.module.decoder.final_layernorm.weight"


def _resolve_teacher_checkpoint_dir(checkpoint_root: str, ckpt_step: int | None) -> Path:
    root = Path(checkpoint_root)
    if ckpt_step is None:
        ckpt_step = int((root / "latest_checkpointed_iteration.txt").read_text().strip())
    return root / f"iter_{ckpt_step:07d}"


def load_teacher_output_layer(
    checkpoint_root: str,
    ckpt_step: int | None,
    *,
    hidden_size: int,
    vocab_size: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read only `output_layer.weight` and the final norm weight out of a Megatron
    torch_dist checkpoint, without loading the rest of the (possibly huge) model."""
    checkpoint_dir = _resolve_teacher_checkpoint_dir(checkpoint_root, ckpt_step)
    state_dict = {
        _OUTPUT_LAYER_KEY: torch.empty((vocab_size, hidden_size), dtype=dtype),
        _FINAL_NORM_KEY: torch.empty((hidden_size,), dtype=dtype),
    }
    dist_cp.load(state_dict, storage_reader=dist_cp.FileSystemReader(str(checkpoint_dir)))
    return state_dict[_OUTPUT_LAYER_KEY], state_dict[_FINAL_NORM_KEY]


class TeacherUnembedding(torch.nn.Module):
    """Applies the teacher's final RMSNorm + unembedding to received hidden states.

    Requires `--tensor-model-parallel-size=1` on the student (see the validation in
    `miles.utils.arguments`): this module holds the full, unsharded vocab weight, and
    reconciling that against vocab-parallel-sharded student logits isn't implemented.
    """

    def __init__(self, output_layer_weight: torch.Tensor, final_norm_weight: torch.Tensor, eps: float) -> None:
        super().__init__()
        self.register_buffer("output_layer_weight", output_layer_weight, persistent=False)
        self.register_buffer("final_norm_weight", final_norm_weight, persistent=False)
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """`[R, H]` hidden states -> `[R, V]` teacher full-vocab log-probs (float32)."""
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        normed = hidden_states * torch.rsqrt(variance + self.eps) * self.final_norm_weight.to(torch.float32)
        logits = F.linear(normed, self.output_layer_weight.to(torch.float32))
        return torch.log_softmax(logits, dim=-1)


def build_teacher_unembedding(args: Namespace, *, device: torch.device) -> TeacherUnembedding:
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    output_layer_weight, final_norm_weight = load_teacher_output_layer(
        args.opd_teacher_load,
        args.opd_teacher_ckpt_step,
        hidden_size=args.hidden_size,
        vocab_size=args.padded_vocab_size,
        dtype=dtype,
    )
    return TeacherUnembedding(
        output_layer_weight.to(device=device),
        final_norm_weight.to(device=device),
        eps=args.norm_epsilon,
    ).to(device=device)
