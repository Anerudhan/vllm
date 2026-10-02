# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.chunk import l2norm_fwd
from vllm.utils.flashinfer import (
    flashinfer_recurrent_kda,
    has_flashinfer_cudnn_kda,
)

logger = init_logger(__name__)


def _validate_gate(lower_bound: float | None) -> float:
    if lower_bound is None or not -5.0 <= lower_bound < 0:
        raise ValueError(
            "FlashInfer cuDNN KDA prefill requires a bounded gate with "
            "-5 <= lower_bound < 0. Unbounded gates are unsupported."
        )
    return lower_bound


def validate_flashinfer_cudnn_kda_prefill(
    head_dim: int,
    input_dtype: torch.dtype,
    state_dtype: torch.dtype,
    lower_bound: float | None,
) -> None:
    _validate_gate(lower_bound)
    capability = (
        current_platform.get_device_capability() if current_platform.is_cuda() else None
    )
    if not (
        capability is not None
        and (capability.major, capability.minor) in ((10, 0), (10, 3))
        and head_dim == 128
        and input_dtype == torch.bfloat16
        and state_dtype in (torch.bfloat16, torch.float32)
        and has_flashinfer_cudnn_kda()
    ):
        raise RuntimeError(
            "FlashInfer cuDNN KDA prefill requires SM100 or SM103, bfloat16 "
            "input, float32 or bfloat16 state, head_dim=128, FlashInfer's "
            "cuDNN KDA API, and cudnn-frontend >= 1.30.0."
        )
    logger.info_once("Using FlashInfer cuDNN KDA prefill backend.")


def flashinfer_cudnn_kda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    lower_bound = _validate_gate(lower_bound)
    if envs.VLLM_BATCH_INVARIANT:
        raise NotImplementedError(
            "FlashInfer's KDA dispatcher does not expose batch_invariant."
        )
    # cuDNN's fused normalization uses a different epsilon.
    q = l2norm_fwd(q.contiguous())
    k = l2norm_fwd(k.contiguous())
    A_log = A_log.reshape(-1)
    dt_bias = dt_bias.reshape(-1, q.shape[-1])
    # The public dispatcher updates its input state in place.
    final_state = initial_state.clone()
    output, _ = flashinfer_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=raw_g,
        beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=q.shape[-1] ** -0.5,
        initial_state=final_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=True,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        beta_is_logit=True,
        output=out,
        backend="cudnn",
    )
    return output, final_state
