# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.chunk import l2norm_fwd
from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import flashinfer_recurrent_kda

logger = init_logger(__name__)


@triton.jit
def _normalize_qk_kernel(
    Q,
    K,
    Output,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    QB: tl.constexpr,
    QT: tl.constexpr,
    QH: tl.constexpr,
    QD: tl.constexpr,
    KB: tl.constexpr,
    KT: tl.constexpr,
    KH: tl.constexpr,
    KD: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    cols = tl.arange(0, BLOCK_D)
    batch, token, head = rows // (T * H), rows // H % T, rows % H
    mask = (rows[:, None] < ROWS) & (cols[None, :] < D)
    if tl.program_id(1) == 0:
        x = tl.load(
            Q
            + batch[:, None] * QB
            + token[:, None] * QT
            + head[:, None] * QH
            + cols[None, :] * QD,
            mask,
            other=0,
        ).to(tl.float32)
    else:
        x = tl.load(
            K
            + batch[:, None] * KB
            + token[:, None] * KT
            + head[:, None] * KH
            + cols[None, :] * KD,
            mask,
            other=0,
        ).to(tl.float32)
    norm = x * tl.rsqrt(tl.sum(x * x, axis=1)[:, None] + 1e-6)
    tl.store(
        Output + tl.program_id(1) * ROWS * D + rows[:, None] * D + cols[None, :],
        norm,
        mask,
    )


def _normalize_qk(q: torch.Tensor, k: torch.Tensor):
    from vllm.third_party.flash_linear_attention.ops.l2norm import USE_DEFAULT_FLA_NORM

    if (
        USE_DEFAULT_FLA_NORM
        or q.ndim != 4
        or q.shape != k.shape
        or q.dtype != k.dtype
        or q.device != k.device
        or q.shape[-1] != 128
    ):
        return l2norm_fwd(q.contiguous()), l2norm_fwd(k.contiguous())
    b, t, h, d = q.shape
    normalized = torch.empty((2, b, t, h, d), dtype=q.dtype, device=q.device)
    _normalize_qk_kernel[(triton.cdiv(b * t * h, 32), 2)](
        q,
        k,
        normalized,
        t,
        h,
        d,
        *q.stride(),
        *k.stride(),
        b * t * h,
        triton.next_power_of_2(d),
    )
    return normalized.unbind(0)


def _validate_gate(lower_bound: float | None, backend: str) -> float:
    if lower_bound is None:
        raise ValueError("FlashInfer KDA prefill requires a bounded gate.")
    if backend == "cudnn" and not -5.0 <= lower_bound < 0:
        raise ValueError(
            "FlashInfer cuDNN KDA prefill requires a bounded gate with "
            "-5 <= lower_bound < 0. Unbounded gates are unsupported."
        )
    return lower_bound


def validate_flashinfer_kda_prefill(
    head_dim: int,
    input_dtype: torch.dtype,
    state_dtype: torch.dtype,
    lower_bound: float | None,
    backend: str = "auto",
) -> None:
    if backend not in ("auto", "cudnn"):
        raise ValueError(f"Unsupported FlashInfer KDA backend: {backend}")
    _validate_gate(lower_bound, backend)
    capability = (
        current_platform.get_device_capability() if current_platform.is_cuda() else None
    )
    if not (
        capability is not None
        and (capability.major, capability.minor) in ((10, 0), (10, 3))
        and head_dim == 128
        and input_dtype == torch.bfloat16
        and (
            state_dtype in (torch.bfloat16, torch.float32)
            if backend == "cudnn"
            else state_dtype == torch.bfloat16
        )
    ):
        raise RuntimeError(
            f"FlashInfer KDA prefill (backend={backend}) requires SM100 or SM103, "
            "bfloat16 input, and head_dim=128. Backend auto requires bfloat16 "
            "state; cudnn also supports float32 state."
        )
    logger.info_once("Using FlashInfer KDA prefill with backend=%s.", backend)


def flashinfer_kda_prefill(
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
    backend: str = "auto",
    seq_order: torch.Tensor | None = None,
    prefill_workspace: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if backend not in ("auto", "cudnn"):
        raise ValueError(f"Unsupported FlashInfer KDA backend: {backend}")
    lower_bound = _validate_gate(lower_bound, backend)
    if backend == "cudnn" and envs.VLLM_BATCH_INVARIANT:
        raise NotImplementedError(
            "FlashInfer's KDA dispatcher does not expose batch_invariant."
        )
    if backend == "cudnn":
        # cuDNN's fused normalization uses a different epsilon.
        q, k = _normalize_qk(q, k)
        final_state = torch.empty_like(initial_state)
    else:
        q, k = q.contiguous(), k.contiguous()
        final_state = initial_state.contiguous()
        v, raw_g, raw_beta = v.contiguous(), raw_g.contiguous(), raw_beta.contiguous()
        with torch.inference_mode(False):
            cu_seqlens = cu_seqlens.to(
                torch.int64, copy=cu_seqlens.is_inference()
            ).contiguous()
    A_log = A_log.reshape(-1).contiguous()
    dt_bias = (
        dt_bias.reshape(-1, q.shape[-1]) if backend == "cudnn" else dt_bias.reshape(-1)
    ).contiguous()
    output, _ = flashinfer_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=raw_g,
        beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=q.shape[-1] ** -0.5,
        initial_state=initial_state if backend == "cudnn" else final_state,
        output_state=final_state if backend == "cudnn" else None,
        output_final_state=backend == "cudnn",
        use_qk_l2norm_in_kernel=backend != "cudnn",
        use_gate_in_kernel=True,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens.contiguous(),
        beta_is_logit=True,
        output=out,
        backend=backend,
        seq_order=seq_order,
        prefill_workspace=prefill_workspace,
    )
    return output, final_state
