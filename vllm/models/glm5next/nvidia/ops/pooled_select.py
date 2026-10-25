# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional cuDNN pooled selection from native DeepGEMM logits."""

import cuda.bindings.driver as cuda
import torch

from vllm.v1.worker.workspace import current_workspace_manager


def pooled_select(
    logits: torch.Tensor,
    positions: torch.Tensor,
    output: torch.Tensor,
    pool_size: int,
    selected_pools: int,
    max_pools: int,
    mode: str,
) -> None:
    try:
        from cudnn.deepseek_sparse_attention.pooled.forward import (
            _compile_pooled_select,
        )
    except ImportError as exc:
        raise RuntimeError(
            "VLLM_GLM5NEXT_USE_CUDNN_KPOOL requires the pooled DSA "
            "cuDNN Frontend additions and CuTe DSL 4.8.0."
        ) from exc

    assert logits.shape[1] <= max_pools and logits.stride(1) == 1
    kernel = _compile_pooled_select(
        logits.device.index,
        torch.cuda.get_device_capability(logits.device),
        max_pools,
        selected_pools,
        pool_size,
        positions.dtype,
        mode,
        logits.dtype,
    )
    bucket = 1 << (max_pools - 1).bit_length()
    # These BF16 variants keep all candidates in shared memory.
    shape = (
        (1, 2, 4)
        if logits.dtype == torch.bfloat16 and 8192 <= bucket <= 32768
        else (logits.shape[0], 2, logits.shape[1])
    )
    (scratch,) = current_workspace_manager().get_simultaneous((shape, torch.int32))
    kernel(
        logits,
        scratch,
        positions,
        output,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
