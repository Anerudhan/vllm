# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional cuDNN pooled selection from native DeepGEMM logits."""

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
        from cudnn.ops.dsa import (
            pooled_select_forward,
            pooled_select_workspace_size,
        )
    except ImportError as exc:
        raise RuntimeError(
            "VLLM_GLM5NEXT_USE_CUDNN_KPOOL requires the pooled DSA "
            "cuDNN Frontend additions and CuTe DSL 4.8.0."
        ) from exc

    workspace_bytes = pooled_select_workspace_size(logits, max_pools=max_pools)
    (workspace,) = current_workspace_manager().get_simultaneous(
        ((workspace_bytes,), torch.uint8)
    )
    pooled_select_forward(
        logits,
        positions,
        selected_pools=selected_pools,
        pool_size=pool_size,
        mode=mode,
        max_pools=max_pools,
        output=output,
        workspace=workspace,
        current_stream=torch.cuda.current_stream().cuda_stream,
    )
