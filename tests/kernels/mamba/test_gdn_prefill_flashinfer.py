# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

if current_platform.is_rocm():
    pytest.skip(
        reason="FlashInfer GDN prefill is not supported on ROCm.",
        allow_module_level=True,
    )

import flashinfer.gdn_prefill  # noqa: E402

from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn  # noqa: E402
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    fi_chunk_gated_delta_rule,
)  # noqa: E402
from vllm.third_party.flash_linear_attention.ops import (  # noqa: E402
    chunk_gated_delta_rule,
)
from vllm.utils.flashinfer import has_flashinfer_cudnn_gdn  # noqa: E402


@pytest.mark.parametrize("backend", ["flashinfer", "cudnn"])
@pytest.mark.parametrize("output_final_state", [False, True])
def test_flashinfer_gdn_prefill_preserves_state_and_gate_contract(
    monkeypatch, backend, output_final_state
):
    captured = {}

    def fake_chunk_gated_delta_rule(**kwargs):
        captured.update(kwargs)
        if kwargs["output_final_state"]:
            return kwargs["q"], kwargs["initial_state"].clone()
        return kwargs["q"]

    if backend == "cudnn":
        monkeypatch.setattr(
            qwen_gdn_linear_attn, "flashinfer_cudnn_gdn", fake_chunk_gated_delta_rule
        )
    else:
        monkeypatch.setattr(
            flashinfer.gdn_prefill,
            "chunk_gated_delta_rule",
            fake_chunk_gated_delta_rule,
        )
    q = torch.zeros(1, 2, 1, 2, dtype=torch.bfloat16)
    g = torch.tensor([[[-1.0], [-0.5]]])
    beta = torch.tensor([[[0.25], [0.75]]], dtype=torch.bfloat16)
    state = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)

    output, final_state = fi_chunk_gated_delta_rule(
        q=q,
        k=q,
        v=q,
        g=g,
        beta=beta,
        initial_state=state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        backend=backend,
    )

    assert captured["cu_seqlens"].dtype == torch.int64
    torch.testing.assert_close(captured["g"], g.squeeze(0).exp())
    torch.testing.assert_close(captured["beta"], beta.squeeze(0).float())
    torch.testing.assert_close(captured["initial_state"], state.float())
    assert output.shape == q.shape
    if output_final_state:
        torch.testing.assert_close(final_state, state.float())
    else:
        assert final_state is None


@pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and (
            current_platform.is_device_capability(100)
            or current_platform.is_device_capability(103)
        )
    ),
    reason="cuDNN GDN prefill requires CUDA SM100/SM103",
)
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_v_heads,normalize_qk", [(2, True), (4, False)])
@torch.inference_mode()
def test_cudnn_gdn_prefill_matches_fla_with_graph_and_initial_state(
    state_dtype: torch.dtype, num_v_heads: int, normalize_qk: bool
):
    if not has_flashinfer_cudnn_gdn():
        pytest.skip("FlashInfer cuDNN GDN prefill is unavailable")

    torch.manual_seed(42)
    num_tokens, num_k_heads, head_dim = 145, 2, 128
    q, k = [
        torch.randn(
            1, num_tokens, num_k_heads, head_dim, device="cuda", dtype=torch.bfloat16
        )
        for _ in range(2)
    ]
    if not normalize_qk:
        q, k = [
            torch.nn.functional.normalize(x.float(), dim=-1).to(x.dtype) for x in (q, k)
        ]
    v = torch.randn(
        1, num_tokens, num_v_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    g = -torch.rand(1, num_tokens, num_v_heads, device="cuda")
    beta = torch.rand_like(g)
    initial_state = (
        torch.randn(
            2, num_v_heads, head_dim, head_dim, device="cuda", dtype=state_dtype
        )
        * 0.05
    )
    state_before = initial_state.clone()
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, 17, num_tokens], device="cuda", dtype=torch.int32),
        use_qk_l2norm_in_kernel=normalize_qk,
    )
    expected_out, expected_state = chunk_gated_delta_rule(**kwargs)
    actual_out, actual_state = fi_chunk_gated_delta_rule(**kwargs, backend="cudnn")
    torch.testing.assert_close(actual_out, expected_out, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(actual_state, expected_state, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(initial_state, state_before, atol=0, rtol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out, graph_state = fi_chunk_gated_delta_rule(**kwargs, backend="cudnn")
    graph.replay()
    torch.testing.assert_close(graph_out, actual_out, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(graph_state, actual_state, atol=2e-2, rtol=2e-2)
