# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare cuDNN GDN/KDA prefill with vLLM's existing kernels."""

import argparse
import functools
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch
from vllm.triton_utils import triton


def make_inputs(args, tokens, heads):
    torch.manual_seed(0)
    value_heads = heads * args.value_head_ratio if args.family == "gdn" else heads
    q, k = [
        torch.randn(1, tokens, heads, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    v, raw_g = [
        torch.randn(1, tokens, value_heads, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    state = torch.randn(args.sequences, value_heads, 128, 128, device="cuda") * 0.05
    offsets = torch.arange(args.sequences + 1, device="cuda", dtype=torch.int64)
    offsets *= tokens // args.sequences
    return dict(q=q, k=k, v=v, initial_state=state, cu_seqlens=offsets), raw_g


def gdn_calls(inputs, raw_g):
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        fi_chunk_gated_delta_rule,
    )
    from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule
    from vllm.third_party.flash_linear_attention.ops.chunk import l2norm_fwd

    # Qwen normalizes Q/K during post-convolution preparation.
    inputs = {**inputs, "q": l2norm_fwd(inputs["q"]), "k": l2norm_fwd(inputs["k"])}

    kwargs = dict(
        **inputs,
        g=-raw_g[..., 0].float().sigmoid(),
        beta=raw_g[..., 1].float().sigmoid(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    return {
        "triton": functools.partial(chunk_gated_delta_rule, **kwargs),
        "flashinfer": functools.partial(fi_chunk_gated_delta_rule, **kwargs),
        "flashinfer_cudnn": functools.partial(
            fi_chunk_gated_delta_rule, **kwargs, backend="cudnn"
        ),
    }


def kda_calls(inputs, raw_g, lower_bound):
    from vllm.model_executor.layers.mamba.ops.flashinfer_cudnn_kda import (
        flashinfer_cudnn_kda_prefill,
    )
    from vllm.models.kimi_k3.nvidia.kda import _flashkda_prefill
    from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
        chunk_kda_with_fused_gate,
    )

    _, tokens, heads, _ = raw_g.shape
    a_log = torch.zeros(heads, device="cuda")
    dt_bias = torch.zeros(heads, 128, device="cuda")
    raw_beta = raw_g[..., 0].contiguous()
    kwargs = dict(
        **inputs,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=a_log,
        lower_bound=lower_bound,
    )
    calls = {
        "triton": functools.partial(
            chunk_kda_with_fused_gate,
            **kwargs,
            g_bias=dt_bias,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            out=torch.empty_like(inputs["v"]),
        ),
        "flashinfer_cudnn": functools.partial(
            flashinfer_cudnn_kda_prefill,
            **kwargs,
            dt_bias=dt_bias,
            out=torch.empty_like(inputs["v"]),
        ),
    }
    if lower_bound is not None:
        import vllm._flashkda_C  # noqa: F401

        workspace = torch.empty(
            torch.ops._flashkda_C.get_workspace_size(
                tokens, heads, inputs["initial_state"].shape[0]
            ),
            device="cuda",
            dtype=torch.uint8,
        )
        calls["flashkda"] = functools.partial(
            _flashkda_prefill,
            **inputs,
            g=raw_g,
            beta=raw_beta,
            A_log=a_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            out=torch.empty_like(inputs["v"]),
            final_state=torch.empty_like(inputs["initial_state"]),
            workspace=workspace,
        )
    return calls


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["gdn", "kda"], required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[128, 2048, 8192])
    parser.add_argument("--heads", type=int, nargs="+", default=[12, 32, 64, 96])
    parser.add_argument(
        "--value-head-ratio", type=int, default=2, help="V heads per Q/K head for GDN"
    )
    parser.add_argument("--sequences", type=int, default=1)
    parser.add_argument("--unbounded-gate", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", choices=["eager", "cuda_graph"], default="eager")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.unbounded_gate:
        parser.error("cuDNN KDA requires a bounded gate; unbounded gates are unsafe")
    if args.repeats < 1 or args.sequences < 1:
        parser.error("repeats and sequences must be positive")
    if any(t < args.sequences or t % args.sequences for t in args.tokens):
        parser.error("each token count must be a positive multiple of sequences")
    if any(h < 1 for h in args.heads):
        parser.error("head counts must be positive")
    if args.value_head_ratio < 1:
        parser.error("value-head-ratio must be positive")

    report = {
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "cuda": torch.version.cuda,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "vllm", "flashinfer-python", "nvidia-cudnn-frontend"]
        },
        "arguments": {**vars(args), "output": str(args.output)},
        "results": [],
    }
    if report["versions"]["nvidia-cudnn-frontend"] != "1.30.0":
        raise RuntimeError("This comparison requires cudnn-frontend 1.30.0")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for tokens in args.tokens:
        for heads in args.heads:
            inputs, raw_g = make_inputs(args, tokens, heads)
            initial_state = inputs["initial_state"].clone()
            calls = (
                gdn_calls(inputs, raw_g)
                if args.family == "gdn"
                else kda_calls(inputs, raw_g, None if args.unbounded_gate else -5.0)
            )
            expected = calls["triton"]()
            for backend, call in calls.items():
                actual = call()
                torch.testing.assert_close(
                    inputs["initial_state"], initial_state, atol=0, rtol=0
                )
                for result, reference in zip(actual, expected):
                    torch.testing.assert_close(result, reference, atol=0.02, rtol=0.02)
                errors = [
                    (result.float() - reference.float()).abs().max().item()
                    for result, reference in zip(actual, expected)
                ]
                samples = []
                for _ in range(args.repeats):
                    if args.mode == "cuda_graph":
                        ms = triton.testing.do_bench_cudagraph(
                            call, rep=200, return_mode="median"
                        )
                    else:
                        ms = triton.testing.do_bench(
                            call, warmup=100, rep=200, return_mode="median"
                        )
                    samples.append(ms * 1000)
                torch.testing.assert_close(
                    inputs["initial_state"], initial_state, atol=0, rtol=0
                )
                row = {
                    "tokens": tokens,
                    "heads": heads,
                    "value_heads": inputs["v"].shape[2],
                    "backend": backend,
                    "samples_us": samples,
                    "median_us": statistics.median(samples),
                    "min_us": min(samples),
                    "max_us": max(samples),
                    "max_abs_error_output_state": errors,
                }
                report["results"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
