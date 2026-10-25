# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time SparseAttnIndexerKpool.forward_native with real attention metadata.

Run from an installed vLLM environment on SM100 for both FP8 and MXFP4:
  .venv/bin/python benchmark_glm5next_kpool.py --profile --graph --output kpool.json

BF16 has no native vLLM path; cuDNN rows use IndexerForward.execute.
Use --implementations native cudnn for a same-input comparison, or --check-only
to validate without timing. Point PYTHONPATH at the cuDNN Frontend checkout.
Inputs, quantization, metadata, JIT, and validation are outside the timers.
Read modes use prepared caches. Write modes also pool/quantize/insert current K
and update the tail cache (vLLM: FP8 only). Decode has one token per request;
prefill has one request with a causal chunk ending at the given context length.
Fixed inputs are reused, so these are warm-cache microbenchmarks.

--profile saves GPU activities, including device copies, in the kernels field.
--graph measures fixed-input full capture, not vLLM's breakable capture replay.
Neither timing includes final sparse attention or a backward pass.
"""

import argparse
import ast
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import socket
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

import vllm
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.models.glm5next.nvidia import sparse_indexer
from vllm.models.glm5next.nvidia.sparse_indexer import SparseAttnIndexerKpool
from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata, mqa_logits_use_mx_scales
from vllm.v1.attention.backends.mla.indexer import (
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerPrefillChunkMetadata,
    DeepseekV32IndexerPrefillMetadata,
)
from vllm.v1.worker.workspace import init_workspace_manager

HEADS, DIM, POOL, TOPK, PAGE = 32, 128, 4, 2048, 64


def load_native_candidate(folder):
    """Load candidate functions while retaining the installed vLLM runtime."""
    path = folder / "ops" / "kpool_compress.py"
    spec = importlib.util.spec_from_file_location("kpool_prefill_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = folder / "sparse_indexer.py"
    tree = ast.parse(source.read_text())
    names = {"_kpool_compress_insert", "sparse_attn_indexer_kpool"}
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(definitions) == len(names)
    scope = dict(vars(sparse_indexer), kpool_ops=module)
    exec(compile(ast.Module(definitions, type_ignores=[]), str(source), "exec"), scope)
    sparse_indexer.sparse_attn_indexer_kpool = scope["sparse_attn_indexer_kpool"]
    return source


def quantize(x, dtype):
    if dtype == "fp8":
        scale = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448
        scale = scale.log2().ceil().exp2()
        return (x.float() / scale).to(torch.float8_e4m3fn), scale
    # E2M1, RNE, one UE8M0 scale per 32 values; matches the vLLM test reference.
    blocks = x.float().reshape(*x.shape[:-1], DIM // 32, 32)
    exponent = blocks.abs().amax(-1).clamp_min(6 * 2**-126) / 6
    exponent = exponent.log2().ceil().clamp(-127, 127)
    scaled = blocks / exponent.exp2().unsqueeze(-1)
    magnitude = scaled.abs()
    code = torch.zeros_like(magnitude, dtype=torch.uint8)
    for i, threshold in enumerate((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), 1):
        choose = magnitude > threshold if i % 2 else magnitude >= threshold
        code = torch.where(choose, i, code)
    code |= scaled.signbit().to(torch.uint8) << 3
    code = code.reshape(x.shape)
    return code[..., ::2] | (code[..., 1::2] << 4), (exponent + 127).byte()


def dequantize(values, scales, dtype):
    if dtype == "fp8":
        return values.view(torch.float8_e4m3fn).float() * scales
    nibble = torch.stack((values & 15, values >> 4), -1).flatten(-2)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=values.device)
    decoded = levels[(nibble & 7).long()] * torch.where(nibble < 8, 1, -1)
    return decoded * (scales.float() - 127).exp2().repeat_interleave(32, -1)


def build_case(dtype, mode, batch, length, queries, seed, implementation="native"):
    torch.manual_seed(seed)
    prefill, write = mode.startswith("prefill"), mode.endswith("write")
    rows = queries if prefill else batch
    requests = 1 if prefill else batch
    pools = length // POOL
    pages = (pools + PAGE - 1) // PAGE
    capacity = pages * PAGE
    device = "cuda"
    q = torch.randn(rows, HEADS, DIM, device=device, dtype=torch.bfloat16)
    keys = torch.randn(requests, capacity, DIM, device=device, dtype=torch.bfloat16)
    weights = torch.randn(rows, HEADS, device=device) / math.sqrt(HEADS * DIM)
    positions = (
        torch.arange(length - rows, length, device=device, dtype=torch.int64)
        if prefill
        else torch.full((rows,), length - 1, device=device)
    )
    lengths = positions.to(torch.int32) + 1
    ends = lengths // POOL
    output = torch.empty(rows, TOPK + POOL - 1, device=device, dtype=torch.int32)
    if dtype == "bf16" and implementation == "native":
        mask = torch.arange(capacity, device=device)[None, :] >= ends[:, None]
        offsets = torch.arange(POOL, device=device)
        tail_offsets = torch.arange(POOL - 1, device=device)

        def reference():
            dots = (
                torch.matmul(q, keys[0].T)
                if prefill
                else torch.bmm(q, keys.transpose(1, 2))
            )
            scores = (dots.float().relu() * weights.unsqueeze(-1)).sum(1)
            scores.masked_fill_(mask, -float("inf"))
            ids = scores.topk(TOPK // POOL, sorted=False).indices
            output[:, :TOPK] = (ids[..., None] * POOL + offsets).flatten(1)
            tail = ends[:, None] * POOL + tail_offsets
            output[:, TOPK:] = torch.where(tail < lengths[:, None], tail, -1)
            return output

        return SimpleNamespace(
            fn=reference,
            metadata={},
            output=output,
            positions=positions,
            q=q.float(),
            weights=weights,
            get_keys=lambda r: keys[0 if prefill else r].float(),
            rows=rows,
            prefill=prefill,
        )

    if dtype == "bf16":
        q_values, q_scales = q, None
        values, scales = keys, None
    else:
        q_values, q_scales = quantize(q, dtype)
        values, scales = quantize(keys, dtype)
    value_bytes = {"bf16": 2 * DIM, "fp8": DIM, "fp4": DIM // 2}[dtype]
    cache = torch.zeros(
        requests * pages,
        PAGE,
        value_bytes + (0 if dtype == "bf16" else 4),
        device=device,
        dtype=torch.uint8,
    )
    block_table = torch.randperm(requests * pages, device=device).to(torch.int32)
    block_table = block_table.reshape(requests, pages)
    flat = cache.flatten(1)
    physical = block_table.flatten().long()
    flat[physical, : PAGE * value_bytes] = values.view(torch.uint8).reshape(
        requests * pages, PAGE * value_bytes
    )
    native_mx = (
        implementation == "native" and dtype == "fp8" and mqa_logits_use_mx_scales()
    )
    cache_scales = scales
    if native_mx:
        cache_scales = (
            (scales.log2() + 127).byte().expand(*scales.shape[:-1], 4).contiguous()
        )
    if cache_scales is not None:
        flat[physical, PAGE * value_bytes :] = cache_scales.view(torch.uint8).reshape(
            requests * pages, PAGE * 4
        )
    q_input = (
        q_values
        if dtype != "fp4"
        else (q_values, q_scales.view(torch.int32).squeeze(-1))
    )
    native_weights = weights * q_scales.squeeze(-1) if dtype == "fp8" else weights
    raw_k = torch.randn(rows, DIM, device=device, dtype=torch.bfloat16)
    gates = torch.randn_like(raw_k)
    ape = torch.randn(POOL, DIM, device=device, dtype=torch.float32)
    tail = torch.randn(requests, 2, POOL, DIM, device=device, dtype=torch.bfloat16)
    request_ids = (
        torch.zeros(rows, device=device, dtype=torch.int64)
        if prefill
        else (torch.arange(requests, device=device))
    )
    pool_ids = (positions // POOL).clamp_max(pools - 1)
    slots = block_table[request_ids, pool_ids // PAGE].long() * PAGE + pool_ids % PAGE
    slots = torch.where(lengths % POOL == 0, slots, -1)

    def get_keys(row):
        blocks = flat[block_table[0 if prefill else row].long()]
        v = blocks[:, : PAGE * value_bytes].reshape(capacity, value_bytes)
        if dtype == "bf16":
            return v.view(torch.bfloat16).float()
        s = blocks[:, PAGE * value_bytes :].reshape(capacity, 4).contiguous()
        if dtype == "fp8":
            s = (s[:, :1].float() - 127).exp2() if native_mx else s.view(torch.float32)
        return dequantize(v, s, dtype)

    if implementation == "cudnn":
        from cudnn.deepseek_sparse_attention.pooled import IndexerForward

        cu_query = (
            torch.tensor([0, rows], device=device, dtype=torch.int32)
            if prefill
            else torch.arange(requests + 1, device=device, dtype=torch.int32)
        )
        updates = (
            dict(
                keys=raw_k,
                gates=gates,
                ape=ape,
                tail=tail,
                pool_slots=slots,
                tail_slots=request_ids * POOL + positions % POOL,
            )
            if write
            else {}
        )
        inputs = (
            q_values,
            native_weights,
            flat,
            block_table,
            cu_query,
            positions,
            output,
        )
        query_scale = q_input[1] if dtype == "fp4" else None
        plan = IndexerForward(
            *inputs,
            query_scale,
            cache_dtype=dtype,
            mode="prefill" if prefill else "decode",
            **updates,
        )
        plan.compile()
        workspace = torch.empty(
            plan.scratch_workspace_bytes(), device=device, dtype=torch.uint8
        )

        def forward_cudnn():
            return plan.execute(
                *inputs,
                workspace,
                query_scale,
                max_query_tokens=rows if prefill else 1,
                **updates,
            )["indices"]

        return SimpleNamespace(
            fn=forward_cudnn,
            metadata={},
            output=output,
            positions=positions,
            q=q.float() if dtype == "bf16" else dequantize(q_values, q_scales, dtype),
            weights=weights,
            get_keys=get_keys,
            rows=rows,
            prefill=prefill,
            workspace_bytes=workspace.numel(),
        )

    metadata = DeepseekV32IndexerMetadata(
        seq_lens=torch.full((requests,), length, device=device, dtype=torch.int32),
        max_seq_len=length,
        slot_mapping=slots,
        num_decodes=0 if prefill else requests,
        num_decode_tokens=0 if prefill else rows,
        num_prefills=requests if prefill else 0,
        num_prefill_tokens=rows if prefill else 0,
    )
    if prefill:
        chunk = DeepseekV32IndexerPrefillChunkMetadata(
            block_table=block_table,
            cu_seqlen_ks=torch.zeros(rows, device=device, dtype=torch.int32),
            cu_seqlen_ke=ends,
            cu_seq_lens=torch.tensor([0, pools], device=device, dtype=torch.int32),
            token_to_seq=torch.zeros(pools, device=device, dtype=torch.int32),
            total_seq_lens=pools,
            max_seqlen_k=pools,
            local_cu_seq_lens=torch.tensor(
                [0, pools], device=device, dtype=torch.int32
            ),
            local_total_seq_lens=pools,
            max_local_total_seq_lens=pools,
            token_start=0,
            token_end=rows,
            num_reqs=1,
        )
        metadata.prefill = DeepseekV32IndexerPrefillMetadata([chunk], length)
    else:
        ones = torch.ones(requests, device=device, dtype=torch.int32)
        context_lens = ends.reshape(requests, 1)
        decode_indices = torch.arange(requests, device=device, dtype=torch.int32)
        schedule = get_paged_mqa_logits_metadata(
            context_lens,
            PAGE,
            torch.cuda.get_device_properties(0).multi_processor_count,
            indices=decode_indices,
        )
        metadata.decode = DeepSeekV32IndexerDecodeMetadata(
            block_table=block_table,
            seq_lens=context_lens,
            decode_lens=ones,
            requires_padding=False,
            schedule_metadata=schedule,
            indices=decode_indices,
            per_req_decode_lens=ones,
            decode_is_uniform=True,
            write_max_decode_len=1,
        )
    tail_meta = dataclasses.replace(
        metadata, slot_mapping=request_ids * POOL + positions % POOL
    )
    op = SparseAttnIndexerKpool(
        k_cache=SimpleNamespace(prefix="bench.index", kv_cache=cache),
        quant_block_size=DIM,
        scale_fmt="ue8m0",
        topk_tokens=TOPK,
        head_dim=DIM,
        max_pool_len=capacity,
        max_total_seq_len=requests * pools,
        topk_indices_buffer=output,
        skip_k_cache_insert=not write,
        use_fp4_cache=dtype == "fp4",
        tail_cache=SimpleNamespace(prefix="bench.tail", kv_cache=tail),
    )
    hidden = torch.empty(rows, 4096, device=device, dtype=torch.bfloat16)

    def forward():
        return op.forward_native(
            hidden,
            q_input,
            raw_k,
            native_weights,
            gate_score=gates,
            compress_ape=ape,
            index_kpool=POOL,
            positions=positions,
        )

    return SimpleNamespace(
        fn=forward,
        metadata={"bench.index": metadata, "bench.tail": tail_meta},
        output=output,
        positions=positions,
        q=dequantize(q_values, q_scales, dtype),
        weights=weights,
        get_keys=get_keys,
        rows=rows,
        prefill=prefill,
    )


def validate(case, out=None):
    out = (case.fn() if out is None else out).clone().long()
    lengths = case.positions + 1
    assert ((out >= -1) & (out < lengths[:, None])).all().item()
    pools = out[:, :TOPK].reshape(case.rows, -1, POOL)
    assert (pools == pools[..., :1] + torch.arange(POOL, device="cuda")).all().item()
    assert (pools[..., 0] % POOL == 0).all().item()
    ids = pools[..., 0] // POOL
    assert (ids.sort().values.diff(dim=-1) > 0).all().item()
    tails = lengths[:, None] // POOL * POOL + torch.arange(POOL - 1, device="cuda")
    expected_tail = torch.where(tails < lengths[:, None], tails, -1)
    assert torch.equal(out[:, TOPK:], expected_tail)
    recalls = []
    for row in sorted({0, case.rows // 2, case.rows - 1}):
        end = int(lengths[row].item()) // POOL
        keys = case.get_keys(row)[:end]
        scores = (case.q[row] @ keys.T).relu()
        scores = (scores * case.weights[row, :, None]).sum(0)
        cutoff = scores.topk(TOPK // POOL).values[-1]
        chosen = scores[ids[row]]
        # Allow BF16 matmul rounding and near-ties; do not accept wrong rows/pages.
        tolerance = scores.abs().max() * 0.01 + 1e-5
        assert (chosen >= cutoff - tolerance).all().item(), "top-k score mismatch"
        reference_ids = scores.topk(TOPK // POOL).indices
        recalls.append(torch.isin(ids[row], reference_ids).float().mean().item())
    assert min(recalls) >= 0.98, "top-k set mismatch beyond rounding near the cutoff"
    return {
        "checked_rows": len(recalls),
        "min_topk_overlap": min(recalls),
        "causal_unique_pool_ids_and_tail": "pass",
    }


def measure(fn, warmup, repeats, inner):
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    wall, event = [], []
    for _ in range(repeats):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        torch.accelerator.synchronize()
        t0 = time.perf_counter()
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - t0) * 1e6 / inner)
        event.append(start.elapsed_time(end) * 1000 / inner)
    return {
        "wall_us_median": statistics.median(wall),
        "wall_us_min": min(wall),
        "wall_us_max": max(wall),
        "cuda_span_us_median": statistics.median(event),
        "wall_samples_us": wall,
        "cuda_span_samples_us": event,
    }


def measure_graph(fn, warmup, repeats, inner):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            fn()
    measured = measure(graph.replay, warmup, repeats, 1)
    return {
        "calls_per_replay": inner,
        **{
            k: [x / inner for x in v] if isinstance(v, list) else v / inner
            for k, v in measured.items()
        },
    }


def profile(fn, count, trace):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as p:
        for _ in range(count):
            fn()
        torch.accelerator.synchronize()
    p.export_chrome_trace(str(trace))
    kernels = defaultdict(lambda: {"calls_per_forward": 0, "us_per_forward": 0})
    for e in p.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            kernels[e.name]["calls_per_forward"] += 1 / count
            kernels[e.name]["us_per_forward"] += e.device_time_total / count
    return sorted(
        ({"name": name, **v} for name, v in kernels.items()),
        key=lambda x: x["us_per_forward"],
        reverse=True,
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["fp8", "fp4", "bf16"],
        default=["fp8", "fp4", "bf16"],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["decode-read", "decode-write", "prefill-read", "prefill-write"],
        default=["decode-read", "decode-write"],
    )
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 64])
    parser.add_argument("--lengths", nargs="+", type=int, default=[8192, 32768, 131072])
    parser.add_argument("--prefill-queries", nargs="+", type=int, default=[128])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--inner", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=["native", "cudnn"],
        default=["native"],
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--native-source-dir",
        type=Path,
        help="Load sparse_indexer.py and ops/kpool_compress.py from this candidate.",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Also time full CUDA graph replay as a diagnostic.",
    )
    parser.add_argument("--output", type=Path, default=Path("kpool.json"))
    args = parser.parse_args()
    if (
        args.implementations == ["native"]
        and "fp8" not in args.dtypes
        and all(m.endswith("write") for m in args.modes)
    ):
        parser.error("Only FP8 supports cache writes; select a read mode or FP8.")
    if min(args.lengths) <= TOPK + max(args.prefill_queries):
        parser.error("Use lengths above topk + query count to exercise scoring/top-k.")
    if (
        min(
            args.batches
            + args.prefill_queries
            + [args.warmup, args.repeats, args.inner]
        )
        < 1
    ):
        parser.error("Counts must be positive.")
    if "prefill-write" in args.modes and any(
        (n - q) % POOL for n in args.lengths for q in args.prefill_queries
    ):
        parser.error("Prefill writes require pool-aligned chunk starts.")
    if "fp4" in args.dtypes and torch.cuda.get_device_capability()[0] != 10:
        parser.error("MXFP4 native indexer requires SM100-family hardware.")
    torch.backends.cuda.matmul.allow_tf32 = False
    init_workspace_manager(torch.device("cuda:0"))
    source = Path(inspect.getfile(sparse_indexer))
    if args.native_source_dir is not None:
        source = load_native_candidate(args.native_source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = args.output.parent / f"{args.output.stem}_traces"
    if args.profile:
        trace_dir.mkdir(exist_ok=True)
    source_copy = args.output.with_suffix(".source.py")
    source_copy.write_bytes(source.read_bytes())
    report = {
        "environment": {
            "vllm_main_commit": "ceb01b6ddf6e19510ba138f810d38dd70e665f52",
            "vllm_wheel_commit": "1757f6720c3c1cc130a834b41f264ac22a782289",
            "host": socket.gethostname(),
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cutedsl": importlib.metadata.version("nvidia-cutlass-dsl"),
            "triton": importlib.metadata.version("triton"),
            "gpu": torch.cuda.get_device_name(),
            "sm": torch.cuda.get_device_capability(),
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "kpool_ops_sha256": hashlib.sha256(
                (source.parent / "ops" / "kpool_compress.py").read_bytes()
            ).hexdigest(),
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "nvidia_smi": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,driver_version,memory.total,clocks.sm,clocks.mem,power.limit",
                    "--format=csv",
                ],
                text=True,
            ),
        },
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "shape": {
            "index_heads": HEADS,
            "head_dim": DIM,
            "pool_size": POOL,
            "topk_tokens": TOPK,
            "page_size_pools": PAGE,
        },
        "unsupported": {
            "bf16_forward_native": (
                "No BF16 index-cache path; native BF16 is omitted."
            ),
            "native_fp4_write": "Unfused FP4 Insert is not supported yet.",
        },
        "timing": (
            "Warm steady-state, repeated fixed inputs, eager Python call; "
            "metadata and input quantization excluded. CUDA span includes host "
            "launch gaps. Graph captures inner calls per replay and reports "
            "time per call as a separate diagnostic. Profiler kernel "
            "sums exclude those gaps."
        ),
        "results": [],
    }
    if "cudnn" in args.implementations:
        import cudnn
        from cudnn.deepseek_sparse_attention.pooled import forward as pooled_forward

        folder = Path(inspect.getfile(pooled_forward)).parent
        report["environment"]["cudnn"] = {
            "version": cudnn.__version__,
            "source": str(cudnn.__file__),
            "sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(folder.glob("*.py"))
            },
        }
    config = VllmConfig()
    for mode in args.modes:
        sizes = args.prefill_queries if mode.startswith("prefill") else args.batches
        for size in sizes:
            for length in args.lengths:
                for dtype in args.dtypes:
                    for implementation in args.implementations:
                        if implementation == "native" and (
                            dtype == "bf16"
                            or (mode.endswith("write") and dtype != "fp8")
                        ):
                            continue
                        name = f"{implementation}_{mode}_{dtype}_n{size}_t{length}"
                        print(f"RUN {name}", flush=True)
                        with set_current_vllm_config(config):
                            case = build_case(
                                dtype,
                                mode,
                                size,
                                length,
                                size,
                                args.seed,
                                implementation,
                            )
                            with set_forward_context(case.metadata, config):
                                checks = validate(case)
                                if args.graph:
                                    graph = torch.cuda.CUDAGraph()
                                    with torch.cuda.graph(graph):
                                        case.fn()
                                    graph.replay()
                                    checks["graph_replay"] = validate(case, case.output)
                                method = (
                                    "IndexerForward.execute"
                                    if implementation == "cudnn"
                                    else (
                                        "torch_bf16_reference"
                                        if dtype == "bf16"
                                        else "forward_native"
                                    )
                                )
                                result = {
                                    "case": name,
                                    "dtype": dtype,
                                    "mode": mode,
                                    "rows": case.rows,
                                    "context_tokens": length,
                                    "method": method,
                                    "validation": checks,
                                }
                                if implementation == "cudnn":
                                    result["workspace_bytes"] = case.workspace_bytes
                                if not args.check_only:
                                    result.update(
                                        measure(
                                            case.fn,
                                            args.warmup,
                                            args.repeats,
                                            args.inner,
                                        )
                                    )
                                    if args.graph:
                                        result["graph_replay"] = measure_graph(
                                            case.fn,
                                            args.warmup,
                                            args.repeats,
                                            args.inner,
                                        )
                                    if args.profile:
                                        trace = trace_dir / f"{name}.trace.json"
                                        result["kernels"] = profile(case.fn, 5, trace)
                                        result["trace"] = str(trace)
                        report["results"].append(result)
                        args.output.write_text(json.dumps(report, indent=2) + "\n")
                        print(
                            json.dumps(
                                {k: v for k, v in result.items() if k != "kernels"}
                            ),
                            flush=True,
                        )
                        del case
                        torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
