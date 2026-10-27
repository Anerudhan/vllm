# GLM-5.3-Flash pooled indexer

This branch adds direct prefill cache input and an experimental cuDNN
pooled-selection route. The DeepGEMM scorer, cache format, and final sparse
attention remain unchanged.

## Enable fused selection

Set `VLLM_GLM5NEXT_USE_CUDNN_KPOOL=1`. The default is `0`.

This route requires CuTe DSL **4.8.0** and cuDNN Frontend develop at
`6fcc0aa59107d9ff690f30934ab936a9ad01a3b9` with the pooled DSA additions.
The additions are in [cuDNN Frontend PR #1847](https://github.com/NVIDIA/cudnn-frontend/pull/1847).
Install the pinned source in your vLLM environment:

```bash
git clone --branch feat/pooled-dsa-blackwell https://github.com/Anerudhan/cudnn-frontend.git
git -C cudnn-frontend checkout dfd51c2f645746219aa7c102b2471f62c2038b85
uv pip install "nvidia-cutlass-dsl==4.8.0" -e ./cudnn-frontend
```

The build needs CUDA and cuDNN headers/libraries; see the
[frontend build guide](https://github.com/Anerudhan/cudnn-frontend/tree/feat/pooled-dsa-blackwell#building-from-source).

The adapter calls the public `cudnn.ops.dsa.pooled_select_forward` API.
It uses `cudnn.ops.dsa.pooled_select_workspace_size` to reserve caller-owned
UINT8 scratch through vLLM's workspace manager. These additions are not
supplied by an unmodified frontend release. An eligible call raises an error
if they are missing.

Eligibility:

- NVIDIA SM100/SM103, DCP size 1, `sparse_indexer_topk_backend="auto"`.
- Pool size 4 or 16, 512 selected pools, at most 32768 configured pools.
- Token positions are available; prefill is not row-sharded; decode needs
  no request padding.

Other cases retain the upstream selector. Explicit top-k backend choices
take priority. Warm up before CUDA graph capture.

The fused selector reads native BF16 logits, selects complete causal pools,
and writes individual token IDs plus incomplete-pool tails. Decode and
prefill use separate CuTe entry points. BF16 scores are not converted to FP32.
Selected score values match exact top-k; tied IDs and ordering may differ.
Short rows can place valid tokens and padding in different output columns.
The tuned 8192–32768-pool variants use 32 bytes of selector scratch. K-gather
and logits storage are additional workspace.

The API validates dtypes, shapes, strides, alignment, and buffer overlap.
With output and workspace supplied, it allocates no device storage, performs
no host read of device values, and launches on vLLM's current stream.
Kernel compilation stays inside cuDNN Frontend. The adapter does not call
private frontend helpers.

The other pooled DSA APIs have different scopes:

| API under `cudnn.ops.dsa` | Scope | Used by this adapter |
| --- | --- | --- |
| `pooled_select_forward` | Logits to selected token IDs, including tails and padding | Yes |
| `pooled_select_workspace_size` | Host query for selector scratch bytes | Yes |
| `pooled_indexer_forward` | Paged scoring plus selection/expansion | No; scoring stays in DeepGEMM |
| `pooled_cache_update_forward` | Gated pooling, rotation, quantization, cache writes | No; cache writes stay in vLLM Triton |
| `pooled_expand_forward` | Expand already-selected pool IDs | No; expansion is fused into selection |

## Prefill cache writes

The Triton writer accepts raw `[tokens, head_dim]` K/gates as well as gathered
`[pools, pool_size, head_dim]` inputs. Pool completions carry valid physical
slots; other tokens carry negative slots. Direct input removes the temporary
pool gathers while preserving pooling, Hadamard rotation, and quantization.
Tail seeding remains separate. This change does not require the cuDNN opt-in.

## Benchmark

The measured cuDNN improvement comes from `pooled_select_forward`: fused
selection/expansion and direct BF16 logit input. This branch also removes
temporary K/gate gathers from prefill cache writes. Against upstream main
`70d67f19dcf822b97b5f8863b7590edd73a59b23`, the combined changes improve
median GLM-5.3-Flash output-token throughput by 1.5–3.2% on GB200 and
1.0–2.7% on GB300 across the workloads below. No selector-only E2E ablation
was measured. See [PR #1847](https://github.com/NVIDIA/cudnn-frontend/pull/1847)
for the measurement scope and output checks.

**The standalone frontend scorer is not yet performant.** At 4,096 queries
and 32,768 pools, the full frontend indexer takes 10.7–11.7× as long as
DeepGEMM plus the same selector on GB200/GB300. This adapter therefore keeps
DeepGEMM scoring. The frontend scorer and cache-update API contribute no
part of the serving gains above.

`benchmarks/kernels/benchmark_glm5next_kpool.py` measures `forward_native` in
eager mode and with `--graph`. It checks causal selection, distinct pools,
tails, and sampled scores before timing. Set the environment variable before
starting each process to compare native and fused selection.

The paired GB200 measurements cover batches 1/64, prefill chunks
128/1024/4096, contexts 32K/128K, FP8/MXFP4 reads, and FP8 writes. Query
projection/quantization and final sparse attention are outside the timer.
Native BF16 cache and MXFP4 insertion are unsupported in this entry point.

GLM-5.3-Flash serving checks cover TP=4, FP8 weights/cache, 128 output tokens,
4K/16K inputs at concurrency 1/8, and 32K inputs at concurrency 1 on GB200 and
GB300. Both variants use matched MoE tuning configurations. Time requests
before profiling, and run both execution orders with fresh server processes.

Pin CPU placement for matched server runs and report each run order as well
as the combined median. Indexer speedup does not establish serving speedup.
Check output tokens and score values; tied pool IDs may change model
log-probabilities. Smoke prompts do not establish full model quality.
The adapter remains experimental and disabled by default.
