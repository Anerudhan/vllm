# GLM-5.3-Flash pooled indexer

This branch adds direct prefill cache input and an experimental cuDNN
pooled-selection route. The DeepGEMM scorer, cache format, and final sparse
attention remain unchanged.

## Enable fused selection

Set `VLLM_GLM5NEXT_USE_CUDNN_KPOOL=1`. The default is `0`.

This route requires CuTe DSL **4.8.0** and cuDNN Frontend develop at
`50950b2aa2cf38b1f73278e74e45f16144e7ecea` with the pooled DSA additions.
The required internal entry is
`cudnn.deepseek_sparse_attention.pooled.forward._compile_pooled_select`.
It is not supplied by an unmodified frontend release. An eligible call
raises an error if these additions are missing.

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
The tuned 8192–32768-pool variants use 32 bytes of selector scratch. K-gather
and logits storage are additional workspace.

## Prefill cache writes

The Triton writer accepts raw `[tokens, head_dim]` K/gates as well as gathered
`[pools, pool_size, head_dim]` inputs. Pool completions carry valid physical
slots; other tokens carry negative slots. Direct input removes the temporary
pool gathers while preserving pooling, Hadamard rotation, and quantization.
Tail seeding remains separate. This change does not require the cuDNN opt-in.

## Benchmark

`benchmarks/kernels/benchmark_glm5next_kpool.py` measures `forward_native` in
eager mode and with `--graph`. It checks causal selection, distinct pools,
tails, and sampled scores before timing. Set the environment variable before
starting each process to compare native and fused selection.

The paired GB200 measurements cover batches 1/64, prefill chunks
128/1024/4096, contexts 32K/128K, FP8/MXFP4 reads, and FP8 writes. Query
projection/quantization and final sparse attention are outside the timer.
Native BF16 cache and MXFP4 insertion are unsupported in this entry point.

The measured kernel chain wins all 30 comparable cases in both run orders:
1.05–1.60× in CUDA graph time. This is an indexer microbenchmark, not an
end-to-end GLM-5.3-Flash result. Model serving with this route is not yet
qualified. The adapter uses an internal frontend API and remains experimental.
