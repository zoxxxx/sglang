# NeurIPS MoE Model Configuration Guide

This guide documents the correct configuration parameters for running large MoE models with different TP (Tensor Parallelism) sizes and quantization settings.

## Model Categories

### 1. FP8-Compatible MoE Models

These models support FP8 quantization with the `flashinfer_trtllm` MoE backend.

#### Models:
- **DeepSeek V3** (`deepseek-ai/DeepSeek-V3`)
- **Qwen 235B** (`Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`)
- **Qwen Coder 480B** (`Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`)

#### Configuration Rules:

**TP4 Configuration:**
```bash
--tp 4 \
--trust-remote-code \
--quantization fp8 \
--moe-runner-backend flashinfer_trtllm
```

**TP8 Configuration:**
```bash
--tp 8 \
--ep 2 \
--trust-remote-code \
--quantization fp8 \
--moe-runner-backend flashinfer_trtllm
```

**Key Requirements:**
- ✅ Must use `--quantization fp8`
- ✅ Must use `--moe-runner-backend flashinfer_trtllm` (cutlass requires `modelopt_fp4`, not `fp8`)
- ⚠️ TP4 requires **2000+ second timeout** for initial model download
- ⚠️ TP8 requires `--ep 2` to satisfy: `(moe_intermediate_size / (tp_size / ep_size)) % weight_block_size_n == 0`
  - Without EP: `moe_intermediate_size / 8` may not be divisible by 128
  - With EP=2: `moe_intermediate_size / 4` is divisible by 128
- ⚠️ **Qwen 235B only**: Add `--mem-fraction-static 0.8` to prevent CUDA graph capture failures on TP4

---

### 2. Native-Precision MoE Models

These models must run in native precision (bf16/fp16) without explicit quantization flags.

#### Models:
- **Minimax M2** (`MiniMaxAI/MiniMax-M2`) - 230B total, **10B active** (smaller model)
- **Kimi K2 Thinking** (`moonshotai/Kimi-K2-Thinking`) - Smaller model with built-in `compressed-tensors` quantization
- **GLM 4.6** (`zai-org/GLM-4.6`) - **357B** large MoE model

#### Configuration Rules:

**Minimax M2 (Smaller - Test TP1, TP2, TP8):**
```bash
# TP1
--tp 1 \
--trust-remote-code

# TP2
--tp 2 \
--trust-remote-code

# TP8
--tp 8 \
--ep 2 \
--trust-remote-code
```

**Kimi K2 Thinking (Smaller - Test TP1, TP2, TP8):**
```bash
# TP1
--tp 1 \
--trust-remote-code \
--tool-call-parser kimi_k2 \
--reasoning-parser kimi_k2

# TP2
--tp 2 \
--trust-remote-code \
--tool-call-parser kimi_k2 \
--reasoning-parser kimi_k2

# TP8
--tp 8 \
--ep 2 \
--trust-remote-code \
--tool-call-parser kimi_k2 \
--reasoning-parser kimi_k2
```

**GLM 4.6 (Large - TP8 only):**
```bash
--tp 8 \
--ep 2 \
--trust-remote-code
```

**Key Requirements:**
- ❌ **NO** `--quantization` flag (causes conflicts or division errors)
- ❌ **NO** `--moe-runner-backend` flag (use default MoE implementation)
- ✅ **Smaller models** (Minimax M2, Kimi K2): Test TP1, TP2, TP8
- ✅ **Large model** (GLM 4.6): TP8 only (TP4 causes OOM)
- ✅ Use `--ep 2` for TP8 to avoid division errors

---

## Common Configuration Errors

### Error: "flashinfer_cutlass requires modelopt_fp4 or bf16"
```
AssertionError: modelopt_fp4 quantization or bf16 is required for Flashinfer Cutlass MOE
```
**Solution:** Use `flashinfer_trtllm` instead of `flashinfer_cutlass` when using `--quantization fp8`

### Error: Quantization block alignment
```
ValueError: For quantized MoE models, please make sure
(moe_intermediate_size=XXX / moe_tp_size=8) % weight_block_size_n=128 == 0
```
**Solution:** Add `--ep 2` when using TP8, so `moe_tp_size = tp_size / ep_size = 8 / 2 = 4`

### Error: Kimi K2 quantization conflict
```
ValueError: Quantization method specified in the model config (compressed-tensors)
does not match the quantization method specified in the `quantization` argument (fp8).
```
**Solution:** Remove `--quantization fp8` flag - Kimi K2 has built-in quantization

### Error: Native-precision model FP8 errors
```
ValueError: The output_size of gate's and up's weight = 192 is not divisible by weight quantization block_n = 128.
```
**Solution:** Remove `--quantization fp8` flag - these models don't support FP8

### Error: TP4 OOM on large models
```
torch.OutOfMemoryError: CUDA out of memory.
```
**Solution:** Use TP8 instead of TP4 for large models (Minimax M2, Kimi K2, GLM 4.6)

### Error: Server launch timeout
```
TimeoutError: Server failed to start within the timeout period.
```
**Solution:** Increase timeout to 2000 seconds (33 minutes) for large model downloads

### Error: CUDA graph capture failure
```
Exception: Capture cuda graph failed: No supported CUDA architectures found for major versions [10].
Possible solutions:
1. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)
2. set --cuda-graph-max-bs to a smaller value (e.g., 16)
```
**Solution:** Add `--mem-fraction-static 0.8` to reduce static memory allocation

---

## Configuration Summary Table

| Model | TP Sizes | Quantization | MoE Backend | EP Size | Notes |
|-------|----------|--------------|-------------|---------|-------|
| DeepSeek V3 | TP4, TP8 | `fp8` | `flashinfer_trtllm` | EP=2 for TP8 | Large download time |
| Qwen 235B | TP4, TP8 | `fp8` | `flashinfer_trtllm` | EP=2 for TP8 | Large download time |
| Qwen Coder 480B | TP4, TP8 | `fp8` | `flashinfer_trtllm` | EP=2 for TP8 | Large download time |
| Minimax M2 | TP1, TP2, TP8 | None (native) | None (default) | EP=2 for TP8 | Smaller model (10B active) |
| Kimi K2 | TP1, TP2, TP8 | None (built-in) | None (default) | EP=2 for TP8 | Has compressed-tensors |
| GLM 4.6 | TP8 | None (native) | None (default) | EP=2 | Large 357B model |

---

## Test Configuration

The automated test script ([test/nightly/test_neurips_tp_moe_configs.py](test/nightly/test_neurips_tp_moe_configs.py)) uses these settings:

- **Batch size:** 1
- **Input length:** 4096 tokens
- **Output length:** 512 tokens
- **Server timeout:** 2000 seconds
- **Memory cleanup:** 5-second sleep between tests with `gc.collect()`

**Total configurations tested:** 13
- FP8 models: 3 models × 2 TP sizes (TP4, TP8) = 6 configs
- Native smaller models: 2 models × 3 TP sizes (TP1, TP2, TP8) = 6 configs
- Native large model: 1 model × 1 TP size (TP8) = 1 config

---

## References

Based on nightly test configurations:
- [test/nightly/test_deepseek_v32_perf.py](test/nightly/test_deepseek_v32_perf.py)
- [test/nightly/test_qwen3_235b_perf.py](test/nightly/test_qwen3_235b_perf.py)
- [test/nightly/test_minimax_m2_perf.py](test/nightly/test_minimax_m2_perf.py)
- [test/nightly/test_kimi_k2_thinking_perf.py](test/nightly/test_kimi_k2_thinking_perf.py)
- [test/nightly/test_glm_4_6_perf.py](test/nightly/test_glm_4_6_perf.py)
- [test/nightly/test_flashinfer_trtllm_gen_moe_backend.py](test/nightly/test_flashinfer_trtllm_gen_moe_backend.py)
- [test/nightly/test_deepseek_v3_fp4_cutlass_moe.py](test/nightly/test_deepseek_v3_fp4_cutlass_moe.py)
