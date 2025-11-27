"""
NeurIPS TP and MoE Configuration Testing

This test runs batch-1 benchmarks with different TP sizes to find optimal
configurations for large MoE models.

Models are divided into two categories:

1. FP8-Compatible MoE Models (with flashinfer_trtllm backend):
   - DeepSeek V3: TP4, TP8 with EP=2
   - Qwen 235B: TP4, TP8 with EP=2
   - Qwen Coder 480B: TP4, TP8 with EP=2

2. Native-Precision MoE Models (no quantization, no backend flag):
   - Minimax M2: TP1, TP2, TP8 (230B total, 10B active - smaller model)
   - Kimi K2 Thinking: TP1, TP2, TP8 (smaller model, has built-in compressed-tensors)
   - GLM 4.6: TP8 only (357B large model)

Configuration details:
- Batch size: 1
- Input length: 4096
- Output length: 512
- Server timeout: 2000 seconds (for large model downloads)
- TP8 configs use EP=2 to satisfy quantization block alignment

See NEURIPS_MOE_CONFIG_GUIDE.md for detailed configuration rules.
"""

import gc
import time
import unittest

from nightly_utils import NightlyBenchmarkRunner

from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

# Profile directory for all results
PROFILE_DIR = "performance_profiles_neurips_tp_moe"

# Batch size 1 only (as requested)
BATCH_SIZES = [1]
INPUT_LENS = (4096,)
OUTPUT_LENS = (512,)

# Models to test
# Two categories:
# 1. FP8-compatible MoE models: DeepSeek V3, Qwen 235B, Qwen Coder 480B
#    - Use --quantization fp8 and --moe-runner-backend flashinfer_trtllm
# 2. Native-precision MoE models: Minimax M2, Kimi K2, GLM 4.6
#    - NO quantization flag, NO moe-runner-backend flag (use defaults)
MODELS = {
    "deepseek-v3": {
        "path": "deepseek-ai/DeepSeek-V3",
        "is_moe": True,
        "use_fp8": True,
        "extra_args": ["--trust-remote-code", "--quantization", "fp8"],
    },
    "qwen3-235b": {
        "path": "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        "is_moe": True,
        "use_fp8": True,
        "extra_args": [
            "--trust-remote-code",
            "--quantization",
            "fp8",
            "--mem-fraction-static",
            "0.8",
        ],
    },
    "qwen3-coder-480b": {
        "path": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        "is_moe": True,
        "use_fp8": True,
        "extra_args": ["--trust-remote-code", "--quantization", "fp8"],
    },
    "minimax-m2": {
        "path": "MiniMaxAI/MiniMax-M2",
        "is_moe": True,
        "use_fp8": False,  # Native precision only
        "extra_args": ["--trust-remote-code"],
        "tp_sizes": [1, 2, 8],  # Smaller model (10B active params)
    },
    "kimi-k2": {
        "path": "moonshotai/Kimi-K2-Thinking",
        "is_moe": True,
        "use_fp8": False,  # Has built-in compressed-tensors quantization
        "extra_args": [
            "--trust-remote-code",
            "--tool-call-parser",
            "kimi_k2",
            "--reasoning-parser",
            "kimi_k2",
        ],
        "tp_sizes": [1, 2, 8],  # Smaller model
    },
    "glm-4-6": {
        "path": "zai-org/GLM-4.6",
        "is_moe": True,  # 357B MoE model
        "use_fp8": False,  # Native precision only
        "extra_args": ["--trust-remote-code"],
        "tp_sizes": [8],  # Large model, only TP8
    },
}

# TP sizes to test
TP_SIZES = [4, 8]

# MoE backends to test (for MoE models only)
# Note: flashinfer_cutlass requires modelopt_fp4, not fp8, so we only test flashinfer_trtllm
MOE_BACKENDS = ["flashinfer_trtllm"]

# Server launch timeout (33 minutes for large model downloads)
SERVER_LAUNCH_TIMEOUT = 2000


class TestNeurIPSTPMoEConfigs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.runner = NightlyBenchmarkRunner(
            PROFILE_DIR, "NeurIPS TP and MoE Configuration Tests", cls.base_url
        )
        cls.runner.setup_profile_directory()

    def test_all_tp_moe_configs(self):
        """Run batch-1 benchmarks for all models with different TP/MoE configs."""

        all_results = []
        failed_configs = []
        successful_configs = []

        for model_key, model_info in MODELS.items():
            model_path = model_info["path"]
            is_moe = model_info["is_moe"]
            use_fp8 = model_info["use_fp8"]
            base_extra_args = model_info["extra_args"]
            # Use per-model TP sizes if specified, otherwise use default
            model_tp_sizes = model_info.get("tp_sizes", TP_SIZES)

            print(f"\n{'='*80}")
            print(f"Testing {model_key}: {model_path}")
            print(f"MoE Model: {is_moe}, FP8: {use_fp8}")
            print(f"{'='*80}\n")

            for tp_size in model_tp_sizes:
                # Build base server args with TP
                server_args = ["--tp", str(tp_size)] + base_extra_args

                # For TP8 with MoE models, add EP to avoid division errors
                # (moe_intermediate_size / moe_tp_size) must be divisible by weight_block_size_n=128
                # This applies to both FP8 and native-precision MoE models
                if is_moe and tp_size == 8:
                    server_args += ["--ep", "2"]  # moe_tp_size = 8/2 = 4

                if is_moe and use_fp8:
                    # FP8 MoE models: Test with flashinfer_trtllm backend
                    for moe_backend in MOE_BACKENDS:
                        ep_str = "_EP2" if tp_size == 8 else ""
                        variant = f"TP{tp_size}{ep_str}_{moe_backend}"
                        config_name = f"{model_key} {variant}"
                        moe_server_args = server_args + [
                            "--moe-runner-backend",
                            moe_backend,
                        ]

                        print(f"\nRunning {config_name}...")

                        try:
                            results, success = self.runner.run_benchmark_for_model(
                                model_path=model_path,
                                batch_sizes=BATCH_SIZES,
                                input_lens=INPUT_LENS,
                                output_lens=OUTPUT_LENS,
                                other_args=moe_server_args,
                                variant=variant,
                            )

                            if success and results:
                                all_results.extend(results)
                                self.runner.add_report(results)
                                successful_configs.append(config_name)
                                print(f"✓ Success: {config_name}")
                            else:
                                failed_configs.append(config_name)
                                print(f"⚠️  Failed: {config_name}")
                        except Exception as e:
                            failed_configs.append(config_name)
                            print(f"⚠️  Error running {config_name}: {e}")

                        # Force garbage collection and wait for GPU memory to clear
                        gc.collect()
                        time.sleep(5)
                else:
                    # Native-precision MoE models: No MoE backend flag, use defaults
                    ep_str = "_EP2" if (is_moe and tp_size == 8) else ""
                    variant = f"TP{tp_size}{ep_str}"
                    config_name = f"{model_key} {variant}"

                    print(f"\nRunning {config_name}...")

                    try:
                        results, success = self.runner.run_benchmark_for_model(
                            model_path=model_path,
                            batch_sizes=BATCH_SIZES,
                            input_lens=INPUT_LENS,
                            output_lens=OUTPUT_LENS,
                            other_args=server_args,
                            variant=variant,
                        )

                        if success and results:
                            all_results.extend(results)
                            self.runner.add_report(results)
                            successful_configs.append(config_name)
                            print(f"✓ Success: {config_name}")
                        else:
                            failed_configs.append(config_name)
                            print(f"⚠️  Failed: {config_name}")
                    except Exception as e:
                        failed_configs.append(config_name)
                        print(f"⚠️  Error running {config_name}: {e}")

                    # Force garbage collection and wait for GPU memory to clear
                    gc.collect()
                    time.sleep(5)

        # Write final report to GitHub summary
        self.runner.write_final_report()

        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"Total successful: {len(successful_configs)}")
        print(f"Total failed: {len(failed_configs)}")
        print(
            f"Total configurations attempted: {len(successful_configs) + len(failed_configs)}"
        )

        if successful_configs:
            print(f"\n✓ Successful configurations:")
            for config in successful_configs:
                print(f"  - {config}")

        if failed_configs:
            print(f"\n⚠️  Failed configurations:")
            for config in failed_configs:
                print(f"  - {config}")

        print(f"{'='*80}\n")

        # Don't fail the test - just log the failures


if __name__ == "__main__":
    unittest.main()
