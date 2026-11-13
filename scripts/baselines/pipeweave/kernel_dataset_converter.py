#!/usr/bin/env python3
"""Convert canonical Midlayer kernel datasets into PipeWeave training CSVs.

Reads `kernel_dataset/<hardware_id>/*.json.zst`, extracts operator shapes
and measured durations, then computes PipeWeave analytical features to produce
training CSVs compatible with train_mlp.py.

"""
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
PIPEWEAVE_ROOT = REPO_ROOT / "pipeweave"
HARDWARE_CONFIG_DIR = REPO_ROOT / "configs/baselines/pipeweave"
DATASET_DIRECTORIES = {
    "nvidia_a800": "dataset",
    "metax_c500": "dataset_c500",
    "ascend_910b": "dataset_910b",
}
sys.path.insert(0, str(PIPEWEAVE_ROOT))
sys.path.insert(0, str(PIPEWEAVE_ROOT / "analytical_model"))
sys.path.insert(0, str(REPO_ROOT))

from aggregator import DurationPredictor
from src.kernel.dataset_storage import load_kernel_dataset_document


def _require_integer_field(mapping, key, context):
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context}.{key} must be an integer, got {value!r}")
    return value


def _require_number_field(mapping, key, context):
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context}.{key} must be a number, got {value!r}")
    return value


def extract_gemm_csv(ncu_data, pred, hw_spec):
    rows = []
    for op_name, rec in ncu_data.items():
        oi = rec["op_info"]
        if oi["Type"] != "GEMM":
            continue
        attr = oi["Attr"]
        context = f"{op_name}.op_info.Attr"
        B = _require_integer_field(attr, "B", context)
        M_total = B * _require_integer_field(attr, "M", context)
        N = _require_integer_field(attr, "N", context)
        K = _require_integer_field(attr, "K", context)
        dur_ns = _require_number_field(
            oi, "Overall Duration [nsecond]", f"{op_name}.op_info")
        dur_ms = dur_ns / 1e6

        config = pred.find_nearest_gemm_config(M_total, N, K)
        if config is None:
            raise ValueError(
                f"PipeWeave has no GEMM analytical config for {op_name} "
                f"shape {(M_total, N, K)}")
        features = pred.generate_gemm_features(
            M_total, N, K, config["tile_M"], config["tile_N"],
            config["tile_K"], config["cta_count"], config["is_split_k"],
        )

        sum_all_cycle = float(features[1])
        sm_freq = hw_spec["sm_freq"]
        if dur_ms <= 0:
            raise ValueError(f"GEMM {op_name} has non-positive duration {dur_ms}")
        overall_perf = sum_all_cycle / (dur_ms * 1e3 * sm_freq)
        if overall_perf <= 0 or overall_perf > 2.0:
            raise ValueError(
                f"GEMM {op_name} has invalid overall_perf {overall_perf}")

        rows.append({
            "M": M_total, "N": N, "K": K, "avg_duration": dur_ms,
            "tile_M": config["tile_M"], "tile_N": config["tile_N"],
            "tile_K": config["tile_K"], "is_split_k": int(config["is_split_k"]),
            "cta_count": config["cta_count"],
            "hardware": hw_spec["name"],
            "tc_bf16": hw_spec["tc_bf16"], "xu_fp32": hw_spec["xu_fp32"],
            "fma_fp32": hw_spec["fma_fp32"], "num_sms": hw_spec["num_sms"],
            "sm_freq": sm_freq, "mem_bandwidth": hw_spec["mem_bandwidth"],
            "l2_cache_bandwidth": hw_spec["l2_cache_bandwidth"],
            "shared_memory_bandwidth": hw_spec["shared_memory_bandwidth"],
            "tensor_all_ops": features[0], "tensor_all_cycle": features[1],
            "tensor_sm_max_ops": features[2], "tensor_sm_max_cycle": features[3],
            "global_in_flight": features[4], "global_cycle": features[5],
            "local_cycle": features[6], "sm_max_in_flight": features[7],
            "sm_max_global_cycle": features[8], "sm_max_shared_cycle": features[9],
            "sm_max_local_cycle": features[10],
            "overall_perf": overall_perf,
        })
    return pd.DataFrame(rows)


def extract_attn_csv(ncu_data, pred, hw_spec):
    arch = str(hw_spec["architecture"]).lower()
    if arch == "hopper":
        attn_type = "fa3_ragged"
    elif arch in (
            "ampere", "metax_c500", "ascend_910b2", "pim", "gemmini", "lpu"):
        attn_type = "fa2_ragged"
    else:
        raise ValueError(f"unsupported PipeWeave attention architecture: {arch!r}")
    rows = []
    for op_name, rec in ncu_data.items():
        oi = rec["op_info"]
        if oi["Type"] != "FLASHATTN":
            continue
        attr = oi["Attr"]
        context = f"{op_name}.op_info.Attr"
        B = _require_integer_field(attr, "B", context)
        S = _require_integer_field(attr, "S", context)
        HQ = _require_integer_field(attr, "HQ", context)
        HKV = _require_integer_field(attr, "HKV", context)
        D = _require_integer_field(attr, "D", context)
        dur_ns = _require_number_field(
            oi, "Overall Duration [nsecond]", f"{op_name}.op_info")
        dur_ms = dur_ns / 1e6

        q_lengths = [S] * B
        kv_lengths = [S] * B
        features = pred.generate_attn_features(
            bs=B, q_lengths=q_lengths, kv_lengths=kv_lengths,
            nh=HQ, nkv=HKV, hd=D, attention_type=attn_type,
        )

        sum_all_cycle = float(features[1])
        sm_freq = hw_spec["sm_freq"]
        if dur_ms <= 0:
            raise ValueError(
                f"FLASHATTN {op_name} has non-positive duration {dur_ms}")
        overall_perf = sum_all_cycle / (dur_ms * 1e3 * sm_freq)
        if overall_perf <= 0 or overall_perf > 2.0:
            raise ValueError(
                f"FLASHATTN {op_name} has invalid overall_perf {overall_perf}")

        rows.append({
            "attention_type": attn_type, "bs": B,
            "q_lengths": str(q_lengths), "kv_lengths": str(kv_lengths),
            "nh": HQ, "nkv": HKV, "hd": D, "avg_duration": dur_ms,
            "hardware": hw_spec["name"],
            "tc_bf16": hw_spec["tc_bf16"], "xu_fp32": hw_spec["xu_fp32"],
            "fma_fp32": hw_spec["fma_fp32"], "num_sms": hw_spec["num_sms"],
            "sm_freq": sm_freq, "mem_bandwidth": hw_spec["mem_bandwidth"],
            "l2_cache_bandwidth": hw_spec["l2_cache_bandwidth"],
            "shared_memory_bandwidth": hw_spec["shared_memory_bandwidth"],
            "tensor_all_ops": features[0], "tensor_all_cycle": features[1],
            "tensor_sm_max_ops": features[2], "tensor_sm_max_cycle": features[3],
            "xu_all_ops": features[4], "xu_all_cycle": features[5],
            "xu_sm_max_ops": features[6], "xu_sm_max_cycle": features[7],
            "global_in_flight": features[8], "global_cycle": features[9],
            "local_cycle": features[10], "sm_max_in_flight": features[11],
            "sm_max_global_cycle": features[12], "sm_max_shared_cycle": features[13],
            "sm_max_local_cycle": features[14],
            "overall_perf": overall_perf,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hardware-id", required=True, choices=sorted(DATASET_DIRECTORIES))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()
    if not 0 < args.test_frac < 1:
        raise ValueError("--test-frac must be in (0, 1)")

    source_dir = REPO_ROOT / "kernel_dataset" / args.hardware_id
    out_dir = Path(args.output_dir) if args.output_dir else (
        PIPEWEAVE_ROOT / DATASET_DIRECTORIES[args.hardware_id])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the two operator families supported by PipeWeave's native trainer.
    ncu_data = {}
    for filename in ("gemm.json.zst", "flash_attn.json.zst"):
        path = source_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = load_kernel_dataset_document(path)
        overlap = set(ncu_data).intersection(payload)
        if overlap:
            raise ValueError(f"duplicate kernel dataset keys in {filename}")
        ncu_data.update(payload)
    print(f"Loaded {len(ncu_data)} cases from {source_dir}")

    pred = DurationPredictor(
        hardware_name=args.hardware_id,
        model_dir=str(PIPEWEAVE_ROOT / "mlp_models"),
        dataset_dir=str(PIPEWEAVE_ROOT / "dataset"),
        hardware_dir=str(HARDWARE_CONFIG_DIR),
        device="cpu",
    )
    hw_spec = pred.hardware_spec

    # Build GEMM CSV
    print("\n=== Building GEMM training data ===")
    gemm_df = extract_gemm_csv(ncu_data, pred, hw_spec)
    print(f"GEMM: {len(gemm_df)} valid samples")

    # Build ATTN CSV
    print("\n=== Building ATTN training data ===")
    attn_df = extract_attn_csv(ncu_data, pred, hw_spec)
    print(f"ATTN: {len(attn_df)} valid samples")

    # Write train/test splits
    for op_type, df in [("gemm", gemm_df), ("attn", attn_df)]:
        if len(df) == 0:
            raise ValueError(f"{op_type} produced no PipeWeave training samples")
        n_test = max(1, int(len(df) * args.test_frac))
        df_shuf = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df_train = df_shuf.iloc[n_test:]
        df_test = df_shuf.iloc[:n_test]

        train_path = out_dir / f"{op_type}_train.csv"
        test_path = out_dir / f"{op_type}_test.csv"
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)
        print(f"  {op_type}: {len(df_train)} train + {len(df_test)} test → {out_dir}")

    print("\nDone. Run model_trainer.py for native PipeWeave MLPs.")


if __name__ == "__main__":
    main()
