#!/usr/bin/env python3
"""Convert benchmark-runner JSON into PipeWeave training CSVs.

Runs on the LOCAL machine where PipeWeave is available. Takes measured durations
from the remote hardware benchmark and computes analytical features using
PipeWeave's pipe calculators.

"""
import argparse
import json
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

from aggregator import DurationPredictor


def _require_record(record, required, context):
    if not isinstance(record, dict) or set(record) != set(required):
        raise ValueError(
            f"{context} must contain exactly {sorted(required)}, got "
            f"{sorted(record) if isinstance(record, dict) else type(record).__name__}")
    return record


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


def build_gemm_csv(bench_data, pred, hw_spec):
    rows = []
    for index, raw_record in enumerate(bench_data):
        context = f"operators.gemm[{index}]"
        rec = _require_record(
            raw_record, {"M", "N", "K", "avg_duration"}, context)
        M = _require_integer_field(rec, "M", context)
        N = _require_integer_field(rec, "N", context)
        K = _require_integer_field(rec, "K", context)
        avg_dur = _require_number_field(rec, "avg_duration", context)
        config = pred.find_nearest_gemm_config(M, N, K)
        if config is None:
            raise ValueError(
                f"PipeWeave has no GEMM analytical config for {(M, N, K)}")
        features = pred.generate_gemm_features(
            M, N, K, config["tile_M"], config["tile_N"], config["tile_K"],
            config["cta_count"], config["is_split_k"],
        )
        sum_all_cycle = float(features[1])
        sm_freq = hw_spec["sm_freq"]
        if avg_dur <= 0:
            raise ValueError(
                f"GEMM {(M, N, K)} has non-positive duration {avg_dur}")
        overall_perf = sum_all_cycle / (avg_dur * 1e3 * sm_freq)
        if overall_perf <= 0 or overall_perf > 1.5:
            raise ValueError(
                f"GEMM {(M, N, K)} has invalid overall_perf {overall_perf}")
        rows.append({
            "M": M, "N": N, "K": K, "avg_duration": avg_dur,
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


def build_attn_csv(bench_data, pred, hw_spec):
    arch = str(hw_spec["architecture"]).lower()
    if arch == "hopper":
        attn_type = "fa3_ragged"
    elif arch in (
            "ampere", "metax_c500", "ascend_910b2", "pim", "gemmini", "lpu"):
        attn_type = "fa2_ragged"
    else:
        raise ValueError(f"unsupported PipeWeave attention architecture: {arch!r}")
    rows = []
    for index, raw_record in enumerate(bench_data):
        context = f"operators.attn[{index}]"
        rec = _require_record(
            raw_record,
            {"bs", "seq_len", "nh", "hd", "nkv", "avg_duration"},
            context,
        )
        bs = _require_integer_field(rec, "bs", context)
        sl = _require_integer_field(rec, "seq_len", context)
        nh = _require_integer_field(rec, "nh", context)
        hd = _require_integer_field(rec, "hd", context)
        nkv = _require_integer_field(rec, "nkv", context)
        avg_dur = _require_number_field(rec, "avg_duration", context)
        q_lengths = [sl] * bs
        kv_lengths = [sl] * bs
        features = pred.generate_attn_features(
            bs=bs, q_lengths=q_lengths, kv_lengths=kv_lengths,
            nh=nh, nkv=nkv, hd=hd, attention_type=attn_type,
        )
        sum_all_cycle = float(features[1])
        sm_freq = hw_spec["sm_freq"]
        if avg_dur <= 0:
            raise ValueError(
                f"attention {(bs, sl, nh, nkv, hd)} has non-positive "
                f"duration {avg_dur}")
        overall_perf = sum_all_cycle / (avg_dur * 1e3 * sm_freq)
        if overall_perf <= 0 or overall_perf > 1.5:
            raise ValueError(
                f"attention {(bs, sl, nh, nkv, hd)} has invalid "
                f"overall_perf {overall_perf}")
        rows.append({
            "attention_type": attn_type, "bs": bs,
            "q_lengths": str(q_lengths), "kv_lengths": str(kv_lengths),
            "nh": nh, "nkv": nkv, "hd": hd, "avg_duration": avg_dur,
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


def build_ew_csv(bench_data, pred, hw_spec, op_type):
    if op_type == "rmsnorm":
        gen_fn = pred.generate_rmsnorm_features
    elif op_type == "siluandmul":
        gen_fn = pred.generate_siluandmul_features
    else:
        raise ValueError(f"unsupported elementwise operator: {op_type!r}")
    rows = []
    for index, raw_record in enumerate(bench_data):
        context = f"operators.{op_type}[{index}]"
        rec = _require_record(
            raw_record, {"seq", "dim", "avg_duration"}, context)
        seq = _require_integer_field(rec, "seq", context)
        dim = _require_integer_field(rec, "dim", context)
        avg_dur = _require_number_field(rec, "avg_duration", context)
        features = gen_fn(seq=seq, dim=dim)
        sum_all_cycle = float(features[1] + features[5])
        sm_freq = hw_spec["sm_freq"]
        if avg_dur <= 0:
            raise ValueError(
                f"{op_type} {(seq, dim)} has non-positive duration {avg_dur}")
        overall_perf = sum_all_cycle / (avg_dur * 1e3 * sm_freq)
        if overall_perf <= 0 or overall_perf > 1.5:
            raise ValueError(
                f"{op_type} {(seq, dim)} has invalid overall_perf "
                f"{overall_perf}")
        rows.append({
            "seq": seq, "dim": dim, "avg_duration": avg_dur,
            "hardware": hw_spec["name"],
            "tc_bf16": hw_spec["tc_bf16"], "xu_fp32": hw_spec["xu_fp32"],
            "fma_fp32": hw_spec["fma_fp32"], "num_sms": hw_spec["num_sms"],
            "sm_freq": sm_freq, "mem_bandwidth": hw_spec["mem_bandwidth"],
            "l2_cache_bandwidth": hw_spec["l2_cache_bandwidth"],
            "shared_memory_bandwidth": hw_spec["shared_memory_bandwidth"],
            "fma_all_ops": features[0], "fma_all_cycle": features[1],
            "fma_sm_max_ops": features[2], "fma_sm_max_cycle": features[3],
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
    parser.add_argument("--benchmark", required=True)
    parser.add_argument(
        "--hardware-id", required=True, choices=sorted(DATASET_DIRECTORIES))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()
    if not 0 < args.test_frac < 1:
        raise ValueError("--test-frac must be in (0, 1)")

    with open(args.benchmark) as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or set(payload) != {
            "schema_version", "hardware_id", "requested_operators", "operators"}:
        raise ValueError("benchmark JSON must match schema version 2 exactly")
    if payload["schema_version"] != 2:
        raise ValueError("benchmark JSON requires schema_version 2")
    if payload["hardware_id"] != args.hardware_id:
        raise ValueError("benchmark hardware_id does not match --hardware-id")
    bench = payload["operators"]
    requested = payload["requested_operators"]
    if not isinstance(bench, dict) or not isinstance(requested, list):
        raise ValueError("benchmark operators/requested_operators have invalid types")
    if set(bench) != set(requested):
        raise ValueError("benchmark operators do not match requested_operators")

    pred = DurationPredictor(
        hardware_name=args.hardware_id,
        model_dir=str(PIPEWEAVE_ROOT / "mlp_models"),
        dataset_dir=str(PIPEWEAVE_ROOT / "dataset"),
        hardware_dir=str(HARDWARE_CONFIG_DIR),
        device="cpu",
    )
    hw_spec = pred.hardware_spec

    out = Path(args.output_dir) if args.output_dir else (
        PIPEWEAVE_ROOT / DATASET_DIRECTORIES[args.hardware_id])
    out.mkdir(parents=True, exist_ok=True)

    builders = {
        "gemm": lambda d: build_gemm_csv(d, pred, hw_spec),
        "attn": lambda d: build_attn_csv(d, pred, hw_spec),
        "rmsnorm": lambda d: build_ew_csv(d, pred, hw_spec, "rmsnorm"),
        "siluandmul": lambda d: build_ew_csv(d, pred, hw_spec, "siluandmul"),
    }

    unknown = set(requested) - set(builders)
    if unknown:
        raise ValueError(f"unsupported benchmark operators: {sorted(unknown)}")
    for op_type in requested:
        builder = builders[op_type]
        df = builder(bench[op_type])
        if len(df) == 0:
            raise ValueError(f"{op_type} produced no training samples")
        n_test = max(1, int(len(df) * args.test_frac))
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df_train = df_shuffled.iloc[n_test:]
        df_test = df_shuffled.iloc[:n_test]
        train_path = out / f"{op_type}_train.csv"
        test_path = out / f"{op_type}_test.csv"
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)
        print(f"{op_type}: {len(df_train)} train + {len(df_test)} test → {train_path}")


if __name__ == "__main__":
    main()
