#!/usr/bin/env python3
"""Train hardware-specific PipeWeave MLPs from generated CSV datasets."""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PIPEWEAVE_ROOT = REPO_ROOT / "pipeweave"
PIPEWEAVE_PATHS = {
    "nvidia_a800": ("dataset", "mlp_models"),
    "metax_c500": ("dataset_c500", "mlp_models_c500"),
    "ascend_910b": ("dataset_910b", "mlp_models_910b"),
}
OPERATORS = ("gemm", "attn", "rmsnorm", "siluandmul")


def _replace_once(content, pattern, replacement):
    content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"PipeWeave train_mlp.py contract changed: {pattern}")
    return content


def _configured_trainer(content, operator, train_path, test_path, model_dir):
    operator_index = ["gemm", "attn", "rmsnorm", "siluandmul"].index(operator)
    content = _replace_once(
        content,
        r"^OPERATOR_TYPE = VALID_OPERATORS\[\d+\]$",
        f"OPERATOR_TYPE = VALID_OPERATORS[{operator_index}]",
    )
    content = _replace_once(
        content, r"^TRAIN_PATH = .*$", f"TRAIN_PATH = {str(train_path)!r}")
    content = _replace_once(
        content, r"^TEST_PATH = .*$", f"TEST_PATH = {str(test_path)!r}")
    return _replace_once(
        content,
        r"^OUTPUT_DIR = .*$",
        f"OUTPUT_DIR = os.path.join({str(model_dir)!r}, OPERATOR_TYPE, timestamp)",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hardware-id", required=True, choices=sorted(PIPEWEAVE_PATHS))
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument(
        "--operators", nargs="+", required=True, choices=OPERATORS,
        help="Exact operator models to train; every requested CSV pair must exist.")
    args = parser.parse_args()
    if len(args.operators) != len(set(args.operators)):
        raise ValueError("--operators must not contain duplicates")
    if args.dataset_dir is not None and not args.dataset_dir.strip():
        raise ValueError("--dataset-dir must not be empty")
    if args.model_dir is not None and not args.model_dir.strip():
        raise ValueError("--model-dir must not be empty")

    dataset_name, model_name = PIPEWEAVE_PATHS[args.hardware_id]
    dataset_dir = (
        PIPEWEAVE_ROOT / dataset_name
        if args.dataset_dir is None else Path(args.dataset_dir))
    model_dir = (
        PIPEWEAVE_ROOT / model_name
        if args.model_dir is None else Path(args.model_dir))
    source = (PIPEWEAVE_ROOT / "train_mlp.py").read_text()

    trained = []
    for operator in args.operators:
        train_path = dataset_dir / f"{operator}_train.csv"
        test_path = dataset_dir / f"{operator}_test.csv"
        if not train_path.is_file() or not test_path.is_file():
            raise FileNotFoundError(
                f"missing requested PipeWeave CSV pair: {train_path}, "
                f"{test_path}")
        configured = _configured_trainer(
            source, operator, train_path, test_path, model_dir)
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", prefix="midlayer_train_",
                dir=PIPEWEAVE_ROOT, delete=False) as stream:
            stream.write(configured)
            temporary_script = Path(stream.name)
        try:
            subprocess.run(
                [sys.executable, str(temporary_script)],
                cwd=PIPEWEAVE_ROOT,
                check=True,
            )
        finally:
            temporary_script.unlink()
        trained.append(operator)

    print(f"trained {', '.join(trained)} for {args.hardware_id} → {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
