#!/usr/bin/env python3
"""Offline tuner for QuACK grouped LoRA tile heuristics on the ART layer bench."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import gc
import itertools
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ART_SRC_ROOT = REPO_ROOT / "src"


def _resolve_art_harness_root() -> Path:
    for candidate in REPO_ROOT.parents:
        maybe_root = candidate / "projects" / "art_harness"
        if maybe_root.is_dir():
            return maybe_root
    raise RuntimeError(
        "Unable to locate projects/art_harness from the current worktree."
    )


ART_HARNESS_ROOT = _resolve_art_harness_root()

if str(ART_HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(ART_HARNESS_ROOT))

import art_harness.layer_benches.bench_moe_lora as bench

ENV_NAMES = {
    "proj_tile_n": "ART_QUACK_PROJ_TILE_N",
    "matmul_tile_n": "ART_QUACK_MATMUL_TILE_N",
    "grad_a_tile_m": "ART_QUACK_GRAD_A_TILE_M",
    "grad_b_tile_m": "ART_QUACK_GRAD_B_TILE_M",
}


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"Expected at least one integer in '{raw}'")
    for value in values:
        if value <= 0:
            raise ValueError(f"Expected positive integers, got {value}")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune QuACK grouped LoRA tile heuristics against the ART layer bench."
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--ffn-hidden-size", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--proj-tile-n", type=str, default="32,64,128")
    parser.add_argument("--matmul-tile-n", type=str, default="64,128")
    parser.add_argument("--grad-a-tile-m", type=str, default="64,128")
    parser.add_argument("--grad-b-tile-m", type=str, default="64,128")
    parser.add_argument("--top-results", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


@contextmanager
def _tile_env(config: dict[str, int]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in ENV_NAMES.values()}
    try:
        for key, value in config.items():
            os.environ[ENV_NAMES[key]] = str(value)
        yield
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value


def _run_config(args: argparse.Namespace, config: dict[str, int]) -> dict[str, Any]:
    bench.ART_WORKTREE_SRC = ART_SRC_ROOT
    with _tile_env(config):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        result = bench.benchmark(args)
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
    return {
        "config": config,
        "timing_ms": result["timing_ms"],
        "timed_module_breakdown_ms": result["timed_module_breakdown_ms"],
        "flops": result["flops"],
        "peak_memory_gib": {
            "allocated": peak_alloc / (1024**3),
            "reserved": peak_reserved / (1024**3),
        },
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for QuACK tile tuning.")

    cli = _parse_args()
    bench_args = argparse.Namespace(
        batch=cli.batch,
        seq_len=cli.seq_len,
        hidden_size=cli.hidden_size,
        ffn_hidden_size=cli.ffn_hidden_size,
        num_experts=cli.num_experts,
        top_k=cli.top_k,
        lora_rank=cli.lora_rank,
        dtype=cli.dtype,
        warmup=cli.warmup,
        iters=cli.iters,
        peak_tflops=None,
        seed=cli.seed,
    )

    configs: list[dict[str, int]] = []
    for proj_tile_n, matmul_tile_n, grad_a_tile_m, grad_b_tile_m in itertools.product(
        _parse_csv_ints(cli.proj_tile_n),
        _parse_csv_ints(cli.matmul_tile_n),
        _parse_csv_ints(cli.grad_a_tile_m),
        _parse_csv_ints(cli.grad_b_tile_m),
    ):
        configs.append(
            {
                "proj_tile_n": proj_tile_n,
                "matmul_tile_n": matmul_tile_n,
                "grad_a_tile_m": grad_a_tile_m,
                "grad_b_tile_m": grad_b_tile_m,
            }
        )

    results: list[dict[str, Any]] = []
    for config in configs:
        try:
            payload = _run_config(bench_args, config)
        except Exception as exc:
            payload = {"config": config, "error": repr(exc)}
        results.append(payload)
        print(json.dumps(payload, sort_keys=True), flush=True)

    successful = [item for item in results if "timing_ms" in item]
    successful.sort(key=lambda item: float(item["timing_ms"]["total_mean"]))
    summary = {
        "search_space": {
            "proj_tile_n": _parse_csv_ints(cli.proj_tile_n),
            "matmul_tile_n": _parse_csv_ints(cli.matmul_tile_n),
            "grad_a_tile_m": _parse_csv_ints(cli.grad_a_tile_m),
            "grad_b_tile_m": _parse_csv_ints(cli.grad_b_tile_m),
        },
        "benchmark_config": vars(bench_args),
        "top_results": successful[: cli.top_results],
        "num_successful": len(successful),
        "num_total": len(results),
    }
    if cli.output_json is not None:
        cli.output_json.parent.mkdir(parents=True, exist_ok=True)
        cli.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
