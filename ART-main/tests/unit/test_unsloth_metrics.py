import asyncio
from collections import defaultdict

from art.unsloth.train import get_log_fn


class _DummyTrainer:
    def __init__(self) -> None:
        self._metrics = {"train": defaultdict(list)}


def test_get_log_fn_routes_eval_metrics_to_val_namespace() -> None:
    trainer = _DummyTrainer()
    trainer._metrics["train"]["loss/train"].append(1.5)
    trainer._metrics["train"]["loss/entropy"].append(0.2)
    results_queue: asyncio.Queue[dict[str, float]] = asyncio.Queue()

    log = get_log_fn(trainer, results_queue)
    log({"eval_loss": 1.0, "eval_runtime": 2.0})

    assert results_queue.get_nowait() == {
        "val/loss/train": 1.0,
        "val/loss/entropy": 0.2,
        "val/runtime": 2.0,
    }
