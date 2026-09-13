import pytest

from art import Trajectory, TrajectoryGroup
from art.metrics_taxonomy import (
    TRAIN_GRADIENT_STEPS_KEY,
    TrajectoryBatchSummary,
    average_metric_samples,
    build_training_summary_metrics,
    summarize_trajectory_groups,
)


def test_average_metric_samples_handles_sparse_keys() -> None:
    averaged = average_metric_samples(
        [
            {"loss/train": 1.0, "loss/grad_norm": 0.5},
            {"loss/train": 0.5},
            {"loss/grad_norm": 1.0},
        ]
    )

    assert averaged["loss/train"] == pytest.approx(0.75)
    assert averaged["loss/grad_norm"] == pytest.approx(0.75)


def test_build_training_summary_metrics_only_includes_data_section() -> None:
    summary = TrajectoryBatchSummary(
        num_scenarios=2,
        num_trajectories=5,
        num_groups_submitted=2,
        num_groups_trainable=1,
        scenario_ids=["a", "b"],
    )

    metrics = build_training_summary_metrics(
        summary,
        include_trainable_groups=True,
    )

    assert metrics["data/step_num_scenarios"] == pytest.approx(2.0)
    assert metrics["data/step_num_groups_trainable"] == pytest.approx(1.0)
    assert metrics["data/step_num_groups_submitted"] == pytest.approx(2.0)
    assert metrics["data/step_num_trajectories"] == pytest.approx(5.0)


def test_average_metric_samples_requires_invariant_gradient_step_count() -> None:
    with pytest.raises(ValueError, match="must be invariant"):
        average_metric_samples(
            [
                {TRAIN_GRADIENT_STEPS_KEY: 2.0},
                {TRAIN_GRADIENT_STEPS_KEY: 3.0},
            ]
        )


def test_summarize_trajectory_groups_only_counts_explicit_scenario_id() -> None:
    summary = summarize_trajectory_groups(
        [
            TrajectoryGroup(
                trajectories=[
                    Trajectory(
                        reward=1.0,
                        messages_and_choices=[{"role": "user", "content": "a"}],
                    )
                ],
                metadata={"scenario_id": "scenario-1"},
            ),
            TrajectoryGroup(
                trajectories=[
                    Trajectory(
                        reward=0.0,
                        messages_and_choices=[{"role": "user", "content": "b"}],
                    )
                ],
                metadata={"scenario_scenario_id": "legacy-scenario"},
            ),
        ]
    )

    assert summary.scenario_ids == ["scenario-1"]
