"""Integration tests for experiment runner."""

from pathlib import Path
import pytest
from experiments.runner import ExperimentRunner


def test_run_exp001_rain(tmp_path):
    config_path = "experiments/configs/EXP001_rain.yaml"
    runner = ExperimentRunner(config_path=config_path, output_root=str(tmp_path))
    summary = runner.run()

    assert summary["experiment_id"] == "EXP001_rain"
    assert summary["num_views"] == 5
    assert summary["overall_status"] in ("PASS", "REJECT")
    assert "mean_bbox_iou" in summary["aggregate_metrics"]
    assert "mean_edge_similarity" in summary["aggregate_metrics"]

    exp_dir = tmp_path / "EXP001_rain"
    assert (exp_dir / "summary.json").exists()
    assert (exp_dir / "summary.md").exists()
    assert (exp_dir / "view_01_orig.png").exists()
    assert (exp_dir / "view_01_aug.png").exists()


def test_run_exp005_ablation(tmp_path):
    config_path = "experiments/configs/EXP005_ablation.yaml"
    runner = ExperimentRunner(config_path=config_path, output_root=str(tmp_path))
    summary = runner.run()

    assert summary["experiment_id"] == "EXP005_ablation"
    assert summary["num_views"] == 5
    assert (tmp_path / "EXP005_ablation" / "summary.json").exists()
