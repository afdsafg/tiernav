"""Verify AEQA result output matches Pred-EQA reference format.
Reference: /home/afdsafg/下载/new/实验结果/Pred-EQA_2026-06-25_qwen3-vl-flash/
"""
import json
import pickle
import tempfile
import os
import numpy as np
from src.logger_aeqa import Logger


def _make_logger(tmpdir, n_total=2):
    return Logger(
        output_dir=tmpdir,
        start_ratio=0.0,
        end_ratio=1.0,
        n_total_questions=n_total,
        voxel_size=0.1,
    )


def test_failed_episode_still_records_answer():
    """gpt_answer must be recorded even when success=False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = _make_logger(tmpdir)
        logger.log_episode_result(
            success=False,
            question_id="qid-fail-1",
            explore_dist=0.0,
            gpt_answer="best guess answer",
            n_filtered_snapshots=2,
            n_total_snapshots=5,
            n_total_frames=20,
        )
        logger.save_results()
        with open(os.path.join(tmpdir, "gpt_answer_0.0_1.0.json")) as f:
            data = json.load(f)
        assert len(data) == 1, f"expected 1 entry, got {len(data)}"
        assert data[0]["question_id"] == "qid-fail-1"
        assert data[0]["answer"] == "best guess answer"


def test_path_length_recorded_for_all_episodes():
    """path_length_list must include failed episodes (with 0.0)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = _make_logger(tmpdir, n_total=2)
        logger.log_episode_result(
            success=True, question_id="qid-ok",
            explore_dist=12.5, gpt_answer="ok answer",
            n_filtered_snapshots=3, n_total_snapshots=6, n_total_frames=30,
        )
        logger.log_episode_result(
            success=False, question_id="qid-fail",
            explore_dist=0.0, gpt_answer="guess",
            n_filtered_snapshots=1, n_total_snapshots=2, n_total_frames=10,
        )
        logger.save_results()
        with open(os.path.join(tmpdir, "path_length_list_0.0_1.0.pkl"), "rb") as f:
            pl = pickle.load(f)
        assert "qid-ok" in pl and pl["qid-ok"] == 12.5
        assert "qid-fail" in pl and pl["qid-fail"] == 0.0


def test_snapshot_counts_recorded():
    """n_filtered_snapshots and n_total_snapshots must be real, not hardcoded 0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = _make_logger(tmpdir)
        logger.log_episode_result(
            success=True, question_id="qid-1",
            explore_dist=5.0, gpt_answer="a",
            n_filtered_snapshots=4, n_total_snapshots=10, n_total_frames=42,
        )
        logger.save_results()
        with open(os.path.join(tmpdir, "n_filtered_snapshots_0.0_1.0.json")) as f:
            nf = json.load(f)
        with open(os.path.join(tmpdir, "n_total_snapshots_0.0_1.0.json")) as f:
            nt = json.load(f)
        assert nf["qid-1"] == 4
        assert nt["qid-1"] == 10


def test_logger_init_tolerates_partial_prior_run():
    """Logger.__init__ must not crash if split files are inconsistent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a partial/inconsistent prior state for the same ratio
        with open(os.path.join(tmpdir, "success_list_0.0_1.0.pkl"), "wb") as f:
            pickle.dump(["qid-x"], f)
        # No matching gpt_answer_0.0_1.0.json — this is the inconsistency
        # Should warn + reset, not raise
        logger = _make_logger(tmpdir)
        assert hasattr(logger, "success_list")
