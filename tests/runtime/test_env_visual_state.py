import base64
import io

import numpy as np
from PIL import Image

from src.tiernav_runtime.contracts import EpisodeState, Observation
from src.tiernav_runtime.env import RuntimeEnvironmentService


class FakeSnapshot:
    image = "rgb-0"
    cluster = [1, 2]


class FakeScene:
    snapshots = {"rgb-0": FakeSnapshot()}
    all_observations = {
        "rgb-0": np.full((3, 4, 3), 64, dtype=np.uint8),
    }
    objects = {
        1: {"class_name": "oven"},
        2: {"class_name": "towel"},
    }


class SnapshotImageScene(FakeScene):
    snapshots = {"snapshot-key": FakeSnapshot()}
    all_observations = {
        "rgb-0": np.full((7, 5, 3), 80, dtype=np.uint8),
    }


class FakeFrontier:
    frontier_id = 7
    image = "frontier-7.png"
    feature = np.full((2, 3, 3), 128, dtype=np.uint8)


class FakeTSDF:
    frontiers = [FakeFrontier()]


class RecordingTSDF(FakeTSDF):
    def __init__(self) -> None:
        self.frontiers = []
        self.calls = []

    def update_frontier_map(self, pts, cfg, scene, cnt_step, save_frontier_image=False):
        self.calls.append((pts, cfg, scene, cnt_step, save_frontier_image))
        self.frontiers = [FakeFrontier()]
        return True


class PanoramaExecutor:
    def __init__(self) -> None:
        self._pts = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self._angle = 0.5
        self._path_length = 0.0
        self.cfg = type("Cfg", (), {"planner": object()})()

    def explore_panorama(self):
        return type(
            "Evidence",
            (),
            {
                "current_image_b64": _image_b64(width=13, height=9, value=144),
                "progress": "Panorama complete",
            },
        )()


def _image_b64(width: int, height: int, value: int = 96) -> str:
    img = Image.fromarray(np.full((height, width, 3), value, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decoded_size(image_b64: str) -> tuple[int, int]:
    with Image.open(io.BytesIO(base64.b64decode(image_b64))) as img:
        return img.size


def test_environment_builds_real_aeqa_visual_state_from_scene_and_tsdf():
    env = RuntimeEnvironmentService.for_aeqa(
        scene=FakeScene(),
        tsdf_planner=FakeTSDF(),
        executor=None,
    )
    episode = EpisodeState(
        episode_id="ep-visual",
        scene_id="scene-1",
        task_name="aeqa",
        task_mode="question_answering",
        prompt="What is hanging on the oven handle?",
        step_index=3,
        last_observation=Observation(
            summary="Arrived near the kitchen frontier.",
            raw={"current_image_b64": _image_b64(width=5, height=7, value=160)},
        ),
    )

    state = env.get_aeqa_visual_state(episode)

    assert state["question"] == "What is hanging on the oven handle?"
    assert state["current_step"] == 3
    assert state["snapshots"][0]["image_id"] == "rgb-0"
    assert state["snapshots"][0]["image_b64"]
    assert "oven" in state["snapshots"][0]["label"]
    assert "towel" in state["snapshots"][0]["label"]
    assert state["frontiers"][0]["frontier_id"] == "7"
    assert state["frontiers"][0]["image_b64"]
    assert _decoded_size(state["egocentric_views"][0]["image_b64"]) == (360, 360)
    assert "Arrived near the kitchen frontier." in state["tool_feedback"]


def test_environment_visual_state_uses_snapshot_image_observation_key():
    env = RuntimeEnvironmentService.for_aeqa(
        scene=SnapshotImageScene(),
        tsdf_planner=FakeTSDF(),
        executor=None,
    )
    episode = EpisodeState(
        episode_id="ep-snapshot-image",
        scene_id="scene-1",
        task_name="aeqa",
        task_mode="question_answering",
        prompt="What is hanging on the oven handle?",
    )

    state = env.get_aeqa_visual_state(episode)

    assert state["snapshots"][0]["image_id"] == "snapshot-key"
    assert _decoded_size(state["snapshots"][0]["image_b64"]) == (360, 360)


def test_environment_visual_state_skips_plain_string_image_ids():
    class PlainStringScene(FakeScene):
        snapshots = {"snapshot-key": type("Snapshot", (), {"image": "step0_view0", "cluster": []})()}
        all_observations = {}

    env = RuntimeEnvironmentService.for_aeqa(
        scene=PlainStringScene(),
        tsdf_planner=object(),
        executor=None,
    )
    episode = EpisodeState(
        episode_id="ep-plain-string",
        scene_id="scene-1",
        task_name="aeqa",
        task_mode="question_answering",
        prompt="What is hanging on the oven handle?",
        last_observation=Observation(raw={"current_image_b64": "step0_view0"}),
    )

    state = env.get_aeqa_visual_state(episode)

    assert state["snapshots"] == []
    assert state["egocentric_views"] == []


def test_environment_visual_state_resizes_uploaded_images_to_360_square():
    current_view_b64 = _image_b64(width=19, height=11, value=192)
    env = RuntimeEnvironmentService.for_aeqa(
        scene=FakeScene(),
        tsdf_planner=FakeTSDF(),
        executor=None,
    )
    episode = EpisodeState(
        episode_id="ep-visual-360",
        scene_id="scene-1",
        task_name="aeqa",
        task_mode="question_answering",
        prompt="What is hanging on the oven handle?",
        last_observation=Observation(raw={"current_image_b64": current_view_b64}),
    )

    state = env.get_aeqa_visual_state(episode)

    assert _decoded_size(state["snapshots"][0]["image_b64"]) == (360, 360)
    assert _decoded_size(state["frontiers"][0]["image_b64"]) == (360, 360)
    assert _decoded_size(state["egocentric_views"][0]["image_b64"]) == (360, 360)


def test_environment_initial_visual_context_seeds_frontier_map():
    tsdf = RecordingTSDF()
    executor = PanoramaExecutor()
    env = RuntimeEnvironmentService.for_aeqa(
        scene=FakeScene(),
        tsdf_planner=tsdf,
        executor=executor,
    )

    env.initialize_aeqa_visual_context()

    assert len(tsdf.calls) == 1
    assert tsdf.calls[0][0] is executor._pts
    assert tsdf.calls[0][2] is env.scene
    assert tsdf.calls[0][4] is False
    assert env.current_pose == {"x": 1.0, "y": 2.0, "z": 3.0, "theta": 0.5}

    state = env.get_aeqa_visual_state(
        EpisodeState(
            episode_id="ep-frontier",
            scene_id="scene-1",
            task_name="aeqa",
            task_mode="question_answering",
            prompt="Where should I look next?",
        )
    )
    assert state["frontiers"][0]["frontier_id"] == "7"
    assert _decoded_size(state["egocentric_views"][0]["image_b64"]) == (360, 360)
