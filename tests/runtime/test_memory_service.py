"""Tests for the spatial memory service."""
from __future__ import annotations

import pytest

from src.tiernav_runtime.contracts import (
    MemoryPack,
    MemoryScope,
    Observation,
)
from src.tiernav_runtime.memory import (
    HypothesisNode,
    MemoryService,
    MemorySession,
    ObjectNode,
    RoomNode,
    SnapshotNode,
)


def _observation(
    *,
    summary: str = "",
    image_ids: list[str] | None = None,
    object_ids: list[str] | None = None,
    room_id: str | None = None,
) -> Observation:
    return Observation(
        summary=summary,
        image_ids=list(image_ids or []),
        object_ids=list(object_ids or []),
        room_id=room_id,
    )


# --- Plan examples ---------------------------------------------------------


def test_memory_updates_room_snapshot_object_layers():
    mem = MemoryService(enabled=True)
    obs = _observation(
        summary="sofa near the window",
        image_ids=["img-1", "img-2"],
        object_ids=["obj-sofa", "obj-window"],
        room_id="room-A",
    )
    mem.update_from_observation(obs, action_type="explore_frontier", round_index=0)

    # room layer
    assert "room-A" in mem.rooms
    room = mem.rooms["room-A"]
    assert isinstance(room, RoomNode)
    assert "snap-img-1" in room.snapshot_ids
    assert "snap-img-2" in room.snapshot_ids

    # snapshot layer
    assert "snap-img-1" in mem.snapshots
    snap = mem.snapshots["snap-img-1"]
    assert isinstance(snap, SnapshotNode)
    assert snap.room_id == "room-A"
    assert snap.summary == "sofa near the window"
    assert snap.round_index == 0
    assert snap.action_type == "explore_frontier"
    # snapshot records the object_ids present in the observation
    assert "obj-sofa" in snap.object_ids
    assert "obj-window" in snap.object_ids

    # object layer
    assert "obj-sofa" in mem.objects
    sofa = mem.objects["obj-sofa"]
    assert isinstance(sofa, ObjectNode)
    assert sofa.room_id == "room-A"
    assert "snap-img-1" in sofa.snapshot_ids
    assert "snap-img-2" in sofa.snapshot_ids


def test_memory_query_returns_context_ready_pack():
    mem = MemoryService(enabled=True)
    obs = _observation(
        summary="a red sofa beside the window",
        image_ids=["img-1"],
        object_ids=["obj-sofa"],
        room_id="room-A",
    )
    mem.update_from_observation(obs, action_type="explore_frontier", round_index=0)

    pack = mem.query("sofa")
    assert isinstance(pack, MemoryPack)
    assert pack.query == "sofa"
    # matching snapshot -> summary contains snapshot summary, evidence has snapshot id
    assert "red sofa" in pack.summary
    assert "snap-img-1" in pack.evidence_ids
    assert pack.reuse_hint != ""


def test_memory_can_be_disabled_without_crashing():
    mem = MemoryService(enabled=False)
    obs = _observation(
        summary="sofa",
        image_ids=["img-1"],
        object_ids=["obj-sofa"],
        room_id="room-A",
    )
    # none of these may raise
    mem.update_from_observation(obs, action_type="explore_frontier", round_index=0)
    mem.add_hypothesis("h1", "the sofa is in room-A")
    mem.support_hypothesis("h1", "saw sofa here")
    mem.contradict_hypothesis("h1", "sofa moved")
    pack = mem.query("sofa")
    assert isinstance(pack, MemoryPack)
    assert pack.summary == ""
    assert pack.evidence_ids == []


def test_memory_records_hypothesis_support_and_contradiction():
    mem = MemoryService(enabled=True)
    mem.update_from_observation(
        _observation(
            summary="sofa",
            image_ids=["img-1"],
            object_ids=["obj-sofa"],
            room_id="room-A",
        ),
        action_type="explore_frontier",
        round_index=0,
    )
    mem.add_hypothesis("h1", "the sofa is in room-A")
    mem.support_hypothesis("h1", "snapshot shows sofa in room-A")
    mem.contradict_hypothesis("h1", "later snapshot shows no sofa")

    pack = mem.query("sofa")
    assert "snapshot shows sofa in room-A" in pack.supports
    assert "later snapshot shows no sofa" in pack.contradictions


# --- Extra coverage --------------------------------------------------------


def test_repeated_update_does_not_duplicate_snapshot_ids():
    mem = MemoryService(enabled=True)
    obs = _observation(
        summary="sofa",
        image_ids=["img-1"],
        object_ids=["obj-sofa"],
        room_id="room-A",
    )
    mem.update_from_observation(obs, action_type="explore_frontier", round_index=0)
    mem.update_from_observation(obs, action_type="explore_frontier", round_index=1)

    room = mem.rooms["room-A"]
    assert room.snapshot_ids.count("snap-img-1") == 1
    obj = mem.objects["obj-sofa"]
    assert obj.snapshot_ids.count("snap-img-1") == 1
    # snapshot itself not duplicated
    assert len(mem.snapshots) == 1


def test_missing_room_id_uses_unknown():
    mem = MemoryService(enabled=True)
    obs = _observation(
        summary="sofa",
        image_ids=["img-1"],
        object_ids=["obj-sofa"],
        room_id=None,
    )
    mem.update_from_observation(obs, action_type="explore_frontier", round_index=0)

    assert "unknown" in mem.rooms
    assert mem.snapshots["snap-img-1"].room_id == "unknown"
    assert mem.objects["obj-sofa"].room_id == "unknown"


def test_no_direct_query_match_falls_back_to_existing_snapshots():
    mem = MemoryService(enabled=True)
    mem.update_from_observation(
        _observation(
            summary="a kitchen counter with fruits",
            image_ids=["img-1"],
            object_ids=["obj-counter"],
            room_id="room-K",
        ),
        action_type="explore_frontier",
        round_index=0,
    )
    # query that does not keyword-match the summary
    pack = mem.query("garage")
    assert pack.evidence_ids  # fallback returned some snapshot
    assert "snap-img-1" in pack.evidence_ids
    # fallback is not a direct hit: confidence must be downgraded so downstream
    # consumers can tell this is a best-effort reuse, not a real match.
    assert pack.confidence == 0.0


def test_direct_query_match_has_high_confidence():
    mem = MemoryService(enabled=True)
    mem.update_from_observation(
        _observation(
            summary="a red sofa beside the window",
            image_ids=["img-1"],
            object_ids=["obj-sofa"],
            room_id="room-A",
        ),
        action_type="explore_frontier",
        round_index=0,
    )
    pack = mem.query("sofa")
    assert "snap-img-1" in pack.evidence_ids
    # direct keyword match -> high confidence, strictly above fallback's 0.0
    assert pack.confidence > 0.0


def test_fallback_reuse_hint_signals_best_effort():
    mem = MemoryService(enabled=True)
    mem.update_from_observation(
        _observation(
            summary="a kitchen counter with fruits",
            image_ids=["img-1"],
            object_ids=["obj-counter"],
            room_id="room-K",
        ),
        action_type="explore_frontier",
        round_index=0,
    )
    pack = mem.query("garage")
    assert pack.evidence_ids
    # downstream must be able to distinguish fallback from direct hit via hint
    assert "fallback" in pack.reuse_hint.lower()


def test_disabled_memory_does_not_record_hypothesis():
    mem = MemoryService(enabled=False)
    mem.add_hypothesis("h1", "the sofa is in room-A")
    mem.support_hypothesis("h1", "saw sofa here")
    mem.contradict_hypothesis("h1", "sofa moved")
    assert mem.hypotheses == {}


# --- Node model sanity -----------------------------------------------------


def test_node_models_use_runtime_model_with_default_factory_lists():
    """Mutable list defaults must be isolated per instance (Field(default_factory=list))."""
    room = RoomNode(room_id="r")
    snap = SnapshotNode(snapshot_id="s", room_id="r")
    obj = ObjectNode(object_id="o", room_id="r")
    hyp = HypothesisNode(hypothesis_id="h", text="t")
    for node, field in [
        (room, "snapshot_ids"),
        (snap, "object_ids"),
        (obj, "snapshot_ids"),
        (hyp, "supports"),
        (hyp, "contradictions"),
    ]:
        assert getattr(node, field) == []
        # mutate one instance; a freshly constructed peer must not share state
        getattr(node, field).append("x")
        peer = type(node)(**{f: getattr(node, f) for f in type(node).model_fields if f != field})
        assert "x" not in getattr(peer, field)


def test_query_with_no_snapshots_returns_empty_pack():
    mem = MemoryService(enabled=True)
    pack = mem.query("anything")
    assert isinstance(pack, MemoryPack)
    assert pack.summary == ""
    assert pack.evidence_ids == []
    assert pack.reuse_hint == ""


def test_evidence_only_uses_real_observation_ids():
    """No fabricated evidence: evidence_ids are snapshot ids derived from image_ids."""
    mem = MemoryService(enabled=True)
    mem.update_from_observation(
        _observation(
            summary="sofa",
            image_ids=["img-1", "img-2"],
            object_ids=["obj-sofa"],
            room_id="room-A",
        ),
        action_type="explore_frontier",
        round_index=0,
    )
    pack = mem.query("sofa")
    for eid in pack.evidence_ids:
        assert eid.startswith("snap-")
        assert eid.replace("snap-", "", 1) in {"img-1", "img-2"}


# --- MemorySession: AEQA per-question reset --------------------------------


def _obs_with(summary: str, image_id: str = "img-1") -> Observation:
    return _observation(
        summary=summary,
        image_ids=[image_id],
        object_ids=["obj-sofa"],
        room_id="room-A",
    )


def test_aeqa_memory_resets_per_question():
    """PER_QUESTION scope: each start_session yields a fresh MemoryService.

    An observation written under q1 must NOT be visible after start_session
    for q2 — even within the same episode.
    """
    session = MemorySession(scope=MemoryScope.PER_QUESTION)

    mem_q1 = session.start_session(episode_id="ep-1", question_id="q1")
    mem_q1.update_from_observation(
        _obs_with("sofa near window"), action_type="explore_frontier", round_index=0
    )
    assert mem_q1.query("sofa").evidence_ids  # sanity: q1 has the observation

    mem_q2 = session.start_session(episode_id="ep-1", question_id="q2")
    pack = mem_q2.query("sofa")
    # Fresh memory: no evidence carried over from q1.
    assert pack.evidence_ids == []
    assert pack.summary == ""
    # The previously active service is detached: mutating it must not leak.
    mem_q1.update_from_observation(
        _obs_with("later sofa", image_id="img-late"),
        action_type="explore_frontier",
        round_index=0,
    )
    assert session.current_memory.query("sofa").evidence_ids == []


def test_aeqa_session_current_memory_matches_returned_service():
    session = MemorySession(scope=MemoryScope.PER_QUESTION)
    mem = session.start_session(episode_id="ep-1", question_id="q1")
    assert session.current_memory is mem


# --- MemorySession: GOATBench cross-subtask persistence ---------------------


def test_goatbench_memory_persists_across_subtasks():
    """SUBTASK_SEQUENCE scope: same episode -> same MemoryService across subtasks.

    An observation written under subtask 0 must remain queryable under subtask 1.
    """
    session = MemorySession(scope=MemoryScope.SUBTASK_SEQUENCE)

    mem0 = session.start_session(episode_id="ep-1", subtask_index=0)
    mem0.update_from_observation(
        _obs_with("sofa near window", image_id="img-0"),
        action_type="explore_frontier",
        round_index=0,
    )
    assert mem0.query("sofa").evidence_ids

    mem1 = session.start_session(episode_id="ep-1", subtask_index=1)
    # Same underlying MemoryService instance: persisted across subtasks.
    assert mem1 is mem0
    pack = mem1.query("sofa")
    assert "snap-img-0" in pack.evidence_ids
    assert "sofa near window" in pack.summary


def test_goatbench_memory_resets_on_new_episode():
    """SUBTASK_SEQUENCE: a new episode_id resets the MemoryService."""
    session = MemorySession(scope=MemoryScope.SUBTASK_SEQUENCE)

    mem_ep1 = session.start_session(episode_id="ep-1", subtask_index=0)
    mem_ep1.update_from_observation(
        _obs_with("sofa near window", image_id="img-0"),
        action_type="explore_frontier",
        round_index=0,
    )
    assert mem_ep1.query("sofa").evidence_ids

    mem_ep2 = session.start_session(episode_id="ep-2", subtask_index=0)
    assert mem_ep2 is not mem_ep1
    pack = mem_ep2.query("sofa")
    assert pack.evidence_ids == []
    assert pack.summary == ""


def test_goatbench_session_holds_notebook_and_scene_graph_across_subtasks():
    """GOATBench reuses Notebook and SceneGraphMemory across subtasks in one episode.

    The session owns these instances; subtask transitions within the same
    episode must NOT recreate them.
    """
    notebook = object()  # duck-typed placeholder; session just holds it
    scene_graph = object()
    session = MemorySession(
        scope=MemoryScope.SUBTASK_SEQUENCE,
        notebook=notebook,
        scene_graph=scene_graph,
    )

    session.start_session(episode_id="ep-1", subtask_index=0)
    assert session.notebook is notebook
    assert session.scene_graph is scene_graph

    session.start_session(episode_id="ep-1", subtask_index=1)
    # Same instances persist across subtasks within the episode.
    assert session.notebook is notebook
    assert session.scene_graph is scene_graph

    # New episode: memory resets, but the held notebook/scene_graph instances
    # are owned by the session and persist for the session's lifetime. Task 8
    # decides whether to replace them per episode; the session itself does not.
    session.start_session(episode_id="ep-2", subtask_index=0)
    assert session.notebook is notebook
    assert session.scene_graph is scene_graph


def test_aeqa_session_does_not_hold_notebook_or_scene_graph():
    """PER_QUESTION scope has no cross-question reuse; notebook/scene_graph stay None."""
    session = MemorySession(scope=MemoryScope.PER_QUESTION)
    assert session.notebook is None
    assert session.scene_graph is None
    session.start_session(episode_id="ep-1", question_id="q1")
    assert session.notebook is None
    assert session.scene_graph is None


# --- Memory bridge: real observations flow into the active service ----------


def test_session_update_from_observation_updates_active_memory_layers():
    """The session delegates update_from_observation to its active MemoryService,
    bridging real observations into room/snapshot/object layers."""
    session = MemorySession(scope=MemoryScope.PER_QUESTION)
    session.start_session(episode_id="ep-1", question_id="q1")

    session.update_from_observation(
        _obs_with("sofa near window", image_id="img-1"),
        action_type="explore_frontier",
        round_index=0,
    )
    mem = session.current_memory
    assert "room-A" in mem.rooms
    assert "snap-img-1" in mem.snapshots
    assert "obj-sofa" in mem.objects


def test_session_query_delegates_to_active_memory():
    session = MemorySession(scope=MemoryScope.PER_QUESTION)
    session.start_session(episode_id="ep-1", question_id="q1")
    session.update_from_observation(
        _obs_with("sofa near window", image_id="img-1"),
        action_type="explore_frontier",
        round_index=0,
    )
    pack = session.query("sofa")
    assert isinstance(pack, MemoryPack)
    assert "snap-img-1" in pack.evidence_ids
