from napari_epicycle.io_registry import CurationSessionService, FileArtifactRegistry


def test_curation_session_lifecycle_and_persist(tmp_path):
    registry = FileArtifactRegistry()
    svc = CurationSessionService(registry)

    svc.start_session(source_artifact="hitl_slice:stack01_XY_10", workflow_name="sam_refine")
    svc.record_accept(slice_label="stack01_XY_10", label_id=3, voxel_count=120)
    svc.record_reject(slice_label="stack01_XY_10", reason="manual_reject")
    svc.record_handoff(to_workflow="scribble_refine")
    svc.finalize(final_label_id_map={"3": 3})

    saved = svc.persist_active(str(tmp_path))
    assert saved is not None
    assert (tmp_path / "artifacts" / "curation" / "latest_session.json").exists()

    payload = svc.snapshot()
    assert payload is not None
    assert payload["workflows_used"][-1] == "scribble_refine"
    assert len(payload["accept_events"]) == 1
    assert len(payload["reject_events"]) == 1
    assert payload["final_label_id_map"]["3"] == 3
