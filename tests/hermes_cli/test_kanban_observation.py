"""External observation cards (``create --observation``).

Observation cards mirror work done by an *external* agent (Claude Code /
Codex adapters): they are born ``running`` with no claim machinery and no
``task_runs`` row, must never become dispatchable, and are closed either by
an explicit ``complete`` or by the orphan TTL sweep (→ ``done``, never
``ready``).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban import run_slash


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_spawn(*args, **kwargs):
    return 12345


def _create_observation(conn, **kwargs):
    kwargs.setdefault("title", "external turn")
    kwargs.setdefault("assignee", "claude-code-external")
    kwargs.setdefault("tenant", "claude")
    kwargs.setdefault("created_by", "kanban-adapter")
    return kb.create_task(conn, observation=True, **kwargs)


def _events(conn, task_id):
    return [
        row["kind"]
        for row in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
            (task_id,),
        )
    ]


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------

def test_observation_created_running_with_no_claim_and_no_runs(kanban_home):
    conn = kb.connect()
    try:
        tid = _create_observation(conn)
        task = kb.get_task(conn, tid)
        assert task.observation is True
        assert task.status == "running"
        assert task.started_at is not None
        assert task.claim_lock is None
        assert task.claim_expires is None
        assert task.worker_pid is None
        assert task.current_run_id is None
        runs = conn.execute(
            "SELECT COUNT(*) AS n FROM task_runs WHERE task_id = ?", (tid,)
        ).fetchone()["n"]
        assert runs == 0
        created_payload = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'created'",
            (tid,),
        ).fetchone()["payload"]
        assert json.loads(created_payload)["observation"] is True
    finally:
        conn.close()


def test_observation_rejects_execution_semantic_options(kanban_home):
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="parent")
        rejected = [
            {"parents": (parent,)},
            {"triage": True},
            {"initial_status": "blocked"},
            {"goal_mode": True},
            {"goal_max_turns": 3},
            {"max_retries": 2},
            {"max_runtime_seconds": 60},
            {"skills": ["translation"]},
            {"model_override": "gpt-x"},
            {"model_override": "gpt-x", "provider_override": "openai"},
            {"project_id": "proj"},
            {"workspace_kind": "dir", "workspace_path": "/tmp/x"},
            {"workspace_kind": "worktree", "branch_name": "wt/x"},
        ]
        for kwargs in rejected:
            with pytest.raises(ValueError, match="observation"):
                _create_observation(conn, **kwargs)
        # Integration-required options stay allowed.
        tid = _create_observation(
            conn, idempotency_key="obs:allowed", session_id="sess-1", priority=1
        )
        assert kb.get_task(conn, tid).status == "running"
    finally:
        conn.close()


def test_observation_idempotency_key_returns_existing_card(kanban_home):
    conn = kb.connect()
    try:
        first = _create_observation(conn, idempotency_key="claude:s1:t1")
        second = _create_observation(conn, idempotency_key="claude:s1:t1")
        assert first == second
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE idempotency_key = ?",
            ("claude:s1:t1",),
        ).fetchone()["n"]
        assert count == 1
    finally:
        conn.close()


def test_observation_idempotency_key_rejects_worker_task_collision(kanban_home):
    conn = kb.connect()
    try:
        normal = kb.create_task(
            conn, title="normal", idempotency_key="shared:key"
        )
        with pytest.raises(ValueError, match="idempotency key.*worker"):
            _create_observation(conn, idempotency_key="shared:key")
        assert kb.get_task(conn, normal).status == "ready"

        observation = _create_observation(
            conn, idempotency_key="observation:key"
        )
        with pytest.raises(ValueError, match="idempotency key.*observation"):
            kb.create_task(
                conn, title="normal", idempotency_key="observation:key"
            )
        assert kb.get_task(conn, observation).status == "running"
    finally:
        conn.close()


def test_observation_idempotency_is_atomic_across_connections(kanban_home):
    def create_once(_: int) -> str:
        conn = kb.connect()
        try:
            return _create_observation(
                conn, idempotency_key="concurrent:observation"
            )
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=8) as pool:
        task_ids = list(pool.map(create_once, range(8)))

    assert len(set(task_ids)) == 1
    conn = kb.connect()
    try:
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE idempotency_key = ?",
            ("concurrent:observation",),
        ).fetchone()["n"] == 1
    finally:
        conn.close()


def test_normal_create_semantics_are_unchanged(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="normal", initial_status="running")
        task = kb.get_task(conn, tid)
        assert task.observation is False
        assert task.status == "ready"
        assert task.started_at is None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Dispatcher / recovery paths must never make an observation dispatchable
# ---------------------------------------------------------------------------

def test_dispatch_once_never_spawns_observation(kanban_home):
    spawned = []

    def spy_spawn(*args, **kwargs):
        spawned.append(args)
        return 12345

    conn = kb.connect()
    try:
        obs = _create_observation(conn)
        result = kb.dispatch_once(conn, spawn_fn=spy_spawn)
        assert spawned == []
        assert result.spawned == []
        assert kb.get_task(conn, obs).status == "running"
    finally:
        conn.close()


def test_running_observation_does_not_consume_in_progress_budget(kanban_home):
    conn = kb.connect()
    try:
        _create_observation(conn)
        normal = kb.create_task(conn, title="real work", assignee="default")
        result = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, max_in_progress=1,
        )
        assert [entry[0] for entry in result.spawned] == [normal]
    finally:
        conn.close()


def test_running_observation_does_not_consume_per_profile_budget(kanban_home):
    conn = kb.connect()
    try:
        _create_observation(conn, assignee="default")
        normal = kb.create_task(conn, title="real work", assignee="default")
        result = kb.dispatch_once(
            conn,
            spawn_fn=_fake_spawn,
            max_in_progress_per_profile=1,
        )
        assert [entry[0] for entry in result.spawned] == [normal]
    finally:
        conn.close()


def test_stale_and_recovery_paths_ignore_observations(kanban_home):
    conn = kb.connect()
    try:
        obs = _create_observation(conn)
        # Backdate far past every staleness threshold.
        conn.execute(
            "UPDATE tasks SET started_at = ? WHERE id = ?",
            (int(time.time()) - 10 * 3600, obs),
        )
        conn.commit()

        assert kb.release_stale_claims(conn) == 0
        assert kb.detect_stale_running(conn, stale_timeout_seconds=60) == []
        assert kb.detect_crashed_workers(conn) == []
        assert kb.enforce_max_runtime(conn) == []
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, obs).status == "running"
    finally:
        conn.close()


def test_recompute_ready_never_promotes_blocked_observation(kanban_home):
    conn = kb.connect()
    try:
        obs = _create_observation(conn)
        conn.execute(
            "UPDATE tasks SET status = 'blocked' WHERE id = ?", (obs,)
        )
        conn.commit()
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, obs).status == "blocked"
    finally:
        conn.close()


def test_migration_backfill_does_not_synthesize_runs_for_observations(kanban_home):
    conn = kb.connect()
    try:
        obs = _create_observation(conn)
        kb._migrate_add_optional_columns(conn)
        runs = conn.execute(
            "SELECT COUNT(*) AS n FROM task_runs WHERE task_id = ?", (obs,)
        ).fetchone()["n"]
        assert runs == 0
        assert kb.get_task(conn, obs).current_run_id is None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Worker-lifecycle mutations are rejected
# ---------------------------------------------------------------------------

def test_claim_is_rejected_even_if_observation_reached_ready(kanban_home):
    conn = kb.connect()
    try:
        obs = _create_observation(conn)
        assert kb.claim_task(conn, obs) is None
        # Simulate an external writer corrupting the status: claim must
        # still refuse instead of turning the card into a worker run.
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (obs,))
        conn.commit()
        assert kb.claim_task(conn, obs) is None
        assert kb.get_task(conn, obs).claim_lock is None
        runs = conn.execute(
            "SELECT COUNT(*) AS n FROM task_runs WHERE task_id = ?", (obs,)
        ).fetchone()["n"]
        assert runs == 0
    finally:
        conn.close()


def test_reclaim_is_rejected_and_never_yields_ready(kanban_home):
    conn = kb.connect()
    try:
        obs = _create_observation(conn)
        assert kb.reclaim_task(conn, obs) is False
        assert kb.get_task(conn, obs).status == "running"
    finally:
        conn.close()


def test_assign_and_reassign_are_rejected(kanban_home):
    conn = kb.connect()
    try:
        obs = _create_observation(conn)
        with pytest.raises(RuntimeError, match="observation"):
            kb.assign_task(conn, obs, "default")
        assert kb.reassign_task(conn, obs, "default") is False
        assert kb.get_task(conn, obs).assignee == "claude-code-external"
    finally:
        conn.close()


def test_promote_and_unblock_are_rejected(kanban_home):
    conn = kb.connect()
    try:
        obs = _create_observation(conn)
        conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (obs,))
        conn.commit()
        ok, reason = kb.promote_task(conn, obs, actor="tester")
        assert ok is False
        assert "observation" in reason
        assert kb.unblock_task(conn, obs) is False
        assert kb.get_task(conn, obs).status == "blocked"
    finally:
        conn.close()


def test_block_and_schedule_are_rejected(kanban_home):
    conn = kb.connect()
    try:
        obs = _create_observation(conn)
        assert kb.block_task(conn, obs, reason="must not strand observation") is False
        assert kb.schedule_task(conn, obs, reason="must not schedule observation") is False
        assert kb.get_task(conn, obs).status == "running"
    finally:
        conn.close()


def test_worker_lifecycle_helpers_cannot_mutate_observation(kanban_home):
    conn = kb.connect()
    try:
        obs = _create_observation(conn)
        before = kb.get_task(conn, obs)

        assert kb.heartbeat_claim(conn, obs, claimer="external") is False
        assert kb.heartbeat_worker(conn, obs, note="not a worker") is False
        assert kb._record_spawn_failure(conn, obs, "not a worker") is False
        kb._set_worker_pid(conn, obs, 12345)
        kb.set_workspace_path(conn, obs, "/tmp/worker")
        kb.set_branch_name(conn, obs, "worker/branch")
        with pytest.raises(RuntimeError, match="observation"):
            kb.set_model_override(conn, obs, "worker-model")

        after = kb.get_task(conn, obs)
        assert after.status == "running"
        assert after.claim_expires is None
        assert after.last_heartbeat_at is None
        assert after.worker_pid is None
        assert after.workspace_path is None
        assert after.branch_name is None
        assert after.model_override is None
        assert after.consecutive_failures == before.consecutive_failures == 0
        assert "heartbeat" not in _events(conn, obs)
        assert "spawned" not in _events(conn, obs)
    finally:
        conn.close()


def test_dashboard_direct_status_move_is_rejected(kanban_home):
    from plugins.kanban.dashboard import plugin_api

    conn = kb.connect()
    try:
        obs = _create_observation(conn)
        assert plugin_api._set_status_direct(conn, obs, "ready") is False
        assert plugin_api._set_status_direct(conn, obs, "todo") is False
        assert kb.get_task(conn, obs).status == "running"
    finally:
        conn.close()


def test_complete_closes_observation_as_done(kanban_home):
    conn = kb.connect()
    try:
        obs = _create_observation(conn)
        assert kb.complete_task(conn, obs, result="external turn finished")
        task = kb.get_task(conn, obs)
        assert task.status == "done"
        assert task.completed_at is not None
        assert task.current_run_id is None
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM task_runs WHERE task_id = ?", (obs,)
        ).fetchone()["n"] == 0

        assert kb.edit_completed_task_result(
            conn, obs, result="corrected external result"
        )
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM task_runs WHERE task_id = ?", (obs,)
        ).fetchone()["n"] == 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Orphan TTL expiry
# ---------------------------------------------------------------------------

def test_orphan_observation_expires_to_done_with_audit_event(kanban_home):
    conn = kb.connect()
    try:
        fresh = _create_observation(conn, idempotency_key="obs:fresh")
        orphan = _create_observation(conn, idempotency_key="obs:orphan")
        conn.execute(
            "UPDATE tasks SET started_at = ? WHERE id = ?",
            (int(time.time()) - 2 * kb.DEFAULT_OBSERVATION_TTL_SECONDS, orphan),
        )
        conn.commit()

        expired = kb.expire_orphan_observations(conn)

        assert expired == [orphan]
        task = kb.get_task(conn, orphan)
        assert task.status == "done"
        assert task.completed_at is not None
        assert "observation" in (task.result or "")
        assert "observation_expired" in _events(conn, orphan)
        assert kb.get_task(conn, fresh).status == "running"
        # Idempotent: a second sweep finds nothing.
        assert kb.expire_orphan_observations(conn) == []
    finally:
        conn.close()


def test_observation_ttl_config_override(kanban_home, monkeypatch):
    from hermes_cli import config

    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {"kanban": {"observation_ttl_seconds": 60}},
    )
    conn = kb.connect()
    try:
        obs = _create_observation(conn)
        conn.execute(
            "UPDATE tasks SET started_at = ? WHERE id = ?",
            (int(time.time()) - 120, obs),
        )
        conn.commit()
        assert kb.expire_orphan_observations(conn) == [obs]
        assert kb.get_task(conn, obs).status == "done"
    finally:
        conn.close()


def test_dispatch_once_sweeps_orphan_observations(kanban_home):
    conn = kb.connect()
    try:
        obs = _create_observation(conn)
        conn.execute(
            "UPDATE tasks SET started_at = ? WHERE id = ?",
            (int(time.time()) - 2 * kb.DEFAULT_OBSERVATION_TTL_SECONDS, obs),
        )
        conn.commit()
        result = kb.dispatch_once(conn, spawn_fn=_fake_spawn)
        assert result.observation_expired == [obs]
        task = kb.get_task(conn, obs)
        assert task.status == "done"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Legacy DB migration
# ---------------------------------------------------------------------------

def test_legacy_schema_gains_observation_column_with_zero_default(kanban_home):
    conn = kb.connect()
    try:
        # Rebuild ``tasks`` in its pre-observation shape (the base columns
        # every legacy DB had before the additive migrations ran).
        legacy = "t_legacy0001"
        conn.execute("DROP TABLE tasks")
        conn.execute(
            """
            CREATE TABLE tasks (
                id             TEXT PRIMARY KEY,
                title          TEXT NOT NULL,
                body           TEXT,
                assignee       TEXT,
                status         TEXT NOT NULL,
                priority       INTEGER DEFAULT 0,
                created_by     TEXT,
                created_at     INTEGER NOT NULL,
                started_at     INTEGER,
                completed_at   INTEGER,
                workspace_kind TEXT NOT NULL DEFAULT 'scratch',
                workspace_path TEXT,
                claim_lock     TEXT,
                claim_expires  INTEGER,
                tenant         TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO tasks (id, title, status, created_at) "
            "VALUES (?, 'pre-migration card', 'ready', 123)",
            (legacy,),
        )
        conn.commit()
        kb._migrate_add_optional_columns(conn)
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
        assert "observation" in cols
        task = kb.get_task(conn, legacy)
        assert task.observation is False
        raw = conn.execute(
            "SELECT observation FROM tasks WHERE id = ?", (legacy,)
        ).fetchone()["observation"]
        assert raw == 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_create_observation_emits_additive_json_field(kanban_home):
    out = run_slash(
        "create 'adapter turn' --observation --assignee claude-code-external "
        "--tenant claude --created-by kanban-adapter --json"
    )
    payload = json.loads(out)
    assert payload["observation"] is True
    assert payload["status"] == "running"
    assert payload["started_at"] is not None

    normal = json.loads(run_slash("create 'plain card' --json"))
    assert normal["observation"] is False


def test_cli_rejects_observation_with_execution_options(kanban_home):
    out = run_slash("create 'bad combo' --observation --goal")
    assert "observation" in out
    conn = kb.connect()
    try:
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE title = 'bad combo'"
        ).fetchone()["n"]
        assert count == 0
    finally:
        conn.close()
