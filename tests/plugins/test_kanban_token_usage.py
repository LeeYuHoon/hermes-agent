from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_token_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def client(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def _usage(source: str, task_id: str, tokens: dict) -> str:
    headers = {
        "claude-code": "Agent tool usage",
        "codex": "Codex tool usage",
        "hermes-agent": "Hermes Agent tool usage",
    }
    return headers[source] + "\n" + json.dumps({
        "schema_version": 2,
        "source": source,
        "event_id": "usage-" + hashlib.sha256(
            f"{source}\0{task_id}".encode("utf-8")
        ).hexdigest()[:32],
        "tokens": tokens,
    })


def test_board_api_aggregates_unique_structured_token_comments(client) -> None:
    conn = kb.connect()
    try:
        claude = kb.create_task(conn, title="claude")
        codex = kb.create_task(conn, title="codex")
        untracked = kb.create_task(conn, title="old card")
        claude_body = _usage("claude-code", claude, {
            "input": 10, "output": 20, "cache_read": 30, "cache_write": 40,
            "reasoning": None, "requests": 2, "total": 88,
        })
        codex_body = _usage("codex", codex, {
            "input": 7, "output": 11, "cache_read": 13, "cache_write": 0,
            "reasoning": 5, "requests": 1, "total": 31,
        })
        kb.add_comment(conn, claude, "claude-code", claude_body)
        kb.add_comment(conn, claude, "retry", claude_body)
        kb.add_comment(conn, codex, "codex", codex_body)
        kb.add_comment(conn, codex, "user", "Codex tool usage\n" + json.dumps({
            "schema_version": 1,
            "source": "codex",
            "event_id": "usage-" + "f" * 32,
            "tokens": {"total": 999_999},
        }))
        kb.add_comment(conn, untracked, "user", "Codex tool usage\n{not json}")
    finally:
        conn.close()

    response = client.get("/api/plugins/kanban/board")
    assert response.status_code == 200
    payload = response.json()
    assert payload["token_usage"] == {
        "tokens": {
            "input": 17,
            "output": 31,
            "cache_read": 43,
            "cache_write": 40,
            "reasoning": 5,
            "requests": 3,
            "total": 119,
        },
        "tracked_tasks": 2,
        "total_tasks": 3,
        "reasoning_coverage": "partial",
    }
    tasks = {
        task["title"]: task
        for column in payload["columns"]
        for task in column["tasks"]
    }
    assert tasks["claude"]["token_usage"]["tokens"]["total"] == 88
    assert tasks["claude"]["token_usage"]["tokens"]["reasoning"] is None
    assert tasks["claude"]["token_usage"]["source"] == "claude-code"
    assert tasks["codex"]["token_usage"]["tokens"]["reasoning"] == 5
    assert tasks["old card"]["token_usage"] is None

    detail = client.get(f"/api/plugins/kanban/tasks/{claude}").json()
    assert detail["task"]["token_usage"]["tokens"]["total"] == 88
    assert detail["task"]["token_usage"]["reasoning_coverage"] == "output_included"


def test_boards_api_exposes_each_board_token_total(client) -> None:
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="hermes")
        kb.add_comment(conn, task_id, "hermes-agent", _usage(
            "hermes-agent", task_id,
            {
                "input": 3, "output": 5, "cache_read": 7, "cache_write": 11,
                "reasoning": 2, "requests": 1,
            },
        ))
    finally:
        conn.close()

    response = client.get("/api/plugins/kanban/boards")
    assert response.status_code == 200
    board = next(item for item in response.json()["boards"] if item["slug"] == "default")
    assert board["token_usage"]["tokens"]["total"] == 26
    assert board["token_usage"]["tracked_tasks"] == 1


def test_dashboard_renders_board_card_and_detail_token_usage() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    js = (repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "index.js").read_text()
    css = (repo_root / "plugins" / "kanban" / "dashboard" / "dist" / "style.css").read_text()

    assert "function TokenBoardSummary" in js
    assert "function TokenUsageDetail" in js
    assert "hermes-kanban-token-badge" in js
    assert "reasoning_coverage" in js
    assert "reasoningIncludedInOutput" in js
    assert "Provider-reported total" in js
    assert "tracked_tasks" in js
    assert ".hermes-kanban-token-summary" in css
    assert ".hermes-kanban-token-grid" in css
