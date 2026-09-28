"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Start a request; return its ID for concurrent request correlation."""
        key = request_id or user_id
        if key in self._open:
            raise ValueError(f"Request already open: {key}")
        self._open[key] = {
            "request_id": request_id or uuid4().hex,
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
            "start": time.monotonic(),
        }
        return key

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Finish a matching request and append its decision and latency."""
        key = request_id or user_id
        entry = self._open[key]
        if entry["user_id"] != user_id:
            raise ValueError("Request belongs to another user")
        entry = self._open.pop(key)
        elapsed = (time.monotonic() - entry.pop("start")) * 1000
        entry.update(output=text, blocked=blocked, layer=layer,
                     finished_at=utc_now_iso(), latency_ms=max(0.0, elapsed))
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Export completed requests; active requests remain in memory."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
