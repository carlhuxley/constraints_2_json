"""
Event log for live monitoring of the hybrid interpreter.

Every interpret() call (and every collected training example) appends one JSON line
to an append-only file. The dashboard (dashboard/) tails that file, so it updates while
a run is in progress and accumulates across runs.
"""
import json
import os
import uuid
from datetime import datetime
from typing import Optional

DEFAULT_EVENT_LOG = "logs/interpreter_events.jsonl"
DEFAULT_COLLECTED = "training_data_collected.json"

# scripts/simulate_live.py writes a replayed stream here, apart from real use
SIMULATED_EVENT_LOG = "logs/simulated_events.jsonl"
SIMULATED_COLLECTED = "logs/simulated_collected.json"
SIMULATED_FAILURES = "logs/simulated_failures.json"

# (event log, collected training data) for each stream the dashboard can show
STREAMS = {
    "Real": (DEFAULT_EVENT_LOG, DEFAULT_COLLECTED),
    "Simulated": (SIMULATED_EVENT_LOG, SIMULATED_COLLECTED),
}


class EventLogger:
    """Appends monitoring events as JSON lines."""

    def __init__(self, path: str = DEFAULT_EVENT_LOG, run_id: Optional[str] = None):
        self.path = path
        # Tells events from different runs apart in the accumulated log
        self.run_id = run_id or uuid.uuid4().hex[:12]
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def log(self, event_type: str, **fields) -> dict:
        """Append one event; returns it as written."""
        event = {
            "type": event_type,
            "ts": datetime.now().isoformat(),
            "run_id": self.run_id,
            **fields,
        }
        # Open per write so each line is flushed and visible to a reader mid-run
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
        return event
