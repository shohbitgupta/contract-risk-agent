import json
from datetime import datetime
from pathlib import Path
from typing import Dict


class AuditLogger:
    """
    Append-only audit logger for legal AI decisions.
    """

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, payload: Dict):
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "payload": payload
        }

        file_path = self.log_dir / f"{event_type}.log.jsonl"

        with open(file_path, "a") as f:
            f.write(json.dumps(record) + "\n")
