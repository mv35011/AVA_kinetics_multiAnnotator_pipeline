import logging
import time
from pathlib import Path
import json
from typing import Optional, Dict, Any, List, Set, Tuple

# ---------------- Logging Setup ----------------
logger = logging.getLogger("metrics")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# ---------------- Metrics Directory ----------------
METRICS_DIR = Path("data/metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Track Seen Events ----------------
# Avoid duplicates within 5 seconds
_seen_events: Set[Tuple[int, Optional[int], str, Optional[str], str, int]] = set()

# ---------------- Utilities ----------------
def _now() -> float:
    """Return current timestamp in seconds since epoch."""
    return time.time()

def _get_project_file(project_id: int) -> Path:
    """Return the per-project metrics JSONL file path."""
    return METRICS_DIR / f"{project_id}_metrics.jsonl"

# ---------------- Log Metric ----------------
def log_metric(
    event_type: str,
    project_id: int = -1,
    task_id: Optional[int] = None,
    annotator: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a metric event.
    'extra' can include:
      - files (for ingest_time)
      - time_on_task_creation (for task_ready)
      - output_file, time_on_export (for export_time)
    """
    extra = extra or {}
    extra_str = json.dumps(extra, sort_keys=True)

    # 5-second deduplication bucket
    time_bucket = int(_now() // 5)
    key = (project_id, task_id, event_type, annotator, extra_str, time_bucket)
    if key in _seen_events:
        logger.info(f"[SKIP] Duplicate metric: {key}")
        return
    _seen_events.add(key)

    record = {
        "timestamp": _now(),
        "event_type": event_type,
        "project_id": project_id,
        "task_id": task_id,
        "annotator": annotator,
        "extra": extra
    }

    metrics_file = _get_project_file(project_id)
    try:
        with open(metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"[METRIC] {event_type} project={project_id} task={task_id} annotator={annotator}")
    except Exception as e:
        logger.exception("Failed to write metric record: %s", e)

# ---------------- Read Metrics ----------------
def read_by_project(project_id: int) -> List[Dict[str, Any]]:
    """Read all metric records for a specific project."""
    metrics_file = _get_project_file(project_id)
    records: List[Dict[str, Any]] = []
    if not metrics_file.exists():
        return records
    try:
        with open(metrics_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.exception("Skipping malformed metric line: %s", line[:200])
    except Exception as e:
        logger.exception("Failed to read metrics file: %s", e)
    return records

def read_all() -> List[Dict[str, Any]]:
    """Read all metric records from all projects."""
    records: List[Dict[str, Any]] = []
    for metrics_file in METRICS_DIR.glob("*_metrics.jsonl"):
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.exception("Skipping malformed metric line: %s", line[:200])
        except Exception as e:
            logger.exception("Failed to read metrics file: %s", e)
    return records

# ---------------- Test Script ----------------
if __name__ == "__main__":
    # Test logging ingest_time
    log_metric("ingest_time", project_id=1, extra={"files": {"zips": ["clip1.zip"]}})
    # Test logging task_ready with time_on_task_creation
    log_metric("task_ready", project_id=1, task_id=101, annotator="user1", extra={"time_on_task_creation": 3.5})
    # Test logging export_time with time_on_export
    log_metric("export_time", project_id=1, extra={"output_file": "clip1.mp4", "time_on_export": 2.0})

    all_metrics = read_all()
    print(f"All metrics ({len(all_metrics)} records):")
    for m in all_metrics:
        print(m)
