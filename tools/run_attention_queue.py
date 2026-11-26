#!/usr/bin/env python3
"""Durable sequential GPU queue with logged commands and explicit completion.

The JSON plan is reviewable before launching. Completed stages are skipped;
failures stop the queue without deleting files or replacing experimental data.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    # Windows readers can briefly hold a file without FILE_SHARE_DELETE.
    # Keep the complete old state readable while retrying the atomic replace.
    for attempt in range(80):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 79:
                raise
            time.sleep(.05)


def complete(stage):
    paths = stage.get("outputs", [])
    if not paths or not all(Path(p).is_file() and Path(p).stat().st_size for p in paths):
        return False
    condition = stage.get("completion_json")
    if condition:
        try:
            data = json.loads(Path(condition["path"]).read_text(encoding="utf-8"))
            return data.get(condition.get("key", "status")) == condition["value"]
        except (OSError, ValueError):
            return False
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text(encoding="utf-8"))
    logs = Path(plan["log_dir"]); logs.mkdir(parents=True, exist_ok=True)
    status_path = a.plan.with_name("queue_status.json")
    status = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
              "pid": os.getpid(), "plan": str(a.plan.resolve()), "stages": []}
    write(status_path, status)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", OUTDATED_IGNORE="1")
    try:
        for stage in plan["stages"]:
            record = {"id": stage["id"], "status": "PENDING"}
            status["stages"].append(record)
            status["current_stage"] = stage["id"]
            if complete(stage):
                record["status"] = "ALREADY_COMPLETE"; write(status_path, status); continue
            selection_path = stage.get("skip_if_empty_samples")
            if selection_path:
                selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
                if selection.get("samples") == []:
                    record.update(status="SKIPPED_EMPTY_SELECTION", reason="No image satisfies the declared selection rule")
                    write(status_path, status); continue
            if stage.get("wait_only"):
                record["status"] = "WAITING_FOR_EXISTING_PROCESS"; write(status_path, status)
                while not complete(stage):
                    failure = stage.get("completion_json")
                    if failure and Path(failure["path"]).is_file():
                        state = json.loads(Path(failure["path"]).read_text(encoding="utf-8"))
                        if state.get("status") == "FAILED":
                            raise RuntimeError("Existing process failed: "+stage["id"])
                    time.sleep(2)
                record["status"] = "COMPLETED"; write(status_path, status); continue
            command = stage["command"]
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            log_path = logs/(stage["id"]+"_"+stamp+".log")
            record.update(status="RUNNING", command=command, log=str(log_path), started_utc=stamp)
            write(status_path, status)
            print("RUN", stage["id"], flush=True)
            started = time.monotonic()
            with log_path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps({"command": command, "cwd": stage.get("cwd", plan["cwd"])})+"\n")
                handle.flush()
                child = subprocess.Popen(command, cwd=stage.get("cwd", plan["cwd"]), env=env,
                                         stdout=handle, stderr=subprocess.STDOUT,
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
                record["child_pid"] = child.pid; write(status_path, status)
                code = child.wait()
            record.update(exit_code=code, elapsed_seconds=time.monotonic()-started)
            if code or not complete(stage):
                record["status"] = "FAILED"
                raise RuntimeError("Stage failed or did not produce complete outputs: "+stage["id"])
            record["status"] = "COMPLETED"; write(status_path, status)
        status["status"] = "COMPLETED"
        status["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write(status_path, status)
        print("QUEUE COMPLETED", flush=True)
    except Exception as exc:
        status.update(status="FAILED", error=repr(exc))
        write(status_path, status)
        raise


if __name__ == "__main__":
    main()
