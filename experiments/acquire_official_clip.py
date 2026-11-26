"""Fetch the exact OpenAI CLIP weights specified by the existing LUNA code."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import ast
import hashlib
import json
from pathlib import Path
import time
import urllib.request

REPO = Path(__file__).resolve().parents[1]
CACHE = Path.home() / ".cache" / "clip"
LOG = Path("E:/LUNA/results_reliability_analysis/logs/asset_acquisition_clip.json")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire(name, url):
    CACHE.mkdir(parents=True, exist_ok=True)
    destination = CACHE / url.rsplit("/", 1)[1]
    expected = url.split("/")[-2]
    if destination.is_file() and sha256(destination) == expected:
        return {"name": name, "path": str(destination), "url": url, "sha256": expected, "status": "VERIFIED_REUSED"}
    partial = destination.with_suffix(".pt.part")
    for attempt in range(1, 5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "LUNA-research-asset-acquisition/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                append = offset > 0 and response.status == 206
                if not append:
                    offset = 0
                total = offset + int(response.headers.get("Content-Length", 0))
                mode = "ab" if append else "wb"
                print(f"{name}: attempt={attempt}, resume={offset}, total={total}", flush=True)
                last = time.monotonic()
                with partial.open(mode) as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        offset += len(chunk)
                        if time.monotonic() - last >= 10:
                            print(f"{name}: {offset / 1024**2:.1f}/{total / 1024**2:.1f} MiB", flush=True)
                            last = time.monotonic()
            actual = sha256(partial)
            if actual != expected:
                raise ValueError(f"SHA256 mismatch: expected {expected}, found {actual}")
            partial.replace(destination)
            print(f"{name}: VERIFIED {destination}", flush=True)
            return {"name": name, "path": str(destination), "url": url, "sha256": actual,
                    "bytes": destination.stat().st_size, "status": "DOWNLOADED_AND_VERIFIED"}
        except Exception as exc:
            print(f"{name}: attempt {attempt} failed: {type(exc).__name__}: {exc}", flush=True)
            if attempt == 4:
                return {"name": name, "url": url, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
            time.sleep(2)


if __name__ == "__main__":
    tree = ast.parse((REPO / "clip" / "clip.py").read_text(encoding="utf-8"))
    models = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_MODELS" for t in node.targets))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(acquire, name, models[name]) for name in ("ViT-B/16", "ViT-L/14")]
        results = [future.result() for future in futures]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(), "assets": results}, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    raise SystemExit(0 if all(item["status"] != "FAILED" for item in results) else 1)
