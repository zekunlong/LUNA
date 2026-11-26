"""Bounded CPU tests for packaging gates; no final ZIP or user figures created."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import package_attention_delivery as package


def npy(descr, shape, body, fortran=False):
    header = repr({"descr": descr, "fortran_order": fortran, "shape": shape}).encode("latin1")
    header += b" " * ((64 - ((10 + len(header) + 1) % 64)) % 64) + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + body


def strings(values, scalar=False):
    width = max(1, max(map(len, values)))
    body = b"".join(value.ljust(width, "\x00").encode("utf-32-le") for value in values)
    return npy("<U" + str(width), () if scalar else (len(values),), body)


def make_bundle(path, methods=package.METHODS, truncate=False, object_dtype=False):
    metadata = {"dataset": "fgvc_aircraft", "samples": [{"impath": "sample"}] * 3,
                "missing_methods": []}
    arrays = {"images": npy("<f4", (3, 2, 2, 3), bytes(3 * 2 * 2 * 3 * 4)),
              "maps": npy("<f4", (3, len(methods), 2, 2), bytes(3 * len(methods) * 2 * 2 * 4)),
              "methods": strings(methods), "classnames": strings(("a", "b", "c")),
              "metadata": strings((json.dumps(metadata),), scalar=True)}
    if truncate:
        arrays["maps"] = arrays["maps"][:-4]
    if object_dtype:
        arrays["extra"] = npy("|O", (), b"no pickle used")
    with zipfile.ZipFile(path, "w") as archive:
        for name, body in arrays.items():
            archive.writestr(name + ".npy", body)


class PackagingTests(unittest.TestCase):
    def test_archive_paths_are_canonical_and_relative(self):
        for value in ("../x", "/root/x", "C:/x", "a\\b", "a/../b", "a/./b", "a//b", "a/"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                package.member_name(value)
        self.assertEqual(package.member_name("paper/main_comparison.pdf"), "paper/main_comparison.pdf")

    def test_npz_complete_five_method_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test_bundle.npz"
            make_bundle(path)
            record = package.inspect_bundle(path)
            self.assertEqual(record["rows"], 3)
            self.assertEqual(record["methods"], list(package.METHODS))

    def test_npz_missing_methods_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test_bundle.npz"
            make_bundle(path, ("CLIP", "LUNA"))
            with self.assertRaisesRegex(ValueError, "Five methods"):
                package.inspect_bundle(path)

    def test_npz_reordered_methods_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test_bundle.npz"
            make_bundle(path, tuple(reversed(package.METHODS)))
            with self.assertRaisesRegex(ValueError, "Five methods"):
                package.inspect_bundle(path)

    def test_npz_truncated_tensor_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test_bundle.npz"
            make_bundle(path, truncate=True)
            with self.assertRaisesRegex(ValueError, "payload length"):
                package.inspect_bundle(path)

    def test_npz_object_dtype_never_deserialized(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test_bundle.npz"
            make_bundle(path, object_dtype=True)
            with self.assertRaisesRegex(ValueError, "Object arrays"):
                package.inspect_bundle(path)

    def test_outside_boundary_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "product"
            root.mkdir()
            outside = base / "outside" / "no_file.json"
            with self.assertRaises(ValueError):
                package.safe_destination(outside, root)
            self.assertFalse(outside.parent.exists())

    def test_sensitive_and_cache_files_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            tree = root / "validation"
            tree.mkdir()
            for name in ("audit.json", ".env.json", "credentials.json", "secrets.json"):
                (tree / name).write_text("{}", encoding="utf-8")
            (tree / "logs").mkdir()
            (tree / "logs" / "hidden.json").write_text("{}", encoding="utf-8")
            entries = {}
            package.add_tree(entries, root, "validation", {".json"})
            self.assertEqual(set(entries), {"validation/audit.json"})

    def test_incomplete_report_never_creates_zip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            verification = package.validate(root, {})
            self.assertIn("Missing payload file: report_manifest.json", verification["blockers"])
            inventory = package.manifest({}, verification, False)
            with self.assertRaisesRegex(ValueError, "Incomplete delivery"):
                package.write_archive(root, {}, inventory)
            self.assertEqual(list(root.iterdir()), [])

    def test_relative_link_closure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "README_zh.md").write_text("[figure](paper/main.png) [external](https://example.org)\n", encoding="utf-8")
            (root / "paper").mkdir()
            (root / "paper/main.png").write_bytes(b"test-only fixture")
            entries = {}
            package.add_file(entries, root / "README_zh.md", root)
            self.assertEqual(len(package.unresolved_links(entries)), 1)
            package.add_file(entries, root / "paper/main.png", root)
            self.assertEqual(package.unresolved_links(entries), [])

    def test_payload_hash_matches_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / "protocol.json"
            content = b'{"protocol":"test fixture"}'
            path.write_bytes(content)
            entries = {}
            package.add_file(entries, path, root)
            record = entries["protocol.json"].record()
            self.assertEqual(record["sha256"], hashlib.sha256(content).hexdigest())
            self.assertEqual(record["size_bytes"], len(content))

    def test_stale_report_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            evidence = root / "validation"
            evidence.mkdir()
            evidence_file = evidence / "audit.json"
            evidence_file.write_text('{"new":true}', encoding="utf-8")
            report = {"status": "COMPLETE", "artifacts": {},
                      "source_files": [{"path": str(evidence_file), "sha256": "0" * 64}]}
            (root / "report_manifest.json").write_text(json.dumps(report), encoding="utf-8")
            entries = {}
            package.add_file(entries, evidence_file, root)
            package.add_file(entries, root / "report_manifest.json", root)
            verification = package.validate(root, entries)
            self.assertIn("Report evidence hash is stale: validation/audit.json", verification["blockers"])

    def test_compact_checkpoint_size_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / "attention_checkpoint.pth"
            # Sparse local test file, never a real model and never archived.
            with path.open("wb") as handle:
                handle.truncate(64 * 1024 * 1024 + 1)
            with self.assertRaisesRegex(ValueError, "64 MiB"):
                package.add_file({}, path, root)


if __name__ == "__main__":
    unittest.main()
