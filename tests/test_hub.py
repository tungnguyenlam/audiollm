"""Backup integrity and recovery tests; no credentials or network required."""
import json
import fnmatch
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from audiollm import config as C
from audiollm.hub import HubSync, prepare_bundle
from scripts.hf_backup import restore


class FakeApi:
    def __init__(self):
        self.files = {}
        self.failures = 0
        self.calls = 0

    def whoami(self):
        return {"name": "tester"}

    def create_repo(self, repo_id, **kwargs):
        self.repo_id = repo_id
        assert kwargs["private"] is False

    def upload_folder(self, **kwargs):
        self.calls += 1
        if self.failures:
            self.failures -= 1
            raise OSError("offline")
        assert "delete_patterns" not in kwargs
        for file in Path(kwargs["folder_path"]).rglob("*"):
            if file.is_file():
                self.files[file.relative_to(kwargs["folder_path"]).as_posix()] = file.read_bytes()
        return "https://huggingface.co/tester/backup/commit/test-sha"


class HubTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name) / "run-123"
        self.run.mkdir()
        (self.run / "config.json").write_text(json.dumps(C.get_config_dictionary()))
        (self.run / "mvn.pt").write_bytes(b"mvn")
        self.api = FakeApi()
        with patch("huggingface_hub.HfApi", return_value=self.api), patch("audiollm.hub.prepare_bundle"):
            self.hub = HubSync("auto", self.run, lambda *args: None)
        self.addCleanup(self.hub.executor.shutdown)

    def checkpoint(self, step):
        ckpt = self.run / "checkpoint"
        ckpt.mkdir(exist_ok=True)
        (ckpt / "lora").mkdir(exist_ok=True)
        (ckpt / "lora/adapter_model.safetensors").write_bytes(str(step).encode())
        (ckpt / "proj.pt").write_bytes(str(step).encode())
        (ckpt / "embed.pt").write_bytes(b"embedding")
        torch.save({"optimizer_step": step, "optimizer": {}, "rng": {}}, ckpt / "trainer_state.pt")

    def test_retains_steps_and_final_and_uploads_immutable_snapshot(self):
        self.checkpoint(500)
        self.hub.push("step 500")
        (self.run / "checkpoint/proj.pt").write_bytes(b"changed during upload")
        self.assertTrue(self.hub.wait())
        self.checkpoint(1000)
        self.hub.push("step 1000")
        self.assertTrue(self.hub.wait())
        (self.run / "proj.pt").write_bytes(b"final")
        self.hub.push("final", final=True)
        self.assertTrue(self.hub.wait())
        self.assertEqual(self.api.repo_id, "tester/audiollm-run-123")
        self.assertEqual(self.api.files["run-123/checkpoints/step-000000500/proj.pt"], b"500")
        self.assertEqual(self.api.files["run-123/checkpoints/step-000001000/embed.pt"], b"embedding")
        self.assertIn("run-123/checkpoint/trainer_state.pt", self.api.files)
        self.assertIn("Quickstart", self.api.files["README.md"].decode())
        self.assertEqual(self.api.files["run-123/proj.pt"], b"final")

    def test_bundles_local_base_tokenizer_and_quickstart(self):
        base = Path(self.tmp.name) / "custom-base"
        base.mkdir()
        (base / "config.json").write_text("{}")
        (base / "tokenizer.json").write_text("custom tokenizer")
        (base / "model.safetensors").write_bytes(b"frozen base")
        with patch.object(C, "MODEL", str(base)):
            prepare_bundle(self.run)
        self.assertEqual((self.run / "base_model/tokenizer.json").read_text(), "custom tokenizer")
        self.assertTrue((self.run / "quickstart/transcribe.py").exists())
        self.assertTrue((self.run / "quickstart/audiollm/model.py").exists())

    def test_retries_and_reports_failure(self):
        self.api.failures = 3
        with patch("audiollm.hub.time.sleep"):
            self.hub.push("failed")
            self.assertFalse(self.hub.wait())
        self.assertEqual(self.api.calls, 3)
        self.assertEqual(json.loads((self.run / "hub_status.json").read_text())["status"], "failed")
        self.hub.push("retry")
        self.assertTrue(self.hub.wait())
        self.assertEqual(json.loads((self.run / "hub_status.json").read_text())["status"], "uploaded")

    def test_restore_step_selectively_and_resume(self):
        self.checkpoint(500)
        self.hub.push("500")
        self.hub.wait()
        def download(repo, **kwargs):
            self.assertEqual(kwargs["revision"], "abc123")
            self.assertIn("run-123/checkpoints/step-000000500/*", kwargs["allow_patterns"])
            for name, data in self.api.files.items():
                if not any(fnmatch.fnmatch(name, pattern) for pattern in kwargs["allow_patterns"]):
                    continue
                path = Path(kwargs["local_dir"]) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        output = Path(self.tmp.name) / "restore"
        with patch("huggingface_hub.snapshot_download", side_effect=download):
            restored = restore("tester/backup", self.run.name, output, "500", "abc123")
            self.assertEqual((restored / "checkpoint/proj.pt").read_bytes(), b"500")
            with self.assertRaises(FileExistsError):
                restore("tester/backup", self.run.name, output)
        with self.assertRaises(ValueError):
            restore("tester/backup", "../bad", output)


if __name__ == "__main__":
    unittest.main()
