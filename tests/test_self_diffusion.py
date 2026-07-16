from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from src.data.vietnamese_g2p import vietnamese_g2p
from src.data.vietnamese_text import normalize_vietnamese_lyrics
from src.integrations.kaggle_auto import DEFAULT_KAGGLE_DATASET_SLUG, DEFAULT_MODEL, KaggleJobConfig, resolve_training_dataset_ref, run_local_generation, stage_text_to_music_job, validate_dataset_ref
from src.models.text_to_music_diffusion import build_lyric_timing, estimate_minimum_lyric_duration
from src.training.distill_training import KnowledgeDistillationTrainer, _load_teacher, run_distillation_training
from src.training.self_diffusion import create_random_dataset, load_reference_conditioning, train_model, validate_dataset
from server import PROJECT_ROOT, WEB_ROOT, _is_relative_to


class SelfDiffusionTests(unittest.TestCase):
    def test_random_dataset_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            report = create_random_dataset(Path(temp) / "dataset", count=2, frames=32)
            self.assertEqual(report["backend"], "genmusic-vn-self-diffusion")
            validation = validate_dataset(Path(temp) / "dataset")
            self.assertEqual(validation["status"], "valid")

    def test_validation_checks_both_separated_stems(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            create_random_dataset(root, count=1, frames=32)
            import torch

            backing_path = root / "mels" / "backing.pt"
            tensor = torch.load(root / "mels" / "sample_00000.pt", weights_only=True)
            torch.save(tensor, backing_path)
            (root / "records.jsonl").write_text(
                json.dumps({
                    "id": "separated",
                    "text": "Mot cau hat.",
                    "style": "pop",
                    "frames": 32,
                    "vocal_mel_path": "mels/vocal.pt",
                    "backing_mel_path": "mels/backing.pt",
                }) + "\n",
                encoding="utf-8",
            )
            validation = validate_dataset(root)
            self.assertEqual(validation["status"], "invalid")
            self.assertTrue(any(item["stem"] == "vocal" for item in validation["missing"]))

    def test_load_reference_conditioning_extracts_real_backing_and_style(self) -> None:
        # generate_audio()'s default (no reference) conditions on a zero backing_mel
        # and a pooled-text style vector -- a real train/inference mismatch, since
        # training uses the real backing_mel + MuQ-MuLan style anchor. This is the
        # extraction path that lets generation match training instead.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            create_random_dataset(root, count=1, frames=32)
            import torch

            vocal_tensor = torch.load(root / "mels" / "sample_00000.pt", weights_only=True)
            backing_tensor = vocal_tensor + 1.0
            style_tensor = torch.arange(512, dtype=torch.float32)
            torch.save(vocal_tensor, root / "mels" / "vocal.pt")
            torch.save(backing_tensor, root / "mels" / "backing.pt")
            torch.save(style_tensor, root / "mels" / "style.pt")
            (root / "records.jsonl").write_text(
                json.dumps({
                    "id": "song-1",
                    "text": "Mot cau hat.",
                    "style": "pop",
                    "frames": 32,
                    "vocal_mel_path": "mels/vocal.pt",
                    "backing_mel_path": "mels/backing.pt",
                    "style_embed_path": "mels/style.pt",
                }) + "\n",
                encoding="utf-8",
            )
            reference = load_reference_conditioning(root)
            self.assertEqual(reference["id"], "song-1")
            self.assertTrue(torch.equal(reference["backing_mel"], backing_tensor))
            self.assertTrue(torch.equal(reference["style_anchor"], style_tensor))

            reference_by_id = load_reference_conditioning(root, record_id="song-1")
            self.assertEqual(reference_by_id["id"], "song-1")

    def test_training_and_local_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset"
            create_random_dataset(dataset, count=2, frames=32)
            progress_path = root / "progress.json"
            report = train_model(
                dataset,
                root / "model.pt",
                epochs=1,
                batch_size=2,
                max_records=2,
                checkpoint_every_steps=1,
                progress_path=progress_path,
            )
            self.assertEqual(report["status"], "complete")
            self.assertEqual(json.loads(progress_path.read_text())["status"], "complete")
            import torch
            payload = torch.load(root / "model.pt", weights_only=False)
            self.assertEqual(payload["training_state"]["global_step"], 1)
            resumed = train_model(
                dataset,
                root / "model.pt",
                epochs=1,
                batch_size=2,
                max_records=2,
                resume=True,
                save_every_epoch=True,
            )
            self.assertEqual(resumed["resumed_from_epoch"], 1)
            generated = run_local_generation(text="Mưa rơi nhẹ nhàng.", style="soft piano", output_dir=root / "audio", duration_seconds=1, checkpoint=root / "model.pt", steps=1)
            self.assertEqual(generated["status"], "complete")
            self.assertTrue(generated["duration_auto_adjusted"])
            self.assertTrue(Path(generated["audio_path"]).exists())

    def test_training_resumes_from_mid_epoch_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset"
            checkpoint = root / "model.pt"
            create_random_dataset(dataset, count=4, frames=16)

            from src.models import text_to_music_diffusion as checkpoint_module

            real_save_checkpoint = checkpoint_module.save_checkpoint
            interrupted = False

            def save_then_interrupt(*args, **kwargs):
                nonlocal interrupted
                result = real_save_checkpoint(*args, **kwargs)
                state = kwargs.get("training_state") or {}
                if not interrupted and state.get("batch_in_epoch") == 1:
                    interrupted = True
                    raise RuntimeError("simulated worker preemption")
                return result

            with patch.object(
                checkpoint_module,
                "save_checkpoint",
                side_effect=save_then_interrupt,
            ):
                with self.assertRaisesRegex(RuntimeError, "preemption"):
                    train_model(
                        dataset,
                        checkpoint,
                        epochs=1,
                        batch_size=2,
                        dim=32,
                        depth=1,
                        heads=2,
                        ff_mult=1,
                        frames_per_chunk=16,
                        checkpoint_every_steps=1,
                    )

            import torch

            interrupted_payload = torch.load(checkpoint, weights_only=False)
            self.assertEqual(
                interrupted_payload["training_state"]["batch_in_epoch"],
                1,
            )
            resumed = train_model(
                dataset,
                checkpoint,
                epochs=1,
                batch_size=2,
                dim=32,
                depth=1,
                heads=2,
                ff_mult=1,
                frames_per_chunk=16,
                resume=True,
                checkpoint_every_steps=1,
            )
            self.assertEqual(resumed["resumed_from_epoch"], 0)
            self.assertEqual(resumed["resumed_from_batch"], 1)
            self.assertEqual(resumed["global_step"], 2)

    def test_kaggle_job_contains_only_project_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            state = stage_text_to_music_job(text="Một ngày mới.", output_root=temp, duration_seconds=4, genre="acoustic pop", config=KaggleJobConfig(submit=False))
            self.assertEqual(state["backend"], "genmusic-vn-self-diffusion")
            self.assertEqual(state["model"], DEFAULT_MODEL)
            self.assertTrue(state["training_dataset_ref"].endswith(DEFAULT_KAGGLE_DATASET_SLUG))
            self.assertIn("request_dataset_ref", state)
            self.assertIn("dataset_url", state)
            self.assertIn("kernel_url", state)
            script = Path(state["kernel_dir"], "run_genmusic.py").read_text(encoding="utf-8")
            source_zip = Path(state["dataset_dir"], "genmusic_vn_source.zip")
            with zipfile.ZipFile(source_zip) as archive:
                names = archive.namelist()
                self.assertIn("src/models/text_to_music_diffusion.py", names)
                self.assertFalse(any(Path(name).name.startswith(".env") for name in names))
                self.assertNotIn("kaggle.json", [Path(name).name for name in names])
            self.assertIn("genmusic_vn_source.zip", script)
            self.assertIn("train-self", script)
            self.assertIn('rglob("request.json")', script)
            self.assertIn("records.jsonl", script)
            self.assertIn("copytree", script)
            self.assertNotIn("make-random-dataset", script)
            self.assertNotIn("git clone", script.lower())
            self.assertNotIn("raw.", script.lower())

    def test_dataset_ref_contract(self) -> None:
        self.assertEqual(resolve_training_dataset_ref("alice/music-data"), "alice/music-data")
        self.assertEqual(validate_dataset_ref("alice/music-data"), "alice/music-data")
        with self.assertRaises(ValueError):
            validate_dataset_ref("not-a-dataset-ref")

    def test_server_uses_project_root_and_rejects_escape(self) -> None:
        self.assertEqual(PROJECT_ROOT, Path(__file__).resolve().parents[1])
        self.assertEqual(WEB_ROOT, PROJECT_ROOT / "web")
        self.assertTrue(_is_relative_to(WEB_ROOT / "index.html", WEB_ROOT))
        self.assertFalse(_is_relative_to(WEB_ROOT.parent / "secret.txt", WEB_ROOT))

    def test_distillation_reports_honest_fallback_when_teacher_unavailable(self) -> None:
        # Without the DiffRhythm2 repo vendored on PYTHONPATH (not the case in this
        # test environment), _load_teacher must return None + a status message --
        # never a silent fake stand-in (the previous "DummyTeacher" behavior this
        # replaced). See docs/experiments/distillation_fix.md.
        teacher, model_config, status = _load_teacher("ASLP-lab/DiffRhythm2", None, "cpu")
        self.assertIsNone(teacher)
        self.assertIsNone(model_config)
        self.assertIsInstance(status, str)
        self.assertTrue(status)

    def test_mel_adapter_gradient_flow(self) -> None:
        # Regression test for a real bug: to_teacher_mel/from_teacher_mel used to be
        # computed entirely inside train_epoch's torch.no_grad() block, so neither
        # ever received a gradient despite being registered as trainable adapter
        # params (see docs/experiments/distillation_fix.md). to_teacher_mel is now a
        # fixed deterministic resize (its output feeds the frozen teacher's own
        # no_grad-scoped forward pass, which must stay cheap); from_teacher_mel stays
        # a real trainable adapter and must receive a real gradient.
        import torch
        import torch.nn as nn
        from src.models.dit_transformer import MicroDiT
        from src.models.text_to_music_diffusion import MusicDiffusionConfig

        class FakeTeacher(nn.Module):
            def __init__(self, cond_dim=512, mel_dim=64):
                super().__init__()
                self.text_embed = nn.Embedding(600, cond_dim)
                self.latent_embed = nn.Sequential(nn.Linear(mel_dim, cond_dim))
                self.proj_out = nn.Linear(cond_dim, mel_dim)

            def forward(self, x, time, position_ids, style_prompt, attn_mask, use_cache=False, past_key_value=None):
                return self.proj_out(x), None, None

        config = MusicDiffusionConfig(frames_per_chunk=16)
        student = MicroDiT(config, dim=32, depth=1, heads=2, ff_mult=1, style_dim=512)
        trainer = KnowledgeDistillationTrainer(
            teacher_model=FakeTeacher(), student_model=student, config=config, optimizer=None,
            device="cpu", alpha_feature=0.5, parse_lyrics_fn=lambda text: [500, 3, 4, 511], teacher_mel_dim=64,
        )
        self.assertFalse(hasattr(trainer, "to_teacher_mel"))
        self.assertIsNotNone(trainer.from_teacher_mel)

        xt = torch.randn(2, 16, 100)
        t = torch.rand(2)
        style = torch.randn(2, 512)
        v_teacher = trainer._teacher_velocity(xt, t, ["xin chao"] * 2, style)
        v_student = student(x=xt, cond=torch.zeros_like(xt), texts=["xin chao"] * 2, timestep=t, style_prompt=style)
        (v_student - v_teacher).pow(2).mean().backward()

        self.assertIsNotNone(trainer.from_teacher_mel.weight.grad)
        self.assertGreater(trainer.from_teacher_mel.weight.grad.abs().sum().item(), 0.0)

    def test_train_distill_raises_when_teacher_unavailable(self) -> None:
        # train-distill must never silently downgrade to ground-truth-only training
        # under the distillation name -- it should raise so the failure is impossible
        # to miss, instead of only being visible via the distillation_active field.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset"
            create_random_dataset(dataset, count=2, frames=32)
            with self.assertRaises(RuntimeError):
                run_distillation_training(dataset, root / "model.pt", epochs=1, batch_size=2)

    def test_vietnamese_text_contract(self) -> None:
        self.assertIn("mười hai", normalize_vietnamese_lyrics("Mưa 12 ngày, ko về."))
        result = vietnamese_g2p("Sóng gió", use_phonemizer=False)
        self.assertEqual(result.backend, "rule-based-ipa")
        self.assertEqual(len(result.tokens), 2)

    def test_model_preserves_lyric_line_structure(self) -> None:
        timing = build_lyric_timing("Một câu chậm.\nMột câu khác.", 8)
        self.assertEqual(len(timing), 2)
        self.assertAlmostEqual(timing[-1]["end_seconds"], 8.0, places=3)


    def test_minimum_duration_prevents_rushed_lyrics(self) -> None:
        self.assertGreaterEqual(estimate_minimum_lyric_duration("Mot ngay moi bat dau, anh van nho em."), 4.0)


if __name__ == "__main__":
    unittest.main()
