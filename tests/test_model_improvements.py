from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

from src.data.preprocess_aligned_vietnamese import (
    aligned_row_is_usable,
    aligned_segments_from_row,
    select_aligned_rows,
)
from src.models.cfm_flow import (
    build_mismatched_texts,
    cfm_loss,
    joint_stem_channel_weights,
    sample_cfm,
)
from src.models.dit_transformer import InputEmbedding, MicroDiT
from src.models.native_text import (
    NATIVE_CTC_VOCAB_SIZE,
    NativeVietnameseTextEncoder,
    NativeProsodyConditioner,
    build_ctc_targets,
    build_frame_text_targets,
    build_timestamped_frame_text_targets,
    ctc_target_text,
    grapheme_token_ids,
    greedy_decode_ctc,
    greedy_decode_frame_text,
    native_frame_text_loss,
    native_text_token_ids,
    utf8_token_ids,
)
from src.models.text_to_music_diffusion import (
    MusicDiffusionConfig,
    denormalize_mel,
    load_checkpoint,
    normalize_mel,
)
from src.training.self_diffusion import (
    DiffusionTrainer,
    MusicDiffusionDataset,
    _is_checkpoint_improvement,
    _music_config_from_json,
    lyric_lines_for_window,
    clean_vietnamese_lyric,
    lyric_text_for_window,
    split_training_records,
    validate_dataset,
)
from scripts.evaluate_generation_quality import (
    compatible_pronunciation_prior_strengths,
    evenly_spaced_records,
    generation_candidate_rank,
    transcription_metrics,
)
from scripts.kaggle_phase_runtime import install_online_audio_dependencies
from scripts.pretrain_native_ctc import (
    NativeVocalCTCDataset,
    evaluate_native_text_recognizer,
)


class ModelImprovementTests(unittest.TestCase):
    def test_dataset_config_loader_ignores_provenance_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            path.write_text(
                json.dumps({
                    "sample_rate": 24_000,
                    "n_mels": 100,
                    "source_dataset": "owner/aligned-songs",
                    "exact_word_timestamps": True,
                }),
                encoding="utf-8",
            )

            config = _music_config_from_json(path)

            self.assertEqual(config.sample_rate, 24_000)
            self.assertEqual(config.n_mels, 100)

    def test_aligned_song_rows_preserve_exact_relative_word_times(self) -> None:
        row = {
            "chunk_start_ms": 10_000,
            "chunk_end_ms": 20_000,
            "chunk_word_timestamps": [
                [
                    {"start": 11_000, "end": 12_000, "word": "Mưa"},
                    {"start": 12_000, "end": 13_500, "word": "rơi"},
                ],
                [
                    {"start": 16_000, "end": 17_000, "word": "bên"},
                    {"start": 17_000, "end": 19_000, "word": "hiên"},
                ],
            ],
        }

        text, segments = aligned_segments_from_row(row)

        self.assertEqual(text, "Mưa rơi bên hiên")
        self.assertEqual(segments[0]["words"][0]["start"], 1.0)
        self.assertEqual(segments[1]["words"][-1]["end"], 9.0)

    def test_timestamped_frame_targets_ignore_silence_and_keep_word_order(self) -> None:
        segments = [[
            {
                "start": 1.0,
                "end": 3.0,
                "text": "mưa rơi",
                "words": [
                    {"start": 1.0, "end": 2.0, "word": "mưa"},
                    {"start": 2.0, "end": 3.0, "word": "rơi"},
                ],
            }
        ]]

        targets = build_timestamped_frame_text_targets(
            segments,
            crop_starts_seconds=[0.0],
            crop_ends_seconds=[4.0],
            time_steps=16,
            device="cpu",
        )

        self.assertTrue(torch.all(targets[0, :4] == -100))
        self.assertTrue(torch.all(targets[0, 12:] == -100))
        voiced = targets[0, 4:12]
        self.assertTrue(torch.all(voiced != -100))
        self.assertEqual(int(voiced[0]), grapheme_token_ids("mưa")[0])
        self.assertEqual(int(voiced[-1]), grapheme_token_ids("rơi")[-1])

    def test_frame_evaluator_excludes_unlabeled_silence_from_cer(self) -> None:
        segments = [[{
            "words": [
                {"start": 1.0, "end": 2.0, "word": "mưa"},
                {"start": 2.0, "end": 3.0, "word": "rơi"},
            ]
        }]]
        targets = build_timestamped_frame_text_targets(
            segments,
            crop_starts_seconds=[0.0],
            crop_ends_seconds=[4.0],
            time_steps=16,
            device="cpu",
        )
        wrong_token = grapheme_token_ids("x")[0]
        logits = torch.full((1, 16, NATIVE_CTC_VOCAB_SIZE), -20.0)
        logits[:, :, wrong_token] = 20.0
        for frame in range(16):
            token = int(targets[0, frame])
            if token != -100:
                logits[0, frame].fill_(-20.0)
                logits[0, frame, token] = 20.0

        class FixedRecognizer:
            def eval(self):
                return self

            def audio_text_logits(self, _vocal):
                return logits

        batch = {
            "vocal_mel": torch.zeros(1, 100, 64),
            "text": ["mưa rơi"],
            "segments": segments,
            "crop_start_seconds": [0.0],
            "crop_end_seconds": [4.0],
        }

        result = evaluate_native_text_recognizer(
            FixedRecognizer(),
            [batch],
            "cpu",
            objective="frame",
        )

        self.assertEqual(result["cer"], 0.0)
        self.assertEqual(result["frame_accuracy"], 1.0)

    def test_aligned_subset_limits_chunks_per_song_and_fast_lyrics(self) -> None:
        def row(song_id: str, chunk_id: str, word_count: int = 8) -> dict:
            interval_ms = 9000 / word_count
            words = [
                {
                    "start": round(index * interval_ms),
                    "end": round((index + 0.8) * interval_ms),
                    "word": ("mưa", "rơi", "bên", "hiên")[index % 4],
                }
                for index in range(word_count)
            ]
            return {
                "song_id": song_id,
                "chunk_id": chunk_id,
                "chunk_start_ms": 0,
                "chunk_end_ms": 10_000,
                "chunk_word_timestamps": [words],
            }

        candidates = [row("song-a", f"a-{index}") for index in range(4)]
        candidates += [row("song-b", "b-0"), row("song-c", "c-0", word_count=40)]
        selected, report = select_aligned_rows(
            candidates,
            max_records=4,
            max_chunks_per_song=2,
        )

        self.assertTrue(all(aligned_row_is_usable(item) for item in selected))
        self.assertEqual([item["chunk_id"] for item in selected], ["a-0", "a-1", "b-0"])
        self.assertEqual(report["songs"], 2)
        self.assertEqual(report["song_limit_rejected"], 2)
        self.assertEqual(report["rejected"], 1)

    def test_joint_stem_channel_weights_prioritize_vocal_only_when_requested(self) -> None:
        weights = joint_stem_channel_weights(3, 4.0)
        self.assertTrue(torch.equal(weights[:3], torch.ones(3)))
        self.assertTrue(torch.equal(weights[3:], torch.full((3,), 4.0)))
        self.assertTrue(torch.equal(joint_stem_channel_weights(3, 1.0), torch.ones(6)))

    def test_native_full_mix_quality_sweep_drops_tts_priors(self) -> None:
        native_model = type(
            "NativeModel",
            (),
            {"native_generation": True, "generation_target": "joint_stems"},
        )()
        legacy_model = type(
            "LegacyModel",
            (),
            {"native_generation": False, "generation_target": "vocal_only"},
        )()

        self.assertEqual(
            compatible_pronunciation_prior_strengths(native_model, [0.0, 0.75, 1.0]),
            [0.0],
        )
        self.assertEqual(
            compatible_pronunciation_prior_strengths(legacy_model, [0.0, 0.75]),
            [0.0, 0.75],
        )

    def test_native_prosody_returns_monotonic_controls_and_backpropagates(self) -> None:
        encoder = NativeVietnameseTextEncoder(out_dim=16, depth=1)
        hidden, mask = encoder(["mưa rơi", "nắng lên"], "cpu")
        present = torch.tensor([True, True])
        conditioner = NativeProsodyConditioner(16)
        output = conditioner(hidden, mask, present, 24)

        self.assertEqual(tuple(output["aligned_text"].shape), (2, 24, 16))
        self.assertEqual(tuple(output["controls"].shape), (2, 24, 3))
        self.assertTrue(torch.isfinite(output["duration_proportions"]).all())
        self.assertTrue(torch.allclose(
            output["duration_proportions"].sum(dim=-1),
            torch.ones(2),
            atol=1e-5,
        ))
        output["aligned_text"].square().mean().backward()
        self.assertGreater(float(conditioner.duration_head.weight.grad.abs().sum()), 0.0)

    def test_native_ctc_dataset_loads_only_vocal_stem(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "mels").mkdir()
            torch.save(torch.zeros(4, 8), root / "mels" / "vocal.pt")
            dataset = NativeVocalCTCDataset(
                root,
                MusicDiffusionConfig(n_mels=4, frames_per_chunk=8),
                [{
                    "id": "song",
                    "text": "Em yêu anh",
                    "vocal_mel_path": "mels/vocal.pt",
                    "backing_mel_path": "mels/does-not-exist.pt",
                    "style_embed_path": "mels/does-not-exist-style.pt",
                }],
            )

            item = dataset[0]
            (root / "mels" / "vocal.pt").unlink()
            cached_item = dataset[0]

        self.assertEqual(tuple(item["vocal_mel"].shape), (4, 8))
        self.assertTrue(item["text"])
        self.assertTrue(torch.equal(item["vocal_mel"], cached_item["vocal_mel"]))

    def test_native_ctc_targets_ignore_unpronounced_punctuation_and_digits(self) -> None:
        self.assertEqual(ctc_target_text("Em ơi, 2026!"), "em ơi")

    def test_native_ctc_greedy_decoder_collapses_blanks_and_repeats(self) -> None:
        target = grapheme_token_ids("ma")
        logits = torch.full((1, 6, NATIVE_CTC_VOCAB_SIZE), -10.0)
        path = [0, target[0], target[0], 0, target[1], 0]
        for frame, token in enumerate(path):
            logits[0, frame, token] = 10.0

        self.assertEqual(greedy_decode_ctc(logits), ["ma"])

    def test_frame_text_decoder_collapses_repeated_frame_labels(self) -> None:
        target = grapheme_token_ids("má")
        logits = torch.full((1, 6, NATIVE_CTC_VOCAB_SIZE), -10.0)
        path = [target[0], target[0], target[0], target[1], target[1], target[1]]
        for frame, token in enumerate(path):
            logits[0, frame, token] = 10.0

        self.assertEqual(greedy_decode_frame_text(logits), ["má"])

    def test_frame_text_targets_keep_graphemes_ordered_and_ignore_empty_text(self) -> None:
        tokens = grapheme_token_ids("má")
        targets = build_frame_text_targets(
            ["má", "   "],
            time_steps=6,
            device="cpu",
        )

        self.assertEqual(tuple(targets.shape), (2, 6))
        self.assertEqual(targets[0].tolist(), [tokens[0]] * 3 + [tokens[1]] * 3)
        self.assertEqual(targets[1].tolist(), [-100] * 6)

    def test_native_frame_text_loss_is_finite_and_backpropagates(self) -> None:
        logits = torch.randn(
            2,
            8,
            NATIVE_CTC_VOCAB_SIZE,
            requires_grad=True,
        )
        loss = native_frame_text_loss(logits, ["mưa", ""])

        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(logits.grad[0].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[1].abs().sum()), 0.0)

    def test_frame_text_supervision_reaches_recognizer_and_vocal_generators(self) -> None:
        config = MusicDiffusionConfig(n_mels=8, frames_per_chunk=64)
        model = MicroDiT(
            config,
            text_encoder_type="native_utf8",
            generation_target="joint_stems",
            dim=16,
            depth=1,
            heads=2,
            ff_mult=2,
            style_dim=12,
        )
        loss, _, _, details = cfm_loss(
            model,
            torch.randn(2, 64, 8),
            torch.randn(2, 64, 8),
            torch.zeros(2, 12),
            ["mưa rơi", "nắng lên"],
            config,
            lambda_vocal=1.0,
            native_frame_text_weight=0.25,
            native_frame_text_teacher_weight=1.0,
            frame_text_segments=[
                [{
                    "words": [
                        {"start": 0.2, "end": 1.2, "word": "mưa"},
                        {"start": 1.2, "end": 2.2, "word": "rơi"},
                    ]
                }],
                [{
                    "words": [
                        {"start": 0.4, "end": 1.4, "word": "nắng"},
                        {"start": 1.4, "end": 2.4, "word": "lên"},
                    ]
                }],
            ],
            frame_text_crop_starts_seconds=[0.0, 0.0],
            frame_text_crop_ends_seconds=[4.0, 4.0],
            native_vocal_prior_weight=0.1,
            return_details=True,
        )

        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(details["native_frame_text_pred"])
        self.assertIsNotNone(details["native_frame_text_teacher"])
        self.assertIsNotNone(details["native_frame_text_prior"])
        self.assertGreater(
            float(model.audio_text_recognizer.classifier.weight.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(float(model.vocal_proj_out.weight.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(model.native_vocal_prior.projection.weight.grad.abs().sum()),
            0.0,
        )

    def test_native_quality_dependencies_do_not_install_g2p_or_download_dictionary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "scripts.kaggle_phase_runtime.run_logged"
        ) as run_logged, patch(
            "scripts.kaggle_phase_runtime.urllib.request.urlretrieve"
        ) as download:
            install_online_audio_dependencies(Path(temp_dir), native_only=True)

        command = run_logged.call_args.args[0]
        self.assertFalse(any("text2phoneme" in value for value in command))
        download.assert_not_called()

    def test_native_utf8_encoder_preserves_vietnamese_tones_without_pretrained_model(self) -> None:
        encoder = NativeVietnameseTextEncoder(out_dim=16, depth=1)
        encoded, mask = encoder(["ma", "má", "mạ"], "cpu")

        self.assertEqual(encoded.shape[0], 3)
        self.assertTrue((mask.sum(dim=1) >= 4).all())
        self.assertNotEqual(utf8_token_ids("ma"), utf8_token_ids("má"))
        self.assertNotEqual(utf8_token_ids("má"), utf8_token_ids("mạ"))

    def test_native_text_conditioning_uses_one_token_per_vietnamese_grapheme(self) -> None:
        self.assertEqual(native_text_token_ids("má"), grapheme_token_ids("má"))
        self.assertEqual(len(native_text_token_ids("má")), 2)
        self.assertGreater(len(utf8_token_ids("má")), len(native_text_token_ids("má")))

    def test_native_vocal_prior_is_supervised_and_empty_prompt_is_neutral(self) -> None:
        config = MusicDiffusionConfig(n_mels=8, frames_per_chunk=32)
        model = MicroDiT(
            config,
            text_encoder_type="native_utf8",
            generation_target="joint_stems",
            dim=16,
            depth=1,
            heads=2,
            ff_mult=2,
            style_dim=12,
        )
        vocal = torch.randn(2, 32, 8)
        loss, _, _, details = cfm_loss(
            model,
            vocal,
            torch.randn_like(vocal),
            torch.zeros(2, 12),
            ["mưa rơi", "nắng lên"],
            config,
            lambda_vocal=1.0,
            style_dropout_prob=0.0,
            text_dropout_prob=0.0,
            native_vocal_prior_weight=0.5,
            vocal_structure_weight=0.1,
            return_details=True,
        )
        loss.backward()

        self.assertIsNotNone(details["native_vocal_prior"])
        self.assertIsNotNone(details["vocal_structure"])
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(
            float(model.native_vocal_prior.projection.weight.grad.abs().sum()),
            0.0,
        )

        _, empty_prior = model(
            x=torch.randn(1, 16, 16),
            texts=["   "],
            timestep=torch.zeros(1),
            style_prompt=torch.zeros(1, 12),
            return_native_vocal_prior=True,
        )
        self.assertTrue(torch.equal(empty_prior, torch.zeros_like(empty_prior)))

    def test_native_vocal_prior_ranks_matched_lyric_above_mismatch(self) -> None:
        config = MusicDiffusionConfig(n_mels=8, frames_per_chunk=32)
        model = MicroDiT(
            config,
            text_encoder_type="native_utf8",
            generation_target="joint_stems",
            dim=16,
            depth=1,
            heads=2,
            ff_mult=2,
            style_dim=12,
        )
        vocal = torch.randn(2, 32, 8)
        loss, _, _, details = cfm_loss(
            model,
            vocal,
            torch.randn_like(vocal),
            torch.zeros(2, 12),
            ["mua roi", "nang len"],
            config,
            lambda_vocal=1.0,
            style_dropout_prob=0.0,
            text_dropout_prob=0.0,
            native_vocal_prior_weight=1.0,
            text_contrastive_weight=0.5,
            text_contrastive_prob=1.0,
            return_details=True,
        )
        loss.backward()

        self.assertIsNotNone(details["native_vocal_prior_contrastive"])
        self.assertTrue(torch.isfinite(details["native_vocal_prior_contrastive"]))
        self.assertGreater(
            float(model.native_vocal_prior.projection.weight.grad.abs().sum()),
            0.0,
        )

    def test_native_audio_ctc_backpropagates_into_vocal_generator(self) -> None:
        config = MusicDiffusionConfig(n_mels=8, frames_per_chunk=64)
        model = MicroDiT(
            config,
            text_encoder_type="native_utf8",
            dim=16,
            depth=1,
            heads=2,
            ff_mult=2,
            style_dim=12,
        )
        vocal = torch.randn(2, 64, 8)
        loss, _, _, details = cfm_loss(
            model,
            vocal,
            torch.randn_like(vocal),
            torch.randn(2, 12),
            ["mưa rơi", "nắng lên"],
            config,
            lambda_vocal=1.0,
            native_ctc_weight=0.05,
            native_ctc_teacher_weight=0.05,
            return_details=True,
        )

        loss.backward()

        self.assertTrue(model.native_generation)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(details["native_ctc_pred"])
        self.assertIsNotNone(model.vocal_proj_out.weight.grad)
        self.assertGreater(float(model.vocal_proj_out.weight.grad.abs().sum()), 0.0)

    def test_frozen_native_ctc_still_backpropagates_to_vocal_input(self) -> None:
        model = MicroDiT(
            MusicDiffusionConfig(n_mels=8, frames_per_chunk=64),
            text_encoder_type="native_utf8",
            dim=16,
            depth=1,
            heads=2,
            ff_mult=2,
            style_dim=12,
        )
        for parameter in model.audio_text_recognizer.parameters():
            parameter.requires_grad_(False)
        vocal = torch.randn(2, 64, 8, requires_grad=True)

        model.audio_text_logits(vocal).square().mean().backward()

        self.assertIsNotNone(vocal.grad)
        self.assertTrue(torch.isfinite(vocal.grad).all())
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.audio_text_recognizer.parameters()
        ))

    def test_native_ctc_uses_available_frames_for_long_vietnamese_lines(self) -> None:
        text = "một ngày mới nắng lên trên con đường dài"
        _, lengths, _ = build_ctc_targets(
            [text], device="cpu", max_target_length=96
        )

        self.assertEqual(int(lengths[0]), len(grapheme_token_ids(text)))
        self.assertLess(len(grapheme_token_ids(text)), len(utf8_token_ids(text)))
        self.assertGreater(int(lengths[0]), 20)
        self.assertLessEqual(int(lengths[0]), 96)

    def test_native_ctc_migrates_old_byte_classifier_without_losing_frontend(self) -> None:
        config = MusicDiffusionConfig(n_mels=8, frames_per_chunk=64)
        model = MicroDiT(
            config,
            text_encoder_type="native_utf8",
            dim=16,
            depth=1,
            heads=2,
            ff_mult=2,
            style_dim=12,
        )
        state = model.state_dict()
        classifier = model.audio_text_recognizer.classifier
        state["audio_text_recognizer.classifier.weight"] = torch.zeros(
            257, classifier.in_features
        )
        state["audio_text_recognizer.classifier.bias"] = torch.zeros(257)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "native-byte.pt"
            torch.save(
                {
                    "config": asdict(config),
                    "model": state,
                    "ema": state,
                    "arch": {
                        "dim": 16,
                        "depth": 1,
                        "heads": 2,
                        "ff_mult": 2,
                        "style_dim": 12,
                        "text_encoder_type": "native_utf8",
                    },
                },
                checkpoint,
            )
            loaded, _, payload = load_checkpoint(
                checkpoint, text_encoder_type="native_utf8", use_ema=False
            )

        self.assertTrue(payload["native_ctc_classifier_migrated"])
        self.assertEqual(
            loaded.audio_text_recognizer.classifier.out_features,
            NATIVE_CTC_VOCAB_SIZE,
        )
        optimizer = torch.optim.AdamW(loaded.parameters(), lr=1e-4)
        trainer = DiffusionTrainer(loaded, config, optimizer)
        trainer.load_ema_state(payload["ema"])
        self.assertEqual(
            trainer.ema_parameters[
                "audio_text_recognizer.classifier.weight"
            ].shape[0],
            NATIVE_CTC_VOCAB_SIZE,
        )

    def test_joint_stem_model_generates_vocal_and_backing_from_one_denoiser(self) -> None:
        config = MusicDiffusionConfig(n_mels=8, frames_per_chunk=32)
        model = MicroDiT(
            config,
            text_encoder_type="native_utf8",
            generation_target="joint_stems",
            dim=16,
            depth=1,
            heads=2,
            ff_mult=2,
            style_dim=12,
        )
        vocal = torch.randn(2, 32, 8)
        loss, _, vocal_loss, details = cfm_loss(
            model,
            vocal,
            torch.randn_like(vocal),
            torch.randn(2, 12),
            ["mưa rơi", "nắng lên"],
            config,
            lambda_vocal=1.0,
            native_ctc_weight=0.05,
            native_ctc_teacher_weight=0.05,
            return_details=True,
        )
        loss.backward()

        self.assertTrue(model.joint_stem_generation)
        self.assertEqual(model.input_embed.proj_x.in_features, 16)
        self.assertEqual(model.proj_out.out_features, 16)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(vocal_loss))
        self.assertIsNotNone(details["native_ctc_pred"])
        self.assertGreater(float(model.proj_out.weight.grad[:8].abs().sum()), 0.0)
        self.assertEqual(float(model.proj_out.weight.grad[8:].abs().sum()), 0.0)
        self.assertGreater(float(model.vocal_proj_out.weight.grad.abs().sum()), 0.0)

        vocal_sample, backing_sample = sample_cfm(
            model,
            ["mưa rơi"],
            16,
            config,
            "cpu",
            steps=1,
            style_prompt=torch.zeros(1, 12),
        )
        self.assertEqual(tuple(vocal_sample.shape), (1, 8, 16))
        self.assertEqual(tuple(backing_sample.shape), (1, 8, 16))

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        trainer = DiffusionTrainer(model, config, optimizer)
        sensitivity = trainer.evaluate_text_sensitivity([
            {
                "vocal_mel": vocal.transpose(1, 2),
                "backing_mel": torch.randn_like(vocal).transpose(1, 2),
                "style_anchor": torch.zeros(2, 12),
                "text": ["mưa rơi", "nắng lên"],
            }
        ])
        self.assertGreaterEqual(sensitivity, 0.0)

    def test_full_mix_checkpoint_migrates_to_joint_stem_heads(self) -> None:
        config = MusicDiffusionConfig(n_mels=8, frames_per_chunk=32)
        old_model = MicroDiT(
            config,
            text_encoder_type="native_utf8",
            generation_target="full_mix",
            dim=16,
            depth=1,
            heads=2,
            ff_mult=2,
            style_dim=12,
        )
        state = old_model.state_dict()
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "full-mix.pt"
            torch.save(
                {
                    "config": asdict(config),
                    "model": state,
                    "arch": {
                        "dim": 16,
                        "depth": 1,
                        "heads": 2,
                        "ff_mult": 2,
                        "style_dim": 12,
                        "text_encoder_type": "native_utf8",
                        "generation_target": "full_mix",
                    },
                },
                checkpoint,
            )
            loaded, _, payload = load_checkpoint(
                checkpoint,
                text_encoder_type="native_utf8",
                generation_target="joint_stems",
                use_ema=False,
            )

        self.assertTrue(loaded.joint_stem_generation)
        self.assertTrue(payload["joint_stem_checkpoint_migrated"])
        self.assertTrue(
            torch.equal(
                loaded.proj_out.weight[8:],
                state["vocal_proj_out.weight"],
            )
        )

    def test_native_ctc_respects_classifier_free_text_dropout(self) -> None:
        class NativeStub(torch.nn.Module):
            native_generation = True

            def __init__(self) -> None:
                super().__init__()
                self.logits = torch.nn.Parameter(
                    torch.zeros(1, 8, NATIVE_CTC_VOCAB_SIZE)
                )

            def forward(
                self, *, x, texts, timestep, style_prompt, return_vocal_aux=False
            ):
                velocity = torch.zeros_like(x)
                return (velocity, velocity) if return_vocal_aux else velocity

            def audio_text_logits(self, vocal_mel):
                return self.logits.expand(vocal_mel.shape[0], -1, -1)

        _, _, _, details = cfm_loss(
            NativeStub(),
            torch.randn(2, 16, 4),
            torch.zeros(2, 16, 4),
            torch.zeros(2, 8),
            ["mưa", "nắng"],
            MusicDiffusionConfig(n_mels=4),
            lambda_vocal=1.0,
            text_dropout_prob=1.0,
            native_ctc_weight=0.1,
            native_ctc_teacher_weight=0.1,
            return_details=True,
        )

        self.assertEqual(float(details["native_ctc_pred"].detach()), 0.0)
        self.assertGreater(float(details["native_ctc_teacher"].detach()), 0.0)

    def test_content_negative_uses_a_different_nonempty_lyric(self) -> None:
        mismatched, valid = build_mismatched_texts(
            ["em yeu anh", "mua roi", "", "em yeu anh"]
        )

        self.assertEqual(mismatched, ["mua roi", "em yeu anh", "", "mua roi"])
        self.assertEqual(valid, [True, True, False, True])

    def test_cfg_tie_prefers_realistic_vocal_over_confident_drone(self) -> None:
        real = {
            "pitch_std_semitones": 2.6,
            "spectral_flatness": 0.022,
            "voiced_ratio": 0.86,
        }
        natural = {
            "asr": {"word_accuracy": 0.0, "cer": 1.0},
            "metrics": {
                "pitch_std_semitones": 2.7,
                "spectral_flatness": 0.029,
                "voiced_ratio": 0.60,
                "mean_voiced_prob": 0.05,
            },
        }
        drone = {
            "asr": {"word_accuracy": 0.0, "cer": 1.0},
            "metrics": {
                "pitch_std_semitones": 0.35,
                "spectral_flatness": 0.00004,
                "voiced_ratio": 1.0,
                "mean_voiced_prob": 0.70,
            },
        }

        self.assertGreater(
            generation_candidate_rank(natural, real),
            generation_candidate_rank(drone, real),
        )

    def test_quality_samples_cover_the_full_combined_dataset(self) -> None:
        records = [{"id": str(index)} for index in range(10)]

        selected = evenly_spaced_records(records, 4)

        self.assertEqual([record["id"] for record in selected], ["0", "3", "6", "9"])

    def test_best_checkpoint_rejects_conditioning_collapse(self) -> None:
        self.assertFalse(
            _is_checkpoint_improvement(0.20, 0.30, 0.08, 0.18, 0.001)
        )
        self.assertTrue(
            _is_checkpoint_improvement(0.20, 0.30, 0.22, 0.18, 0.001)
        )

    def test_resume_validation_samples_tensor_shapes_but_checks_all_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "mels").mkdir()
            (root / "config.json").write_text(
                json.dumps({"n_mels": 4}), encoding="utf-8"
            )
            records = []
            for index in range(3):
                vocal_path = root / "mels" / f"vocal-{index}.pt"
                backing_path = root / "mels" / f"backing-{index}.pt"
                torch.save(torch.zeros(4, 8), vocal_path)
                torch.save(torch.zeros(4, 8), backing_path)
                records.append({
                    "id": str(index),
                    "frames": 8,
                    "vocal_mel_path": f"mels/{vocal_path.name}",
                    "backing_mel_path": f"mels/{backing_path.name}",
                })
            (root / "records.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            report = validate_dataset(root, max_tensor_records=1)

        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["path_checked_records"], 3)
        self.assertEqual(report["tensor_shape_checked_records"], 1)

    def test_frame_aligned_text_gate_has_nonzero_floor(self) -> None:
        embedding = InputEmbedding(mel_dim=4, text_dim=8, out_dim=8)
        with torch.no_grad():
            embedding.text_frame_gate.fill_(-100.0)

        self.assertGreaterEqual(float(embedding.text_frame_strength().detach()), 0.20)

    def test_text_sensitivity_loss_penalizes_identical_empty_branch(self) -> None:
        class TextIgnoringModel(torch.nn.Module):
            def forward(self, *, x, texts, timestep, style_prompt):
                return torch.zeros_like(x)

        clean = torch.randn(2, 12, 4)
        backing = torch.zeros_like(clean)
        style = torch.zeros(2, 8)
        config = MusicDiffusionConfig(n_mels=4)
        common = {
            "text_contrastive_weight": 0.0,
            "text_contrastive_prob": 1.0,
            "text_sensitivity_target": 0.20,
        }
        torch.manual_seed(17)
        baseline = cfm_loss(
            TextIgnoringModel(), clean, backing, style, ["xin chao", "viet nam"], config,
            lambda_vocal=0.0, text_sensitivity_weight=0.0, **common,
        )[0]
        torch.manual_seed(17)
        penalized = cfm_loss(
            TextIgnoringModel(), clean, backing, style, ["xin chao", "viet nam"], config,
            lambda_vocal=0.0, text_sensitivity_weight=2.0, **common,
        )[0]

        # Shortfall=0.20 is in Huber's linear region: (0.20 - 0.025) * 2.
        self.assertAlmostEqual(float(penalized - baseline), 0.35, places=5)

    def test_content_loss_stays_finite_for_large_fp16_velocities(self) -> None:
        class LargeVelocityModel(torch.nn.Module):
            def forward(self, *, x, texts, timestep, style_prompt):
                # Squaring this in float16 overflows; cfm_loss must promote its
                # numerical objective to float32 before doing so.
                return torch.full_like(x, 400.0, dtype=torch.float16)

        clean = torch.randn(2, 12, 4)
        backing = torch.zeros_like(clean)
        style = torch.zeros(2, 8)
        loss = cfm_loss(
            LargeVelocityModel(),
            clean,
            backing,
            style,
            ["em yeu anh", "mua roi"],
            MusicDiffusionConfig(n_mels=4),
            lambda_vocal=0.0,
            text_contrastive_weight=0.5,
            text_contrastive_prob=1.0,
            text_sensitivity_weight=1.0,
            text_sensitivity_target=0.08,
        )[0]

        self.assertTrue(torch.isfinite(loss))

    def test_collapsed_text_response_has_finite_backward(self) -> None:
        class TextIgnoringModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bias = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, *, x, texts, timestep, style_prompt):
                # Both lyric branches are exactly equal. This is the collapse
                # state the sensitivity objective must escape without sqrt(0)
                # creating a NaN gradient.
                return torch.zeros_like(x) + self.bias

        model = TextIgnoringModel()
        clean = torch.randn(2, 12, 4)
        loss = cfm_loss(
            model,
            clean,
            torch.zeros_like(clean),
            torch.zeros(2, 8),
            ["em yeu anh", "mua roi"],
            MusicDiffusionConfig(n_mels=4),
            lambda_vocal=0.0,
            text_contrastive_weight=0.5,
            text_contrastive_prob=1.0,
            text_sensitivity_weight=1.0,
            text_sensitivity_target=0.08,
        )[0]

        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.bias.grad)
        self.assertTrue(torch.isfinite(model.bias.grad))

    def test_validation_split_is_stable_and_disjoint(self) -> None:
        records = [{"id": f"song-{index}", "text": "lời bài hát"} for index in range(20)]
        train_a, validation_a = split_training_records(records, validation_fraction=0.2, seed=123)
        train_b, validation_b = split_training_records(
            [dict(record) for record in records], validation_fraction=0.2, seed=123
        )

        train_ids = {record["id"] for record in train_a}
        validation_ids = {record["id"] for record in validation_a}
        self.assertFalse(train_ids & validation_ids)
        self.assertEqual(len(validation_ids), 4)
        self.assertEqual(validation_ids, {record["id"] for record in validation_b})
        self.assertEqual(train_ids, {record["id"] for record in train_b})

    def test_validation_split_keeps_chunks_from_the_same_song_together(self) -> None:
        records = [
            {
                "id": f"{song}-{chunk}",
                "song_id": song,
                "text": "mưa rơi bên hiên",
            }
            for song in ("song-a", "song-b", "song-c", "song-d")
            for chunk in range(2)
        ]

        training, validation = split_training_records(
            records,
            validation_fraction=0.25,
            seed=5602,
        )

        train_songs = {record["song_id"] for record in training}
        validation_songs = {record["song_id"] for record in validation}
        self.assertFalse(train_songs & validation_songs)
        self.assertEqual(len(validation), 2)

    def test_transcription_metrics_are_accent_insensitive(self) -> None:
        metrics = transcription_metrics(
            "Đêm nay mưa rơi trên lối cũ",
            "dem nay mua roi tren loi cu",
        )

        self.assertEqual(metrics["wer"], 0.0)
        self.assertEqual(metrics["cer"], 0.0)
        self.assertEqual(metrics["word_accuracy"], 1.0)


    def test_old_segment_records_get_approximate_word_alignment(self) -> None:
        segments = [{"start": 0.0, "end": 4.0, "text": "mot hai ba bon"}]

        selected = lyric_text_for_window("mot hai ba bon", segments, 1.1, 2.9)
        silence = lyric_text_for_window("mot hai ba bon", segments, 5.0, 6.0)

        self.assertEqual(selected, "hai ba")
        self.assertEqual(silence, "")

    def test_generation_prompt_preserves_segment_boundaries(self) -> None:
        segments = [
            {"start": 0.0, "end": 2.0, "text": "mot hai", "words": [
                {"start": 0.0, "end": 1.0, "word": "mot"},
                {"start": 1.0, "end": 2.0, "word": "hai"},
            ]},
            {"start": 2.1, "end": 4.0, "text": "ba bon", "words": [
                {"start": 2.1, "end": 3.0, "word": "ba"},
                {"start": 3.0, "end": 4.0, "word": "bon"},
            ]},
        ]
        self.assertEqual(
            lyric_lines_for_window("mot hai ba bon", segments, 0.0, 4.0),
            "mot hai\nba bon",
        )

    def test_training_label_filter_rejects_mixed_asr_noise(self) -> None:
        self.assertEqual(
            clean_vietnamese_lyric("thì Since my love comes to the past và tủy trùr"),
            "",
        )
        self.assertEqual(clean_vietnamese_lyric("Hãy subscribe cho kênh của mình"), "")
        self.assertTrue(clean_vietnamese_lyric("Anh sẽ chờ em dù biển xanh kia cạn khô"))

    def test_mel_normalization_round_trip(self) -> None:
        config = MusicDiffusionConfig(mel_mean=-4.0, mel_std=2.0)
        raw = torch.tensor([[-6.0, -4.0, 0.0]])

        restored = denormalize_mel(normalize_mel(raw, config), config)

        self.assertTrue(torch.allclose(restored, raw))

    def test_dataset_returns_lyrics_without_style_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "mels").mkdir()
            config = MusicDiffusionConfig(frames_per_chunk=100)
            vocal = torch.linspace(-6.0, 1.0, 100 * 400).reshape(100, 400)
            backing = vocal + 0.25
            torch.save(vocal, root / "mels" / "vocal.pt")
            torch.save(backing, root / "mels" / "backing.pt")
            record = {
                "id": "song",
                "text": "mot hai ba bon",
                "style": "Vietnamese pop, 120 BPM",
                "frames": 400,
                "has_vocal": True,
                "vocal_mel_path": "mels/vocal.pt",
                "backing_mel_path": "mels/backing.pt",
                "segments": [{"start": 0.0, "end": 4.0, "text": "mot hai ba bon"}],
            }
            (root / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

            dataset = MusicDiffusionDataset(root, config, lyric_aligned_crop_prob=0.0)
            with (
                patch("src.training.self_diffusion.random.random", return_value=1.0),
                patch("src.training.self_diffusion.random.randint", return_value=100),
            ):
                item = dataset[0]

            self.assertNotIn("Vietnamese pop", item["text"])
            self.assertEqual(item["text"], "hai ba")
            self.assertEqual(tuple(item["vocal_mel"].shape), (100, 100))

    def test_dataset_exposes_exact_word_times_for_joint_frame_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "mels").mkdir()
            config = MusicDiffusionConfig(frames_per_chunk=100)
            mel = torch.zeros(100, 100)
            torch.save(mel, root / "mels" / "vocal.pt")
            torch.save(mel, root / "mels" / "backing.pt")
            record = {
                "id": "aligned",
                "song_id": "song",
                "text": "mưa rơi",
                "style": "Vietnamese pop",
                "frames": 100,
                "has_vocal": True,
                "vocal_mel_path": "mels/vocal.pt",
                "backing_mel_path": "mels/backing.pt",
                "exact_word_timestamps": True,
                "segments": [{
                    "start": 0.1,
                    "end": 0.9,
                    "text": "mưa rơi",
                    "words": [
                        {"start": 0.1, "end": 0.5, "word": "mưa"},
                        {"start": 0.5, "end": 0.9, "word": "rơi"},
                    ],
                }],
            }
            (root / "records.jsonl").write_text(
                json.dumps(record, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            item = MusicDiffusionDataset(root, config)[0]

            self.assertEqual(len(item["frame_text_segments"]), 1)
            self.assertEqual(
                item["frame_text_segments"][0]["words"][0]["word"],
                "mưa",
            )
            self.assertEqual(item["frame_text_crop_start_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
