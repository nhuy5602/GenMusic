"""Submit Stage-2 phonemize kernels for each of quynhvu03's pre-processed
raw-audio dataset kernels.

Each kernel:
  - Takes one quynhvu03 kernel output as kernel_source
  - Runs G2P (text2phonemesequence) to add phoneme fields
  - Outputs updated records.jsonl + symlinked waveforms
  - Runs on CPU (T4 GPU not needed — G2P is pure Python)

Usage:
    python scripts/run_kaggle_multi_part_phonemize.py
    python scripts/run_kaggle_multi_part_phonemize.py --wait-and-loop
    python scripts/run_kaggle_multi_part_phonemize.py --max-new-jobs 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.integrations.kaggle_auto import (
    kaggle_auth_available,
    kaggle_auth_environment,
    kaggle_cli_command,
    load_kaggle_api_tokens,
    resolve_kaggle_username,
    write_source_zip,
)

# ── source kernels from quynhvu03 to phonemize ────────────────────────────────
# Format: (part_number, kernel_ref_for_kernel_sources, kernel_slug_for_path)
QUYNHVU03_KERNELS: list[tuple[int, str, str]] = [
    (1, "quynhvu03/genmusic-data-prep-p1-1784830145", "genmusic-data-prep-p1-1784830145"),
    (2, "quynhvu03/genmusic-data-prep-p2-1784830148", "genmusic-data-prep-p2-1784830148"),
    (3, "quynhvu03/genmusic-data-prep-p3-1784843636", "genmusic-data-prep-p3-1784843636"),
    (4, "quynhvu03/genmusic-data-prep-p4-1784843640", "genmusic-data-prep-p4-1784843640"),
    (5, "quynhvu03/genmusic-data-prep-p5-1784859349", "genmusic-data-prep-p5-1784859349"),
    (6, "quynhvu03/genmusic-data-prep-p6-1784859353", "genmusic-data-prep-p6-1784859353"),
]


def _run_cli(cli: list[str], args: list[str], env: dict, *, timeout: int = 120) -> "subprocess.CompletedProcess[str]":
    import subprocess
    result = subprocess.run(
        cli + args, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result


def _wait_for_dataset(cli: list[str], ref: str, env: dict) -> None:
    time.sleep(10)
    for _ in range(30):
        try:
            result = _run_cli(cli, ["datasets", "status", ref], env, timeout=60)
            status = (result.stdout + result.stderr).lower()
            if result.returncode == 0 and "ready" in status:
                time.sleep(5)
                return
        except Exception:
            pass
        time.sleep(3)
    print("⚠️ Dataset source upload finished (continuing kernel push)...", flush=True)


def _is_kernel_finished(cli: list[str], kernel_ref: str, env: dict) -> bool:
    try:
        res = _run_cli(cli, ["kernels", "status", kernel_ref], env, timeout=60)
        output = (res.stdout + res.stderr).lower()
        if "complete" in output or "error" in output or "cancel" in output:
            return True
    except Exception:
        pass
    return False


def _build_kernel_script(kernel_slug: str) -> str:
    """Return the phonemize kernel script with the source slug substituted in."""
    template_path = Path(__file__).parent / "run_kaggle_phonemize_records.py"
    template = template_path.read_text(encoding="utf-8")
    return template.replace("{{KERNEL_SLUG}}", kernel_slug)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-jobs", type=int, default=2,
                        help="Max kernels to submit per invocation (Kaggle limit: 2 concurrent).")
    parser.add_argument("--wait-and-loop", action="store_true",
                        help="Automatically wait for submitted jobs and loop until all 6 parts are done.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    tokens = kaggle_auth_environment(load_kaggle_api_tokens())
    username = resolve_kaggle_username(tokens.get("KAGGLE_USERNAME"))
    if not username or not kaggle_auth_available(tokens):
        raise RuntimeError("Missing KAGGLE_USERNAME or Kaggle auth")

    os.environ.update({k: v for k, v in tokens.items() if k.startswith("KAGGLE_")})
    kaggle_env = {**os.environ, **tokens, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

    state_file = project_root / "outputs" / "kaggle_phonemize" / "submitted_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    submitted_history: dict[str, str] = {}
    if state_file.exists():
        try:
            submitted_history = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    pending = [(p, ref, slug) for (p, ref, slug) in QUYNHVU03_KERNELS
               if ref not in submitted_history][: args.max_new_jobs]

    if not QUYNHVU03_KERNELS or all(ref in submitted_history for _, ref, _ in QUYNHVU03_KERNELS):
        print("🎉 All 6 parts have already been submitted for phonemization!")
        for _, ref, _ in QUYNHVU03_KERNELS:
            print(f"  {ref}: {submitted_history.get(ref, 'unknown')}")
        return

    print("Phonemize parts status:")
    for p, ref, slug in QUYNHVU03_KERNELS:
        if ref in submitted_history:
            tag = f"[DONE: {submitted_history[ref]}]"
        elif (p, ref, slug) in pending:
            tag = "[PENDING SUBMIT]"
        else:
            tag = "[DEFERRED]"
        print(f"  part {p}: {slug} {tag}")

    cli = kaggle_cli_command()

    # Upload GenMusic source code (contains run_kaggle_phonemize_records.py)
    run_id = f"phonemize-{int(time.time())}"
    run_dir = project_root / "outputs" / "kaggle_phonemize" / run_id
    source_dir = run_dir / "source_dataset"
    kernels_dir = run_dir / "kernels"
    source_dir.mkdir(parents=True, exist_ok=True)
    kernels_dir.mkdir(parents=True, exist_ok=True)

    source_slug = f"genmusic-source-phonemize-{int(time.time())}"
    source_ref = f"{username}/{source_slug}"

    write_source_zip(project_root, source_dir / "genmusic_vn_source.zip")
    (source_dir / "dataset-metadata.json").write_text(
        json.dumps({"title": f"GenMusic Source {run_id}", "id": source_ref,
                    "licenses": [{"name": "other"}]}, indent=2),
        encoding="utf-8",
    )

    print("\n📦 Uploading GenMusic source code...")
    created = _run_cli(cli, ["datasets", "create", "-p", str(source_dir), "-r", "zip"], kaggle_env)
    if created.returncode != 0:
        raise RuntimeError("Could not create GenMusic source dataset on Kaggle")
    _wait_for_dataset(cli, source_ref, kaggle_env)

    submitted: list[tuple[int, str]] = []
    for part, source_kernel_ref, kernel_slug in pending:
        kernel_name = f"genmusic-phonemize-p{part}-{int(time.time())}"
        kernel_ref = f"{username}/{kernel_name}"
        kernel_dir = kernels_dir / f"part{part}"
        kernel_dir.mkdir(parents=True, exist_ok=True)

        # Write the phonemize script with the source slug substituted
        script = _build_kernel_script(kernel_slug)
        (kernel_dir / "run_phonemize.py").write_text(script, encoding="utf-8")
        (kernel_dir / "kernel-metadata.json").write_text(
            json.dumps({
                "id": kernel_ref,
                "title": kernel_name,
                "code_file": "run_phonemize.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": "true",
                "enable_gpu": "true",   # GPU — ByT5 inference 10-20x faster on T4
                "enable_internet": "true",
                "dataset_sources": [source_ref],
                "kernel_sources": [source_kernel_ref],  # ← quynhvu03 kernel output
            }, indent=2),
            encoding="utf-8",
        )

        print(f"\n🚀 Submitting Phonemize Kernel for Part {part} ({source_kernel_ref})...")
        pushed = _run_cli(cli, ["kernels", "push", "-p", str(kernel_dir)], kaggle_env)
        push_ok = pushed.returncode == 0 and "error" not in (pushed.stdout + pushed.stderr).lower()
        if not push_ok:
            raise RuntimeError(f"Kaggle rejected phonemize kernel for part {part}")

        url = f"https://www.kaggle.com/code/{kernel_ref}"
        submitted.append((part, url))
        submitted_history[source_kernel_ref] = url
        state_file.write_text(json.dumps(submitted_history, indent=2), encoding="utf-8")

    print("\n✅ SUBMITTED PHONEMIZE KERNELS:")
    for part, url in submitted:
        print(f"  Part {part}: {url}")

    if args.wait_and_loop and submitted:
        print("\n⏳ [--wait-and-loop] Watching submitted kernels...")
        submitted_refs = [url.split("kaggle.com/code/")[-1] for _, url in submitted]
        while submitted_refs:
            time.sleep(30)
            submitted_refs = [r for r in submitted_refs
                              if not _is_kernel_finished(cli, r, kaggle_env)]
            if submitted_refs:
                print(f"  [Waiting] {len(submitted_refs)} job(s) still running...", flush=True)
        print("🎉 Batch done! Submitting next batch...\n")
        main()


if __name__ == "__main__":
    main()
