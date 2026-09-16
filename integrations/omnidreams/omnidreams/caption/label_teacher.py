# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 2 teacher labeling: Cosmos-Reason1-7B over captured caption windows.

Runs the Cosmos-Reason1-7B VLM (Qwen2.5-VL backbone) over each window's 8 RGB
frames and emits a closed-set label over the 6 ego-motion classes — a hard
argmax plus a soft 6-way distribution (the KL distillation target from Phase
0.4). Reads the WebDataset tar shards written by ``capture.py`` and writes one
JSONL line per window into ``labeled/labels.jsonl`` (resumable).

Soft distribution: the assistant is prompted to answer with a single letter
(A-F); we generate one token at temperature 1.0 and read the top-k logprobs at
that position, matching decoded tokens back to the letters and softmax-ing over
the six. Hard label = argmax of that distribution.

RUN IN A DEDICATED ENV. vLLM pins its own torch/transformers, which conflict
with the omnidreams runtime deps, so this file imports NOTHING from
omnidreams/flashdreams and is meant to be run by path in a vLLM venv::

    uv venv .venv-teacher && . .venv-teacher/bin/activate
    uv pip install "vllm>=0.8.5" "qwen-vl-utils>=0.0.11"
    HF_HOME=<same cache as omnidreams-prepare> \
      python integrations/omnidreams/omnidreams/caption/label_teacher.py \
        --shards-dir ~/data/omnidreams-caption/raw \
        --out ~/data/omnidreams-caption/labeled --limit 50   # smoke first

The Cosmos-Reason1-7B snapshot is already cached by ``omnidreams-prepare`` at
``DEFAULT_REVISION`` (~14 GB); point ``HF_HOME`` at that cache to reuse it.
"""

from __future__ import annotations

import argparse
import json
import math
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

#: The 6-class ego-motion bank (Phase 0), presented as options A-F.
CLASSES = [
    "driving_straight",
    "turning_left",
    "turning_right",
    "slowing_or_stopping",
    "stopped",
    "reversing",
]
LETTERS = ["A", "B", "C", "D", "E", "F"]
MENU = "\n".join(f"{L}. {c.replace('_', ' ')}" for L, c in zip(LETTERS, CLASSES))

# Cosmos-Reason1.1 SFT checkpoint already staged by omnidreams-prepare (cached,
# fully-capable VLM, version-locked with the world model's text encoder). Pass
# --revision main for the published card revision (requires a ~14 GB download).
DEFAULT_REVISION = "3210bec0495fdc7a8d3dbb8d58da5711eab4b423"

SYSTEM = (
    "You are a precise driving-motion classifier. You are shown a short clip "
    "from a vehicle's front camera. Answer with a single letter only."
)
USER = (
    "Classify the ego-vehicle's own motion in this clip.\n{menu}\n"
    "Reply with only the letter of the single best option."
)


def iter_windows(shards_dir: Path):
    """Yield ``(key, shard, rgb_fields)`` for every window with RGB frames."""
    manifest = shards_dir / "manifest.jsonl"
    with manifest.open() as fh:
        for line in fh:
            d = json.loads(line)
            rgb = sorted(f for f in d.get("fields", []) if f.startswith("rgb."))
            if rgb:
                yield d["__key__"], d["shard"], rgb


def load_done(out_path: Path) -> set[str]:
    """Keys already labeled, so a re-run resumes instead of redoing them."""
    done: set[str] = set()
    if out_path.exists():
        with out_path.open() as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["__key__"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def extract_frames(
    tar: tarfile.TarFile, key: str, rgb_fields: list[str], tmpdir: Path
) -> list[str]:
    """Extract a window's JPEG frames to ``tmpdir``; return file:// paths."""
    paths = []
    for field in rgb_fields:
        data = tar.extractfile(f"{key}.{field}").read()
        p = tmpdir / f"{key}.{field}"
        p.write_bytes(data)
        paths.append(p.resolve().as_uri())
    return paths


def build_input(proc, process_vision_info, frame_paths: list[str], fps: int) -> dict:
    """Assemble one vLLM request: chat prompt + processed video frames."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER.format(menu=MENU)},
                {"type": "video", "video": frame_paths, "fps": fps},
            ],
        },
    ]
    prompt = proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    # Some qwen-vl-utils versions return fps as a per-video list, but the
    # Qwen2.5-VL processor (transformers v5+) validates it as a scalar.
    fps_kw = video_kwargs.get("fps")
    if isinstance(fps_kw, (list, tuple)) and fps_kw:
        video_kwargs["fps"] = fps_kw[0]
    mm: dict = {}
    if image_inputs:
        mm["image"] = image_inputs
    if video_inputs:
        mm["video"] = video_inputs
    return {
        "prompt": prompt,
        "multi_modal_data": mm,
        "mm_processor_kwargs": video_kwargs,
    }


def soft_hard(out) -> tuple[dict[str, float], str | None, float, int]:
    """Turn one generation's answer-token logprobs into a 6-way distribution.

    Matches the top-k decoded tokens back to the letters A-F (robust to leading
    spaces / case), sums each letter's probability mass, and renormalizes.
    Returns ``(dist, hard, confidence, letters_found)``.
    """
    logprobs = out.outputs[0].logprobs[0]  # {token_id: Logprob} at answer pos
    mass = {L: 0.0 for L in LETTERS}
    for _tid, lp in logprobs.items():
        tok = (lp.decoded_token or "").strip().upper()
        if tok in mass:
            mass[tok] += math.exp(lp.logprob)
    total = sum(mass.values())
    if total <= 0.0:  # model emitted no letter in the top-k (non-compliant)
        uniform = {c: 1.0 / len(CLASSES) for c in CLASSES}
        return uniform, None, 0.0, 0
    dist = {c: mass[L] / total for c, L in zip(CLASSES, LETTERS)}
    hard = max(dist, key=dist.get)
    found = sum(1 for v in mass.values() if v > 0.0)
    return dist, hard, dist[hard], found


def main() -> None:
    ap = argparse.ArgumentParser(description="Cosmos-Reason1-7B teacher labeling.")
    ap.add_argument("--shards-dir", required=True, help="Capture raw/ shard dir.")
    ap.add_argument("--out", required=True, help="Output dir for labels.jsonl.")
    ap.add_argument("--model", default="nvidia/Cosmos-Reason1-7B")
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--fps", type=int, default=4, help="Cosmos trains on fps=4.")
    ap.add_argument("--logprobs", type=int, default=20)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0, help="Label only N (smoke test).")
    ap.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Skip CUDA-graph capture: much faster startup, slightly slower "
        "per-step. Good for smoke tests; omit for the full sweep.",
    )
    args = ap.parse_args()

    # Heavy deps live only in the dedicated vLLM env; import them here.
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    shards_dir = Path(args.shards_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "labels.jsonl"

    (out_dir / "label_config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "revision": args.revision,
                "classes": CLASSES,
                "letters": LETTERS,
                "fps": args.fps,
                "system": SYSTEM,
                "user": USER.format(menu=MENU),
            },
            indent=2,
        )
    )

    done = load_done(out_path)
    by_shard: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    for key, shard, rgb in iter_windows(shards_dir):
        if key not in done:
            by_shard[shard].append((key, rgb))
    remaining = sum(len(v) for v in by_shard.values())
    print(f"[label] {len(done)} already done; {remaining} to label", flush=True)

    llm = LLM(
        model=args.model,
        revision=args.revision,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 10, "video": 10},
        max_logprobs=max(32, args.logprobs),
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
    )
    proc = AutoProcessor.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True
    )
    sp = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1, logprobs=args.logprobs
    )

    labeled = 0
    hard_hist: Counter = Counter()
    found_hist: Counter = Counter()
    agree = 0
    stop = False
    with out_path.open("a") as out_fh:
        for shard in sorted(by_shard):
            if stop:
                break
            items = by_shard[shard]
            tar = tarfile.open(shards_dir / shard)
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                keys, inputs, hints = [], [], []
                for key, rgb in items:
                    if args.limit and labeled + len(keys) >= args.limit:
                        break
                    paths = extract_frames(tar, key, rgb, tmp)
                    inputs.append(
                        build_input(proc, process_vision_info, paths, args.fps)
                    )
                    keys.append(key)
                if not inputs:
                    tar.close()
                    continue
                outs = llm.generate(inputs, sampling_params=sp)
                for key, out in zip(keys, outs):
                    dist, hard, conf, found = soft_hard(out)
                    out_fh.write(
                        json.dumps(
                            {
                                "__key__": key,
                                "teacher_hard": hard,
                                "teacher_soft": {
                                    c: round(p, 5) for c, p in dist.items()
                                },
                                "teacher_conf": round(conf, 5),
                                "letters_found": found,
                            }
                        )
                        + "\n"
                    )
                    labeled += 1
                    hard_hist[hard] += 1
                    found_hist[found] += 1
                out_fh.flush()
            tar.close()
            print(f"[label] {labeled} labeled (shard {shard})", flush=True)
            if args.limit and labeled >= args.limit:
                stop = True

    print(f"\n[label] done: {labeled} windows -> {out_path}")
    print("[label] teacher_hard distribution:", dict(hard_hist))
    print("[label] letters_found histogram:", dict(found_hist))


if __name__ == "__main__":
    main()
