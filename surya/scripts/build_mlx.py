"""Build an MLX (Apple Silicon) copy of a Surya checkpoint, optionally quantized.

Runs `mlx_vlm convert`, then repairs two things the converter gets wrong for the character-level Surya vocabulary:
  - tokenizer: mlx-vlm re-saves tokenizer.json / tokenizer_config.json through transformers, which rebuilds the
    tokenizer as a byte-level BPE (Qwen3_5Tokenizer, ByteLevel decoder). Generation then crashes (KeyError ' ') or
    decodes wrongly. The original files are copied back over the converted ones.
  - end token: config.json carries a stale eos_token_id (248044, outside the vocabulary), so generation never stops at
    <|im_end|>. The id is looked up from the tokenizer and written to config.json (top level and text_config).

The original checkpoint is never modified (outputs that overlap it are refused; a non-empty output needs --overwrite). Needs `pip install mlx-vlm` (Apple Silicon only).

Usage:
    python surya/scripts/build_mlx.py --checkpoint datalab-to/surya-ocr-2.5 --out-dir ./mlx-build --q-bits 8
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Tokenizer assets the converter may rewrite; whichever exist in the source are copied back unchanged.
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.json",
    "chat_template.jinja",
    "added_tokens.json",
)
REQUIRED_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


def resolve_checkpoint(checkpoint: str) -> Path:
    p = Path(checkpoint)
    if p.exists():
        return p.resolve()
    from huggingface_hub import snapshot_download

    print(f"[download] {checkpoint}")
    return Path(snapshot_download(checkpoint))


def eos_token_id(src: Path) -> int:
    """Id of the checkpoint's eos token (tokenizer_config.json `eos_token`), from tokenizer.json."""
    cfg = json.loads((src / "tokenizer_config.json").read_text())
    eos = cfg.get("eos_token")
    if isinstance(eos, dict):
        eos = eos.get("content")
    if not eos:
        raise RuntimeError("tokenizer_config.json has no eos_token")
    tok = json.loads((src / "tokenizer.json").read_text())
    for added in tok.get("added_tokens", []):
        if added.get("content") == eos:
            return int(added["id"])
    vocab = tok.get("model", {}).get("vocab", {})
    if eos in vocab:
        return int(vocab[eos])
    raise RuntimeError(f"eos token {eos!r} not found in tokenizer.json")


def _filter_ids(value, eos: int, vocab_size):
    """Keep ids that are inside the vocabulary; always include `eos`. Preserves int vs list."""
    ids = value if isinstance(value, list) else [value]
    kept = [
        i
        for i in ids
        if isinstance(i, int) and (vocab_size is None or 0 <= i < vocab_size)
    ]
    if eos not in kept:
        kept.insert(0, eos)
    return kept if isinstance(value, list) and len(kept) > 1 else kept[0]


def repair(src: Path, dst: Path) -> int:
    """Copy the original tokenizer assets over the converted ones and fix the eos id in config.json and
    generation_config.json (mlx-vlm overwrites the config eos from generation_config.json at load time).
    Out-of-vocabulary stop ids are dropped. Returns the eos id."""
    for name in TOKENIZER_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    eos = eos_token_id(src)
    cfg_path = dst / "config.json"
    cfg = json.loads(cfg_path.read_text())
    text_cfg = (
        cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else None
    )
    vocab_size = (text_cfg or cfg).get("vocab_size")
    cfg["eos_token_id"] = eos
    if text_cfg is not None:
        text_cfg["eos_token_id"] = eos
    cfg_path.write_text(json.dumps(cfg, indent=2))
    gen_path = dst / "generation_config.json"
    if gen_path.exists():
        gen = json.loads(gen_path.read_text())
        gen["eos_token_id"] = _filter_ids(gen.get("eos_token_id", eos), eos, vocab_size)
        gen_path.write_text(json.dumps(gen, indent=2))
    return eos


def check_paths(src: Path, dst: Path, overwrite: bool) -> None:
    """Refuse destinations that would delete or alias the source checkpoint, or clobber unrelated files."""
    src, dst = src.resolve(), dst.resolve()
    if dst == src or src in dst.parents or dst in src.parents:
        raise SystemExit(f"refusing: output {dst} overlaps the source checkpoint {src}")
    if dst.exists() and any(dst.iterdir()) and not overwrite:
        raise SystemExit(
            f"refusing: {dst} is not empty (pass --overwrite to replace it)"
        )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--checkpoint", required=True, help="HF repo id or local checkpoint dir"
    )
    ap.add_argument("--out-dir", type=Path, default=Path("./mlx-build"))
    ap.add_argument(
        "--name",
        default=None,
        help="Output folder name (default: <checkpoint>-mlx[-q<bits>])",
    )
    ap.add_argument(
        "--q-bits",
        type=int,
        choices=(4, 8),
        default=None,
        help="Quantize to this many bits (default: bf16)",
    )
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument(
        "--overwrite", action="store_true", help="Replace a non-empty output folder"
    )
    args = ap.parse_args()

    src = resolve_checkpoint(args.checkpoint)
    name = args.name or (
        src.name + "-mlx" + (f"-q{args.q_bits}" if args.q_bits else "")
    )
    final = args.out_dir / name
    check_paths(src, final, args.overwrite)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Build next to the final folder and rename at the end, so a failed build never leaves a half-repaired model.
    dst = args.out_dir / f".{name}.partial"
    if dst.exists():
        shutil.rmtree(dst)

    cmd = [
        sys.executable,
        "-m",
        "mlx_vlm",
        "convert",
        "--hf-path",
        str(src),
        "--mlx-path",
        str(dst),
        "--dtype",
        args.dtype,
    ]
    if args.q_bits:
        cmd += ["-q", "--q-bits", str(args.q_bits)]
    print("[convert]", " ".join(cmd))
    subprocess.check_call(cmd)

    eos = repair(src, dst)
    for name_ in REQUIRED_TOKENIZER_FILES:
        assert sha(src / name_) == sha(dst / name_), f"{name_} was not restored"
    if final.exists():
        shutil.rmtree(final)
    dst.rename(final)
    print(f"[repair] original tokenizer files restored, eos_token_id set to {eos}")
    print(f"  MLX model: {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
