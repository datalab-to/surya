"""build_mlx repair helpers (no mlx, no model): the converter's tokenizer rewrite and stale eos id are undone."""

import json

from surya.scripts import build_mlx


def _src(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {"type": "WordLevel", "vocab": {"a": 0, "<|im_end|>": 2}},
                "added_tokens": [{"id": 2, "content": "<|im_end|>"}],
                "decoder": {"type": "Fuse"},
            }
        )
    )
    (src / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "TokenizersBackend", "eos_token": "<|im_end|>"})
    )
    return src


def test_eos_token_id_from_tokenizer(tmp_path):
    assert build_mlx.eos_token_id(_src(tmp_path)) == 2


def test_repair_restores_tokenizer_and_fixes_eos(tmp_path):
    src = _src(tmp_path)
    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "BPE"}, "decoder": {"type": "ByteLevel"}})
    )
    (dst / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen3_5Tokenizer"})
    )
    (dst / "config.json").write_text(
        json.dumps(
            {
                "eos_token_id": 248044,
                "text_config": {"eos_token_id": 248044, "vocab_size": 88526},
            }
        )
    )
    assert build_mlx.repair(src, dst) == 2
    assert build_mlx.sha(src / "tokenizer.json") == build_mlx.sha(
        dst / "tokenizer.json"
    )
    cfg = json.loads((dst / "config.json").read_text())
    assert (
        cfg["eos_token_id"] == 2
        and cfg["text_config"]["eos_token_id"] == 2
        and cfg["text_config"]["vocab_size"] == 88526
    )


def test_repair_fixes_generation_config_and_drops_out_of_vocab_ids(tmp_path):
    src = _src(tmp_path)
    (src / "chat_template.jinja").write_text("tmpl")
    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "config.json").write_text(
        json.dumps({"eos_token_id": 248044, "text_config": {"vocab_size": 100}})
    )
    (dst / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [248044, 7], "top_k": 1})
    )
    (dst / "tokenizer.json").write_text("{}")
    (dst / "tokenizer_config.json").write_text("{}")
    build_mlx.repair(src, dst)
    gen = json.loads((dst / "generation_config.json").read_text())
    assert gen["eos_token_id"] == [2, 7] and gen["top_k"] == 1
    assert (dst / "chat_template.jinja").read_text() == "tmpl"


def test_check_paths_refuses_overlap_and_nonempty(tmp_path):
    import pytest

    src = _src(tmp_path)
    for bad in (src, src / "out", tmp_path):
        with pytest.raises(SystemExit, match="overlaps"):
            build_mlx.check_paths(src, bad, overwrite=True)
    alias = tmp_path / "alias"
    alias.symlink_to(src)
    with pytest.raises(SystemExit, match="overlaps"):
        build_mlx.check_paths(src, alias, overwrite=True)
    out = tmp_path / "out_nonempty"
    out.mkdir()
    (out / "x").write_text("1")
    with pytest.raises(SystemExit, match="not empty"):
        build_mlx.check_paths(src, out, overwrite=False)
    build_mlx.check_paths(src, out, overwrite=True)
    build_mlx.check_paths(src, tmp_path / "fresh", overwrite=False)
