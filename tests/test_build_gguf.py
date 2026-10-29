"""build_gguf converter-patch helpers (no llama.cpp, no model): the pre-tokenizer hash is registered per checkpoint."""

from surya.scripts import build_gguf

FAKE_CONVERT = f"""
    def get_vocab_base_pre(self, tokenizer) -> str:
        chktxt = '\\n \\n\\n 3 33 333 \\u200d (normal)'
        chktok = tokenizer.encode(chktxt)
{build_gguf._PATCH_REGISTER_HASH_ANCHOR}
        raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")

{build_gguf._PATCH_VOCAB_ANCHOR}
"""


def test_extract_chktxt_returns_the_literal():
    assert build_gguf.extract_chktxt(FAKE_CONVERT) == "\n \n\n 3 33 333 ‍ (normal)"


def test_register_hash_patch_inserts_default_pre_tokenizer(tmp_path):
    f = tmp_path / "convert_hf_to_gguf.py"
    f.write_text(FAKE_CONVERT)
    h = "ab" * 32
    build_gguf.patch_converter(f, [build_gguf.register_hash_patch(h, "test tokenizer")])
    text = f.read_text()
    assert f'if chkhsh == "{h}":' in text and text.count('res = "default"') >= 1
    before = text
    build_gguf.patch_converter(
        f, [build_gguf.register_hash_patch(h, "test tokenizer")]
    )  # idempotent
    assert f.read_text() == before


def test_rename_eos_patch_applies_after_vocab_patch(tmp_path):
    f = tmp_path / "convert_hf_to_gguf.py"
    f.write_text(FAKE_CONVERT)
    build_gguf.patch_converter(f, [build_gguf.rename_eos_patch()])
    text = f.read_text()
    assert '"<|im_end!>" if t == "<|im_end|>" else t for t in tokens' in text
    assert text.index("_maybe_encode_gpt2_bytes(tokens, toktypes)") < text.index(
        "<|im_end!>"
    )
    before = text
    build_gguf.patch_converter(f, [build_gguf.rename_eos_patch()])  # idempotent
    assert f.read_text() == before


class _Tok:
    eos_token_id = 2

    def __init__(self, spell_if):
        self.spell_if = spell_if

    def encode(self, text, add_special_tokens=False):
        return [10, 11] if self.spell_if(text) else [10, 2]


def test_decide_eos_rename():
    prompts = {"a": "A.", "b": "B."}
    assert (
        build_gguf.decide_eos_rename(
            build_gguf.eos_spelling_report(_Tok(lambda t: True), prompts)
        )
        is True
    )
    assert (
        build_gguf.decide_eos_rename(
            build_gguf.eos_spelling_report(_Tok(lambda t: False), prompts)
        )
        is False
    )


def test_decide_eos_rename_refuses_mixed():
    import pytest

    report = build_gguf.eos_spelling_report(
        _Tok(lambda t: t.startswith("A")), {"a": "A.", "b": "B."}
    )
    with pytest.raises(RuntimeError, match="some package prompts"):
        build_gguf.decide_eos_rename(report)


def test_rename_preconditions(tmp_path):
    import json

    import pytest

    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {"model": {"type": "WordLevel", "vocab": {"a": 0}}, "added_tokens": []}
        )
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<|im_end|>"})
    )
    build_gguf.check_rename_preconditions(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"eos_token": "</s>"}))
    with pytest.raises(RuntimeError, match="expected eos_token"):
        build_gguf.check_rename_preconditions(tmp_path)
    (tmp_path / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "BPE", "vocab": {}}, "added_tokens": []})
    )
    with pytest.raises(RuntimeError, match="WordLevel"):
        build_gguf.check_rename_preconditions(tmp_path)


def test_patches_do_not_leak_between_builds(tmp_path):
    """A checkpoint that needs the eos rename, then one that does not, in the same llama.cpp checkout: the rename must not carry over."""
    import subprocess

    repo = tmp_path / "llama.cpp"
    repo.mkdir()
    conv = repo / "convert_hf_to_gguf.py"
    conv.write_text(FAKE_CONVERT)
    for cmd in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.email=a@b", "-c", "user.name=t", "commit", "-q", "-m", "x"],
    ):
        subprocess.check_call(["git", "-C", str(repo), *cmd])
    build_gguf.reset_converter(repo)
    build_gguf.patch_converter(conv, [build_gguf.rename_eos_patch()])
    assert "<|im_end!>" in conv.read_text()
    build_gguf.reset_converter(repo)  # next build starts pristine
    build_gguf.patch_converter(conv, [])
    assert "<|im_end!>" not in conv.read_text()
    build_gguf.reset_converter(repo)
    build_gguf.patch_converter(
        conv, [build_gguf.rename_eos_patch()]
    )  # and the reverse order
    assert "<|im_end!>" in conv.read_text()
