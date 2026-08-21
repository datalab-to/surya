"""Tiny content-hash JSON cache shared by the OCR and figure-captioning steps.

Both are expensive (minutes per page / tens of seconds per image on CPU) and
both are pure functions of their input bytes plus a few config knobs, so a
flat hash -> cached-result-file cache is all that's needed — no invalidation
logic beyond "the input changed".
"""

import hashlib
import json
import os

CACHE_ROOT = ".surya_cache"


def _cache_path(namespace: str, key: str) -> str:
    return os.path.join(CACHE_ROOT, namespace, key[:2], key + ".json")


def hash_bytes(*parts: bytes) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
        h.update(b"\0")
    return h.hexdigest()


def get(namespace: str, key: str):
    path = _cache_path(namespace, key)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set(namespace: str, key: str, value) -> None:
    path = _cache_path(namespace, key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False)
