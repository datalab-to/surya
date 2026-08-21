"""Ask a natural-language question over an AST tree, e.g. "what's the CBR
design value for the Merangkong road?", instead of manually walking the
hierarchy to find it.

Usage:
    uv run python -m dpr_pipeline.ask ast.json "What is the resilient modulus for CBR 5%?"

Retrieval is TF-IDF over each section's own text (no embedding model/extra
heavy dependency needed for this) — the top-K matching sections are handed
to a text LLM (gpt-oss-20b, the same VLLM_* endpoint given for figure
captioning; confirmed then that it's text-only, which is exactly what
retrieval-augmented Q&A over already-extracted text needs) with instructions
to answer *only* from those excerpts and cite which section each fact came
from, rather than letting the model improvise from its own training data.
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter

import requests
from dotenv import load_dotenv

from .diff_ast import own_text

load_dotenv("local.env")

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]*")
TOP_K = 6

ANSWER_PROMPT = """Answer the question using ONLY the excerpts below, which are sections from a document. \
For every fact you use, cite its section in brackets, e.g. [4.9.1. For CBR 5%]. \
If the excerpts don't contain the answer, say so plainly instead of guessing.

Excerpts:
{excerpts}

Question: {question}"""


def tokenize(text: str) -> list:
    return [w.lower() for w in WORD_RE.findall(text)]


def collect_chunks(tree: dict) -> list:
    chunks = []

    def walk(node, path):
        if node.get("type") == "section":
            text = own_text(node)
            title = node.get("title")
            full_path = path + (title,)
            if text:
                chunks.append({"path": " > ".join(p for p in full_path if p), "title": title, "page": node.get("page"), "text": text})
        else:
            full_path = path

        for child in node.get("children", []):
            walk(child, full_path)

    walk(tree, ())
    return chunks


def rank_chunks(question: str, chunks: list, top_k: int = TOP_K) -> list:
    doc_freq = Counter()
    chunk_tokens = []
    for chunk in chunks:
        tokens = tokenize(chunk["text"])
        chunk_tokens.append(tokens)
        doc_freq.update(set(tokens))

    n_docs = max(len(chunks), 1)
    idf = {term: math.log((n_docs + 1) / (df + 1)) + 1 for term, df in doc_freq.items()}

    query_terms = tokenize(question)
    scored = []
    for chunk, tokens in zip(chunks, chunk_tokens):
        tf = Counter(tokens)
        total = sum(tf.values()) or 1
        score = sum((tf.get(term, 0) / total) * idf.get(term, 0) for term in query_terms)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def ask_llm(question: str, chunks: list) -> str:
    excerpts = "\n\n".join(f"[{c['path']}] (page {c['page']})\n{c['text']}" for c in chunks)
    prompt = ANSWER_PROMPT.format(excerpts=excerpts, question=question)

    base_url = os.environ.get("VLLM_BASE_URL", "http://43.242.38.16:8009/v1")
    model = os.environ.get("VLLM_MODEL", "openai/gpt-oss-20b")

    resp = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "temperature": float(os.environ.get("VLLM_TEMPERATURE", 0.0)),
            "max_tokens": int(os.environ.get("VLLM_MAX_TOKENS", 2048)),
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def ask(tree: dict, question: str) -> dict:
    chunks = collect_chunks(tree)
    top_chunks = rank_chunks(question, chunks)
    if not top_chunks:
        return {"answer": "No section in this document matches the question closely enough to answer from.", "sources": []}

    answer = ask_llm(question, top_chunks)
    return {
        "answer": answer,
        "sources": [{"path": c["path"], "page": c["page"]} for c in top_chunks],
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Ask a question over an AST tree")
    parser.add_argument("tree_json")
    parser.add_argument("question")
    args = parser.parse_args()

    with open(args.tree_json, "r", encoding="utf-8") as f:
        tree = json.load(f)

    result = ask(tree, args.question)
    print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"  [p{source['page']}] {source['path']}")
