"""Manual smoke test for surya.figure_caption.analyze_image.

Not part of the pytest suite (deliberately) — it downloads and loads a ~1B
parameter model on first run, which is too slow/heavy to run on every test
pass. Run directly instead:

    uv run python -m dpr_pipeline.test_figure_caption [image_path]

Defaults to static/images/corporate.png (a real sample document page already
in the repo) if no path is given.
"""

import sys

from surya.figure_caption import analyze_image

DEFAULT_IMAGE = "static/images/corporate.png"

TEST_PROMPTS = [
    "Describe this image in detail.",
    "Extract all visible text.",
    "Explain this engineering drawing.",
    "Analyze this DPR page.",
    "Extract all engineering parameters.",
    "Return the result as structured JSON.",
]


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE

    for prompt in TEST_PROMPTS:
        print(f"\n{'=' * 80}\nPROMPT: {prompt}\n{'-' * 80}")
        result = analyze_image(image_path, prompt)
        if result["success"]:
            print(result["response"])
        else:
            print(f"[FAILED] {result['error']}")


if __name__ == "__main__":
    main()
