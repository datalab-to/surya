"""DPR (Detailed Project Report) AST pipeline, built on top of surya's OCR.

Turns a raw surya OCR results.json into a section/subsection hierarchy with
figure captions, exported tables, structured parameters, and resolved
cross-references. See process_pdf.py for the end-to-end entry point.

Run any module standalone with `uv run python -m dpr_pipeline.<module> ...`
(see each module's own docstring for its specific usage).
"""
