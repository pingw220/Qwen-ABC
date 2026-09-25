"""Paper-final evaluation layer (reports/paper_final/).

Additive on top of qwen_abc/ and scripts/: it reuses the parser, metrics, prompt
formats and best-of-n selector, and adds paired-seed sampling, control
interventions, OOD plans and the cross-cutting analyses the paper needs.
"""
