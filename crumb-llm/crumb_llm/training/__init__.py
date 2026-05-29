"""Experimental training utilities for a local scratch model.

NOTHING here runs automatically. CrumbLLM never trains a model by default.
These modules exist so power users can (a) export their CRUMB corpus as a
dataset, (b) prepare it for supervised fine-tuning, and (c) optionally train a
tiny from-scratch transformer for the experimental ``scratch`` provider.

Design cannibalized from FareedKhan-dev/train-llm-from-scratch:
  - jsonl dataset export,
  - SFT pair concatenation with an end-of-text marker,
  - a minimal decoder-only transformer + AdamW training loop.

Heavy deps (torch, tiktoken) are imported lazily so importing this package is
always cheap and safe.
"""
