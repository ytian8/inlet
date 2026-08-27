"""Inlet -- Text-to-Prompt.

A hypernetwork that maps a natural-language task description to input-layer
soft-prompt vectors for a FROZEN base model. The black-box-serviceable
counterpart to Text-to-LoRA: T2L writes into the weights, Inlet writes into the
input, which is the only surface an inference API actually exposes.
"""

__version__ = "0.1.0"
