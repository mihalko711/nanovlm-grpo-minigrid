"""
Shared utilities for evaluation: image string construction and action parsing.
"""

from env_utils import ACTION_NAMES, text_to_action


def make_image_string(tokenizer, n_h, n_w, mp_len):
    parts = []
    if hasattr(tokenizer, "global_image_token"):
        parts.append(tokenizer.global_image_token)
        parts.append(tokenizer.image_token * mp_len)
        if n_h == 1 and n_w == 1:
            return "".join(parts)
    for i in range(n_h):
        for j in range(n_w):
            parts.append(getattr(tokenizer, f"r{i+1}c{j+1}"))
            parts.append(tokenizer.image_token * mp_len)
    return "".join(parts)


def parse_action(text):
    text = text.strip().lower()
    for name in ACTION_NAMES:
        if name in text:
            return text_to_action(name)
    return None
