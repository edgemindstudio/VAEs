# vaes/sample.py

from __future__ import annotations
from typing import Dict

# Reuse the real implementation that lives in `vae/sample.py`
from vae.sample import synth as _synth

def synth(cfg: dict, output_root: str, seed: int = 42) -> Dict:
    return _synth(cfg, output_root, seed)
