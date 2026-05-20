# -*- coding: utf-8 -*-
"""Hugging Face cache configuration for local model weights."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HF_HOME_PATH = PROJECT_ROOT / "data" / "hf_cache"
HF_HUB_CACHE_PATH = HF_HOME_PATH / "hub"


def configure_hf_cache() -> None:
    """Force Hugging Face libraries to use local cache only."""
    HF_HUB_CACHE_PATH.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(HF_HOME_PATH)
    os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE_PATH)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    # If huggingface_hub was imported before this function runs, its constants
    # have already been evaluated. Patch them too so long-running backend
    # processes do not fall back to C:\\Users\\...\\.cache.
    try:
        import huggingface_hub.constants as hf_constants

        hf_constants.HF_HOME = str(HF_HOME_PATH)
        hf_constants.HF_HUB_CACHE = str(HF_HUB_CACHE_PATH)
        if hasattr(hf_constants, "HF_HUB_OFFLINE"):
            hf_constants.HF_HUB_OFFLINE = True
    except Exception:
        pass

    try:
        import transformers.utils.hub as transformers_hub

        transformers_hub.HF_MODULES_CACHE = str(HF_HOME_PATH / "modules")
    except Exception:
        pass
