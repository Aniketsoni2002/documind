"""Streamlit Community Cloud entrypoint.

Streamlit Cloud runs the app from the repo root and does not `pip install -e .`,
so the `src/` layout isn't importable by default. We add `src/` to the path,
then hand off to the real UI module.

Point the Streamlit Cloud app's "Main file path" at this file.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Execute the UI module as if it were the script Streamlit launched.
runpy.run_module("documind.ui.app", run_name="__main__")
