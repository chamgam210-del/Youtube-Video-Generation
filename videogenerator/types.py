"""Compatibility shim.

This project originally had a module named `videogenerator.types` for dataclasses.

Unfortunately, some Python environments end up with the `videogenerator/` directory
on `sys.path`. In that case, importing the standard library module `types`
accidentally resolves to this file (because it's named `types.py`), which breaks
the interpreter (e.g. `enum` expects `MappingProxyType`).

To keep the project working:
- If this file is imported as top-level module name `types`, we dynamically load
  the real stdlib `types.py` by file path and mirror its symbols.
- If imported as `videogenerator.types`, we re-export the project dataclasses
  from `videogenerator.models`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_stdlib_types() -> None:
    stdlib_dir = None
    try:
        # Preferred: sysconfig knows the stdlib path
        import sysconfig

        stdlib_dir = sysconfig.get_paths().get("stdlib")
    except Exception:
        stdlib_dir = None

    candidates: list[Path] = []
    if stdlib_dir:
        candidates.append(Path(stdlib_dir) / "types.py")

    # Fallbacks (work in many Windows installs, including conda)
    candidates.extend(
        [
            Path(sys.base_prefix) / "Lib" / "types.py",
            Path(sys.prefix) / "Lib" / "types.py",
        ]
    )

    types_path = next((p for p in candidates if p.exists()), None)
    if not types_path:
        raise ImportError("Could not locate stdlib types.py for compatibility shim")

    spec = importlib.util.spec_from_file_location("_videogenerator_stdlib_types", types_path)
    if not spec or not spec.loader:
        raise ImportError("Could not load spec for stdlib types.py")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[call-arg]

    # Mirror everything into this module's globals
    globals().update({k: v for k, v in mod.__dict__.items() if k not in {"__name__"}})


if __name__ == "types":
    _load_stdlib_types()
else:
    # Legacy re-exports
    from .models import Slide, TranscriptSegment

    __all__ = ["TranscriptSegment", "Slide"]
