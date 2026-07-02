"""Poster-quality output for the SPA figure scripts.

`savefig_poster(fig, out)` writes a 300-DPI PNG *and* a vector PDF sibling (crisp at any poster
scale, with editable text). It preserves each figure's existing layout — no font/style changes, so
it is safe to apply to the already-laid-out panels. Import from any `scripts/eval/plot_*.py` (the
script's own directory is on `sys.path` when run as `python scripts/eval/plot_x.py`).
"""

from __future__ import annotations

from pathlib import Path


def savefig_poster(fig, out) -> list[str]:
    """Write ``out`` as both a 300-DPI PNG and a vector PDF sibling; return the paths written."""
    import matplotlib

    matplotlib.rcParams["pdf.fonttype"] = 42     # embed editable TrueType text in the PDF (not outlines)
    matplotlib.rcParams["svg.fonttype"] = "none"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for ext in (".png", ".pdf"):
        p = str(out.with_suffix(ext))
        fig.savefig(p, dpi=300)                   # PNG at 300 DPI; PDF is vector (dpi irrelevant for vectors)
        written.append(p)
    print(f"[poster] wrote {written[0]} + {written[1]}")
    return written
