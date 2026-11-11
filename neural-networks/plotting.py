"""Shared matplotlib styling so every figure in the tutorial looks the same."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render straight to files, no window needed
import matplotlib.pyplot as plt

IMAGES = Path(__file__).parent / "images"

BLUE = "#2a78d6"  # the data
ORANGE = "#eb6834"  # the model / the thing we compare against
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "font.size": 10,
    }
)


def save(fig, name):
    IMAGES.mkdir(exist_ok=True)
    path = IMAGES / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path.relative_to(Path(__file__).parent)}")
