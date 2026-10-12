"""Shared colors for every vortex-comparison visualization.

The palette matches the teal-background reference rendering: low values are
teal, coherent-vortex strength rises through green to yellow, streamlines are
white, detected boundaries are yellow, and detected centres are red.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import FancyArrowPatch


_VIRIDIS = plt.get_cmap("viridis")

# Starting viridis at 0.50 makes zero-valued fluid and uncovered regions use
# the same teal as the supplied reference instead of viridis' dark purple.
FIELD_CMAP = LinearSegmentedColormap.from_list(
    "vortex_teal_to_yellow",
    _VIRIDIS(np.linspace(0.50, 1.0, 256)),
)
BACKGROUND_COLOR = FIELD_CMAP(0.0)

STREAMLINE_COLOR = "white"
PDF_STREAMLINE_WIDTH = 0.10
THRESHOLD_COLOR = "#ffd400"
OBSTACLE_FACE_COLOR = "#111111"
OBSTACLE_EDGE_COLOR = "white"


def streamline_color(alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Return the shared white streamline color with embedded transparency."""

    return to_rgba(STREAMLINE_COLOR, alpha=alpha)


def thin_streamplots_for_pdf(streamplots: list[object]) -> None:
    """Use a finer common linewidth for streamlines in vector PDF output."""

    for streamplot in streamplots:
        streamplot.lines.set_visible(True)
        streamplot.lines.set_linewidth(PDF_STREAMLINE_WIDTH)
        streamplot.arrows.set_visible(True)
        streamplot.arrows.set_linewidth(PDF_STREAMLINE_WIDTH)
        axis = streamplot.lines.axes
        for patch in axis.patches:
            if isinstance(patch, FancyArrowPatch):
                patch.set_visible(True)
                patch.set_linewidth(PDF_STREAMLINE_WIDTH)
