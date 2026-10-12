#!/usr/bin/env python3
"""
Plot a 2D space-filling curve for serialized point/token ordering.

Default: Hilbert curve
Optional: Morton / Z-order curve

Examples
--------
python plot_space_filling_curve.py
python plot_space_filling_curve.py --curve hilbert --order 4
python plot_space_filling_curve.py --curve morton --order 4
python plot_space_filling_curve.py --curve hilbert --order 5 --show-arrows
python plot_space_filling_curve.py --curve hilbert --order 4 --output outputs/serialization_curve.png

The 16:9 figure contains only graphics, without labels, axes, titles, or notes.
By default, PNG and SVG files are saved in the script's outputs directory.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def hilbert_index_to_xy(index: int, order: int):
    """Convert Hilbert distance/index to integer (x, y) coordinates."""
    n = 1 << order
    x = 0
    y = 0
    t = index
    s = 1

    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)

        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x

        x += s * rx
        y += s * ry
        t //= 4
        s *= 2

    return x, y


def hilbert_curve(order: int):
    """Return ordered coordinates of a 2D Hilbert curve."""
    num_points = 4 ** order
    pts = np.array(
        [hilbert_index_to_xy(i, order) for i in range(num_points)],
        dtype=float,
    )
    return pts


def compact1by1(x: int):
    """Inverse Morton bit interleaving helper."""
    x &= 0x55555555
    x = (x ^ (x >> 1)) & 0x33333333
    x = (x ^ (x >> 2)) & 0x0F0F0F0F
    x = (x ^ (x >> 4)) & 0x00FF00FF
    x = (x ^ (x >> 8)) & 0x0000FFFF
    return x


def morton_index_to_xy(index: int):
    """Convert Morton/Z-order index to integer (x, y) coordinates."""
    x = compact1by1(index)
    y = compact1by1(index >> 1)
    return x, y


def morton_curve(order: int):
    """Return ordered coordinates of a 2D Morton / Z-order curve."""
    num_points = 4 ** order
    pts = np.array(
        [morton_index_to_xy(i) for i in range(num_points)],
        dtype=float,
    )
    return pts


def normalize_points(points):
    """Map integer lattice coordinates to [0, 1] x [0, 1]."""
    max_val = np.max(points)
    if max_val == 0:
        return points.copy()
    return points / max_val


def plot_curve(
    points,
    order,
    output,
    show_grid=True,
    show_arrows=False,
):
    points = normalize_points(points)

    fig, ax = plt.subplots(figsize=(12.8, 7.2))

    # Optional background grid: visually indicates the spatial cells that
    # become a 1D serialized sequence.
    if show_grid:
        n = 1 << order
        grid = np.linspace(0.0, 1.0, n)
        for g in grid:
            ax.axhline(g, linewidth=0.35, alpha=0.22, zorder=0)
            ax.axvline(g, linewidth=0.35, alpha=0.22, zorder=0)

    # Space-filling curve
    ax.plot(
        points[:, 0],
        points[:, 1],
        linewidth=1.65,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=2,
    )

    # Serialized tokens / points
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=16,
        zorder=3,
    )

    # Start/end markers
    ax.scatter(
        points[0, 0],
        points[0, 1],
        s=72,
        marker="o",
        zorder=4,
    )
    ax.scatter(
        points[-1, 0],
        points[-1, 1],
        s=72,
        marker="s",
        zorder=4,
    )

    # Optional arrows showing traversal direction.
    if show_arrows:
        stride = max(len(points) // 24, 1)
        for i in range(0, len(points) - 1, stride):
            p0 = points[i]
            p1 = points[i + 1]
            ax.annotate(
                "",
                xy=p1,
                xytext=p0,
                arrowprops=dict(
                    arrowstyle="->",
                    lw=0.8,
                    shrinkA=1.5,
                    shrinkB=1.5,
                ),
                zorder=5,
            )

    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_aspect("auto")

    # Hide all axis decorations, leaving only the curve graphics.
    ax.set_axis_off()

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if output:
        output = Path(output)
        if not output.suffix:
            output = output.with_suffix(".png")
        output.parent.mkdir(parents=True, exist_ok=True)
        output_paths = [output]
        if output.suffix.lower() != ".svg":
            output_paths.append(output.with_suffix(".svg"))
        for output_path in output_paths:
            # Preserve the full 16:9 canvas when exporting.
            fig.savefig(output_path, dpi=600)
            print(f"Saved figure to: {output_path}")
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot Hilbert or Morton space-filling curves."
    )
    parser.add_argument(
        "--curve",
        choices=["hilbert", "morton"],
        default="hilbert",
        help="Type of space-filling serialization curve.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=4,
        help="Curve order. order=4 gives 16x16 = 256 points.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output figure path; an SVG copy is also saved. "
            "Default: outputs/<curve>_space_filling_curve_order<order>.png "
            "beside this script."
        ),
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Disable background spatial grid.",
    )
    parser.add_argument(
        "--show-arrows",
        action="store_true",
        help="Show sparse arrows indicating traversal direction.",
    )
    args = parser.parse_args()

    if args.order < 1:
        raise ValueError("--order must be >= 1")

    if args.curve == "hilbert":
        points = hilbert_curve(args.order)
    else:
        points = morton_curve(args.order)

    output = args.output or (
        Path(__file__).resolve().parent
        / "outputs"
        / f"{args.curve}_space_filling_curve_order{args.order}.png"
    )

    plot_curve(
        points=points,
        order=args.order,
        output=output,
        show_grid=not args.no_grid,
        show_arrows=args.show_arrows,
    )


if __name__ == "__main__":
    main()
