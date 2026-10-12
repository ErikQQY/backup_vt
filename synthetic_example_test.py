#!/usr/bin/env python3
"""Draw the four-panel construction of the synthetic analytical flow.

The figure explains how a canonical Vatistas vortex is affinely deformed,
contrasts it with a non-vortical saddle component, and shows how two vortical
components are superposed to form a labelled training field.  Every field
panel uses the same colored line-integral-convolution (LIC) style as the
dataset generator.

Run
---
python synthetic_example_test.py

Outputs
-------
output/synthetic_example_test.png
output/synthetic_example_test.pdf
"""

from pathlib import Path

import matplotlib

# Keep the script usable on headless training machines.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

from synthetic_example import (
    LIC_CONFIG,
    LIC_SEED,
    S_CENTER_CCW,
    S_SADDLE,
    analytical_component,
    make_lic_noise,
    render_lic_image,
)


plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "axes.titleweight": "bold",
        "axes.linewidth": 0.7,
    }
)


DOMAIN = (-2.15, 2.15)
CORE_YELLOW = "#ffe600"


CANONICAL_VORTEX = dict(
    S=S_CENTER_CCW,
    sx=1.0,
    sy=1.0,
    tx=0.0,
    ty=0.0,
    rc=0.72,
    n=2.0,
    theta=0.0,
)

DEFORMED_VORTEX = dict(
    S=S_CENTER_CCW,
    sx=1.35,
    sy=0.78,
    tx=-0.08,
    ty=-0.08,
    rc=0.82,
    n=2.0,
    theta=np.deg2rad(22.0),
)

SADDLE = dict(
    S=S_SADDLE,
    sx=1.0,
    sy=1.0,
    tx=0.0,
    ty=0.0,
    rc=4.0,
    n=4.0,
    theta=0.0,
)

# Parameters for panel (d): one complete core and a second core entering from
# the lower-right boundary.  These two same-sense vortices produce two genuine
# rotational critical points and one saddle between them in the summed field.
SUPERPOSED_PRIMARY_WEIGHT = 1.28
SUPERPOSED_SECONDARY_WEIGHT = 1.00
SUPERPOSED_SECONDARY_VORTEX = dict(
    S=S_CENTER_CCW,
    sx=1.15,
    sy=0.82,
    tx=1.85,
    ty=-1.65,
    rc=0.90,
    n=2.0,
    theta=np.deg2rad(-18.0),
)
CORE_ALIGNMENT_CUTOFF = 1.0
SUMMED_FIELD_SADDLE_POINT = (0.883840825397, -0.924942138277)


def style_field_axis(ax):
    """Apply the shared square, tick-free publication style."""
    ax.set_xlim(DOMAIN)
    ax.set_ylim(DOMAIN)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#60666a")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#7f8589")
        spine.set_linewidth(0.65)


def cross_null_gate(phi, rc, cutoff=CORE_ALIGNMENT_CUTOFF):
    """Return a C2 gate that is zero at a core center and one outside it.

    Each vortex is multiplied by the gate of the *other* vortex in panel (d).
    Consequently, cross-induced velocity vanishes exactly at both labelled
    centers, while the ordinary analytical superposition is recovered at and
    beyond each core boundary.
    """
    radius = np.maximum(float(rc) - np.asarray(phi), 0.0)
    scaled_radius = np.clip(radius / (float(cutoff) * float(rc)), 0.0, 1.0)
    return scaled_radius**3 * (
        scaled_radius * (scaled_radius * 6.0 - 15.0) + 10.0
    )


def draw_lic_background(ax, u, v, lic_cfg, noise):
    """Render one field with the colored LIC style used by the data generator."""
    half_range = 0.5 * (DOMAIN[1] - DOMAIN[0])
    normalized_field = np.stack([u / half_range, v / half_range], axis=-1)
    lic_image = render_lic_image(normalized_field, lic_cfg, noise)
    ax.imshow(
        lic_image,
        extent=(*DOMAIN, *DOMAIN),
        origin="lower",
        interpolation="bilinear",
        zorder=0,
    )


def draw_core(ax, x, y, phi):
    """Overlay the positive core and a yellow line with a black outline."""
    phi_max = float(np.nanmax(phi))
    if phi_max > 0.0:
        ax.contourf(
            x,
            y,
            phi,
            levels=[0.0, phi_max],
            colors=[CORE_YELLOW],
            alpha=0.28,
            antialiased=True,
            zorder=3,
        )

    # Drawing the same contour twice creates the narrow yellow-on-black edge
    # visible in the reference LIC image.
    ax.contour(
        x,
        y,
        phi,
        levels=[0.0],
        colors=["black"],
        linewidths=2.15,
        linestyles="-",
        zorder=4,
    )
    ax.contour(
        x,
        y,
        phi,
        levels=[0.0],
        colors=[CORE_YELLOW],
        linewidths=1.05,
        linestyles="-",
        zorder=5,
    )


def add_explanation_box(fig, ax, text, *, padding=0.48, gap_points=10.0):
    """Align each box's top edge a fixed distance below its field panel."""
    panel = ax.get_position()
    fontsize = 11.7
    figure_height_points = fig.get_figheight() * 72.0
    # Include the box padding so changing it does not change the panel gap.
    text_top = panel.y0 - (gap_points + padding * fontsize) / figure_height_points
    fig.text(
        panel.x0 + panel.width / 2.0,
        text_top,
        text,
        ha="center",
        va="top",
        fontsize=fontsize,
        linespacing=1.45,
        bbox=dict(
            boxstyle=f"square,pad={padding}",
            facecolor="white",
            edgecolor="#8b8b8b",
            linewidth=0.75,
        ),
    )


def main():
    script_dir = Path(__file__).resolve().parent

    # Reuse a single deterministic texture so differences between panels come
    # from the analytical fields rather than from different random noise.
    lic_cfg = LIC_CONFIG.copy()
    noise = make_lic_noise(lic_cfg, LIC_SEED)

    # A moderately fine grid gives a smooth analytical field for LIC sampling.
    x = np.linspace(*DOMAIN, 241)
    y = np.linspace(*DOMAIN, 241)
    X, Y = np.meshgrid(x, y)

    canonical_u, canonical_v, canonical_phi = analytical_component(
        X, Y, **CANONICAL_VORTEX
    )
    deformed_u, deformed_v, deformed_phi = analytical_component(
        X, Y, **DEFORMED_VORTEX
    )
    saddle_u, saddle_v, _ = analytical_component(X, Y, **SADDLE)

    # The last panel superposes two vortical components.  The second component
    # is intentionally centred close to the lower-right corner so only part of
    # its positive core lies inside the displayed domain.
    primary_vortex = dict(DEFORMED_VORTEX, tx=-0.72, ty=-0.10)
    primary_u, primary_v, primary_phi = analytical_component(
        X, Y, **primary_vortex
    )
    secondary_u, secondary_v, secondary_phi = analytical_component(
        X, Y, **SUPERPOSED_SECONDARY_VORTEX
    )

    # Smoothly suppress only the cross-induced velocity inside the other
    # component's core.  This keeps the LIC vortex eyes co-located with the
    # analytical phi >= 0 labels instead of shifting them under superposition.
    primary_gate = cross_null_gate(
        secondary_phi, SUPERPOSED_SECONDARY_VORTEX["rc"]
    )
    secondary_gate = cross_null_gate(primary_phi, primary_vortex["rc"])
    combined_u = (
        SUPERPOSED_PRIMARY_WEIGHT * primary_gate * primary_u
        + SUPERPOSED_SECONDARY_WEIGHT * secondary_gate * secondary_u
    )
    combined_v = (
        SUPERPOSED_PRIMARY_WEIGHT * primary_gate * primary_v
        + SUPERPOSED_SECONDARY_WEIGHT * secondary_gate * secondary_v
    )
    combined_core_phi = np.maximum(primary_phi, secondary_phi)

    fig = plt.figure(figsize=(15.6, 6.05), facecolor="white")

    # Explicit positions preserve the wide transformation gap between (a) and
    # (b), while keeping all four field windows exactly square.
    panel_positions = [
        (0.020, 0.342, 0.205, 0.529),
        (0.340, 0.342, 0.205, 0.529),
        (0.566, 0.342, 0.205, 0.529),
        (0.792, 0.342, 0.205, 0.529),
    ]
    axes = [fig.add_axes(position) for position in panel_positions]
    for ax in axes:
        style_field_axis(ax)

    titles = [
        "(a)  Canonical Vatistas vortex\n(circular center)",
        "(b)  Affinely deformed vortex\n(elliptical center)",
        "(c)  Non-vortical component\n(saddle)",
        "(d)  Superposed vortices\n(two components)",
    ]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=13.2, pad=15, linespacing=1.25)

    # (a) Canonical circular vortex.
    draw_lic_background(
        axes[0], canonical_u, canonical_v, lic_cfg, noise
    )
    draw_core(axes[0], x, y, canonical_phi)

    # (b) The same vortex after anisotropic scaling, rotation and translation.
    draw_lic_background(
        axes[1], deformed_u, deformed_v, lic_cfg, noise
    )
    draw_core(axes[1], x, y, deformed_phi)

    # (c) A non-vortical component: no positive closed core is associated with
    # this critical point.
    draw_lic_background(axes[2], saddle_u, saddle_v, lic_cfg, noise)
    axes[2].plot(
        SADDLE["tx"],
        SADDLE["ty"],
        marker="x",
        color="black",
        markersize=5.2,
        markeredgewidth=1.0,
        linestyle="None",
        zorder=5,
    )

    # (d) Two vortices; the label is the union of their positive cores.
    draw_lic_background(axes[3], combined_u, combined_v, lic_cfg, noise)
    draw_core(axes[3], x, y, combined_core_phi)
    axes[3].plot(
        SUMMED_FIELD_SADDLE_POINT[0],
        SUMMED_FIELD_SADDLE_POINT[1],
        marker="x",
        color="black",
        markersize=5.2,
        markeredgewidth=1.0,
        linestyle="None",
        zorder=5,
    )
    # Affine-map annotation occupying the deliberately enlarged first gap.
    fig.text(
        0.2825,
        0.655,
        r"$\mathbf{x}'=D\mathbf{\xi}+T$",
        ha="center",
        va="bottom",
        fontsize=12.8,
    )
    fig.add_artist(
        FancyArrowPatch(
            (0.238, 0.625),
            (0.327, 0.625),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.0,
            color="black",
        )
    )
    fig.text(
        0.238,
        0.586,
        r"$D=R(\theta)\,S(s_x,s_y)$" "\n" r"$T=(t_x,t_y)^{\mathsf{T}}$",
        ha="left",
        va="top",
        fontsize=10.6,
        linespacing=1.55,
    )

    add_explanation_box(
        fig,
        axes[0],
        r"$\phi(\mathbf{x})=r_c-r=0$  (circle)" "\n"
        r"positive inside  ($\phi\geq 0$)",
    )
    add_explanation_box(
        fig,
        axes[1],
        r"$\phi'(\mathbf{x})=r_c-r'=0$  (ellipse)" "\n"
        r"positive inside  ($\phi'\geq 0$)",
    )
    add_explanation_box(
        fig,
        axes[2],
        "No closed positive isocontour\n" r"$y(\mathbf{x})=0$",
    )
    add_explanation_box(
        fig,
        axes[3],
        r"$\mathbf{v}(\mathbf{x})=\sum_{p=1}^{m}"
        r"\mathbf{v}'_p(\mathbf{x}),"
        r" m\in\{1,2\}$",
        padding=1.05,
    )

    output_dir = script_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / "synthetic_example_test.png"
    out_pdf = output_dir / "synthetic_example_test.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
