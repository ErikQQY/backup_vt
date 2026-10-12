#!/usr/bin/env python3
"""
Reproduce the three-panel analytical-flow figure used for the synthetic-data
illustration: circular Vatistas vortex, affine-deformed saddle, and rotated
elliptical vortex.

The analytical fields use the same LIC renderer and configuration as
``plot_vatistas_sampling_methods.py``.

Run
---
python synthetic_example.py

Outputs
-------
output/synthetic_example.png
output/synthetic_example.pdf
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# Use a publication-style font set for all mathematical annotations.
plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
})


# Match the effective LIC settings used by plot_vatistas_sampling_methods.py.
LIC_CONFIG = {
    "image_size": 384,
    "noise_size": 384,
    "noise_seed_offset": 811,
    "step_size": 0.002,
    "max_steps": 64,
    "color_by_velocity_magnitude": True,
}
LIC_SEED = 7068


def bilinear_sample(data, x, y):
    """Vectorized clamped bilinear sampling used by the project LIC renderer."""
    data = np.asarray(data)
    height, width = data.shape[:2]
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, width - 1.0)
    y = np.clip(np.asarray(y, dtype=np.float64), 0.0, height - 1.0)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx, wy = x - x0, y - y0
    if data.ndim == 3:
        wx, wy = wx[..., None], wy[..., None]
    top = data[y0, x0] * (1.0 - wx) + data[y0, x1] * wx
    bottom = data[y1, x0] * (1.0 - wx) + data[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def lic_output_coords(image_size):
    """Return the normalized coordinates used by the project LIC renderer."""
    ratio = np.arange(int(image_size), dtype=np.float64) / float(image_size)
    yy, xx = np.meshgrid(ratio, ratio, indexing="ij")
    return np.stack([2.0 * xx - 1.0, 2.0 * yy - 1.0], axis=-1)


def velocity_colormap(values):
    """Apply the same velocity-magnitude color map as the reference LIC."""
    anchors = np.asarray(
        [
            [0.267, 0.005, 0.329],
            [0.230, 0.322, 0.546],
            [0.128, 0.567, 0.551],
            [0.369, 0.789, 0.383],
            [0.993, 0.906, 0.144],
        ],
        dtype=np.float64,
    )
    scaled = np.clip(values, 0.0, 1.0) * (len(anchors) - 1)
    lower = np.floor(scaled).astype(np.int64)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    weight = (scaled - lower)[..., None]
    return anchors[lower] * (1.0 - weight) + anchors[upper] * weight


def make_lic_noise(lic_cfg, seed):
    """Create the same deterministic white-noise texture as the reference."""
    rng = np.random.default_rng(int(seed) + int(lic_cfg["noise_seed_offset"]))
    size = int(lic_cfg["noise_size"])
    return rng.random((size, size), dtype=np.float32)


def render_lic_image(field, lic_cfg, noise):
    """Render an LIC using the same algorithm as the dataset generator."""
    field = np.asarray(field, dtype=np.float64)
    size = int(lic_cfg["image_size"])
    positions0 = lic_output_coords(size)
    height, width = field.shape[:2]
    noise_height, noise_width = noise.shape
    accumulated = np.zeros((size, size), dtype=np.float64)
    counts = np.zeros((size, size), dtype=np.int32)

    for direction in (1.0, -1.0):
        positions = positions0.copy()
        active = np.ones((size, size), dtype=bool)
        for _ in range(int(lic_cfg["max_steps"])):
            field_x = (positions[..., 0] + 1.0) * 0.5 * (width - 1)
            field_y = (positions[..., 1] + 1.0) * 0.5 * (height - 1)
            active &= (
                np.isfinite(field_x)
                & np.isfinite(field_y)
                & (field_x >= 0.0)
                & (field_x < width)
                & (field_y >= 0.0)
                & (field_y < height)
            )
            if not np.any(active):
                break

            noise_x = field_x * (noise_width - 1) / max(1, width - 1)
            noise_y = field_y * (noise_height - 1) / max(1, height - 1)
            noise_values = bilinear_sample(noise, noise_x, noise_y)
            accumulated[active] += noise_values[active]
            counts[active] += 1

            velocity = bilinear_sample(field, field_x, field_y)
            positions[active] += (
                direction * float(lic_cfg["step_size"]) * velocity[active]
            )

    lic = accumulated / np.maximum(counts, 1)
    rgb = np.repeat(lic[..., None], 3, axis=-1)
    if lic_cfg["color_by_velocity_magnitude"]:
        speed = np.linalg.norm(field, axis=-1)
        sx = (positions0[..., 0] + 1.0) * 0.5 * (width - 1)
        sy = (positions0[..., 1] + 1.0) * 0.5 * (height - 1)
        speed_at_pixels = bilinear_sample(speed, sx, sy)
        speed_min, speed_max = float(speed.min()), float(speed.max())
        if speed_max > speed_min + 1.0e-12:
            normalized_speed = (speed_at_pixels - speed_min) / (
                speed_max - speed_min
            )
            criterion_color = velocity_colormap(normalized_speed)
            whiteish = np.clip((lic - 0.4) * (1.5 / 0.4), 0.0, 1.0)
            rgb = (
                rgb * (1.0 - whiteish[..., None])
                + criterion_color * whiteish[..., None]
            )

    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------
# Analytical field
# ---------------------------------------------------------------------

# Topological matrices.
# S_CENTER_CCW and S_CENTER_CW generate rotational centers;
# S_SADDLE generates a saddle.
S_SADDLE = np.array([[1.0, 0.0],
                     [0.0, -1.0]])

S_CENTER_CCW = np.array([[0.0, -1.0],
                         [1.0,  0.0]])

S_CENTER_CW = np.array([[0.0,  1.0],
                        [-1.0, 0.0]])


def vatistas_radial_factor(r, rc, n, eps=1.0e-12):
    """
    Scalar radial factor multiplying S @ xi.

    For a center, |S @ xi| = r, so the resulting tangential speed is

        v_theta(r) ∝ r / [1 + (r/rc)^(2n)]^(1/n),

    which reaches its maximum at r = rc.

    The overall multiplicative constant is irrelevant for the streamline
    geometry and for the normalized background shown in this figure.
    """
    q = np.maximum(r / rc, 0.0)
    return 1.0 / np.power(1.0 + q ** (2.0 * n), 1.0 / n)


def affine_matrix(theta, sx, sy):
    """
    D(theta,sx,sy) =
        [[sx cos(theta), -sy sin(theta)],
         [sx sin(theta),  sy cos(theta)]]
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[sx * c, -sy * s],
                     [sx * s,  sy * c]])


def analytical_component(X, Y, *, S, sx, sy, tx, ty, rc, n, theta):
    """
    Construct the affine-pushed analytical velocity

        x = D xi + T,
        v'(x) = D v(D^{-1}(x-T)).

    Returns
    -------
    U, V : transformed velocity components
    phi  : analytical implicit core function
           phi = rc - ||D^{-1}(x-T)||.
    """
    D = affine_matrix(theta, sx, sy)
    Dinv = np.linalg.inv(D)

    pts = np.stack([X - tx, Y - ty], axis=0)      # (2, Ny, Nx)
    Xi = np.einsum("ij,j...->i...", Dinv, pts)

    XI, ETA = Xi[0], Xi[1]
    r = np.hypot(XI, ETA)

    base = np.stack([XI, ETA], axis=0)
    Sv = np.einsum("ij,j...->i...", S, base)

    f = vatistas_radial_factor(r, rc, n)
    local_vel = Sv * f[None, ...]

    global_vel = np.einsum("ij,j...->i...", D, local_vel)

    U, V = global_vel[0], global_vel[1]
    phi = rc - r
    return U, V, phi


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def draw_panel(ax, X, Y, params, annotation, lic_cfg, noise):
    U, V, _ = analytical_component(X, Y, **params)

    # The shared LIC renderer integrates in normalized [-1, 1]^2 coordinates.
    # Convert the physical-domain velocity components to that coordinate system
    # so affine deformation and streamline geometry remain correct.
    x_half_range = 0.5 * (float(X.max()) - float(X.min()))
    y_half_range = 0.5 * (float(Y.max()) - float(Y.min()))
    normalized_field = np.stack(
        [U / x_half_range, V / y_half_range], axis=-1
    )
    lic_image = render_lic_image(normalized_field, lic_cfg, noise)

    ax.imshow(
        lic_image,
        extent=(X.min(), X.max(), Y.min(), Y.max()),
        origin="lower",
        interpolation="bilinear",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove axis spines to match the clean image-only panels.
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(
        0.5, -0.060, annotation,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=13,
    )


def main():
    script_dir = Path(__file__).resolve().parent

    # Reuse one deterministic noise texture for controlled visual comparison.
    lic_cfg = LIC_CONFIG.copy()
    noise = make_lic_noise(lic_cfg, LIC_SEED)

    # The reference LIC samples each analytical field on a 128 x 128 grid.
    x = np.linspace(-2.0, 2.0, 128)
    y = np.linspace(-2.0, 2.0, 128)
    X, Y = np.meshgrid(x, y)

    panels = [
        (
            dict(
                S=S_CENTER_CCW,
                sx=1.0, sy=1.0,
                tx=0.0, ty=0.0,
                rc=0.5, n=2.0,
                theta=0.0,
            ),
            (
                r"$(s_x,\,s_y)=(1,\,1),\ "
                r"(t_x,\,t_y)=(0,\,0), $"
                "\n"
                r"$r_c=0.5,\ n=2,\ \theta=0$"
            ),
        ),
        (
            dict(
                S=S_SADDLE,
                sx=2.0, sy=0.5,
                tx=0.5, ty=-0.5,
                rc=2.5, n=6.0,
                theta=0.0,
            ),
            (
                r"$(s_x,\,s_y)=(2,\,0.5),\ "
                r"(t_x,\,t_y)=(0.5,\,-0.5), $"
                "\n"
                r"$r_c=2.5,\ n=6,\ \theta=0$"
            ),
        ),
        (
            dict(
                S=S_CENTER_CCW,
                sx=1.0, sy=0.5,
                tx=0.0, ty=0.0,
                rc=0.5, n=2.0,
                theta=np.pi / 4.0,
            ),
            (
                r"$(s_x,\,s_y)=(1,\,0.5),\ "
                r"(t_x,\,t_y)=(0,\,0), $"
                "\n"
                r"$r_c=0.5,\ n=2,\ \theta=\pi/4$"
            ),
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.1))

    for ax, (params, annotation) in zip(axes, panels):
        draw_panel(ax, X, Y, params, annotation, lic_cfg, noise)

    # Tight horizontal spacing and enough bottom margin for LaTeX-like labels.
    fig.subplots_adjust(
        left=0.018, right=0.992,
        top=0.985, bottom=0.105,
        wspace=0.105,
    )

    output_dir = script_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / "synthetic_example.png"
    out_pdf = output_dir / "synthetic_example.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
