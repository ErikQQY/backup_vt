#!/usr/bin/env python3
"""Render a cached cylinder-flow model inference as a scientific animation.

The NPZ must contain x [X], y [Y], times [T], time_indices [T], and u, v,
vorticity, probability, mask [T,Y,X], plus scalar radius and threshold.
NaN probabilities denote pixels outside valid inference coverage.

Example:
    .venv/bin/python render_vortex_animation.py --input inference.npz \
        --output-dir output/cylinder_vortices --fps 12 --dpi 240
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import BinaryIO

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pyflowvis-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import matplotlib.patheffects as path_effects
import numpy as np
from PIL import GifImagePlugin, Image, ImageDraw, ImageFont
from vorticity_reference import (
    add_reference_arguments, compute_reference, default_reference_options, options_from_args,
)


class StreamingGifWriter:
    """Write complete, independent GIF frames without retaining the sequence."""

    def __init__(self, path: Path, fps: float):
        self.file: BinaryIO = path.open("wb")
        self.fps = fps
        self.durations_ms: list[int] = []
        self.frame_count = 0

    def append(self, rgb: Image.Image) -> None:
        frame = rgb.quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                             dither=Image.Dither.NONE)
        if self.frame_count == 0:
            header, _ = GifImagePlugin.getheader(frame, info={"loop": 0})
            for block in header:
                self.file.write(block)
        # GIF stores centiseconds; cumulative rounding preserves the requested
        # mean frame rate instead of speeding a 12 fps sequence up to 12.5 fps.
        duration_ms = max(10, 10 * (round((self.frame_count + 1) * 100 / self.fps)
                                    - round(self.frame_count * 100 / self.fps)))
        self.durations_ms.append(duration_ms)
        for block in GifImagePlugin.getdata(
            frame, duration=duration_ms, disposal=2, include_color_table=True
        ):
            self.file.write(block)
        self.frame_count += 1

    def close(self) -> None:
        if not self.file.closed:
            self.file.write(b";")
            self.file.close()


def _resolve_ffmpeg(ffmpeg_path: str | Path | None) -> str:
    """Find and verify a video encoder before loading or rendering any frames."""
    candidate = str(ffmpeg_path) if ffmpeg_path is not None else os.environ.get("IMAGEIO_FFMPEG_EXE")
    if not candidate:
        candidate = shutil.which("ffmpeg")
    if not candidate:
        try:
            import imageio_ffmpeg
            candidate = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            candidate = None
    help_text = ("Pass --ffmpeg /path/to/ffmpeg, set IMAGEIO_FFMPEG_EXE, or install "
                 "imageio-ffmpeg in this Python environment. Use --formats gif for GIF only.")
    if not candidate:
        raise RuntimeError(f"MP4 was requested but no FFmpeg executable was found. {help_text}")
    expanded = os.path.expanduser(candidate)
    executable = shutil.which(expanded)
    if executable is None:
        raise RuntimeError(f"FFmpeg executable is unavailable or not executable: {candidate}. {help_text}")
    try:
        checked = subprocess.run([executable, "-version"], capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Cannot run FFmpeg at {executable}: {error}. {help_text}") from error
    if checked.returncode:
        error_text = checked.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg validation failed at {executable}: {error_text}. {help_text}")
    return str(Path(executable).resolve())


class StreamingMp4Writer:
    """Encode RGB frames and publish the MP4 only after FFmpeg succeeds."""

    def __init__(self, path: Path, ffmpeg: str, size: tuple[int, int], fps: float):
        self.path = path
        descriptor, temporary = tempfile.mkstemp(prefix=".vortex_detection-", suffix=".mp4", dir=path.parent)
        os.close(descriptor)
        self.temporary = Path(temporary)
        self.log = tempfile.TemporaryFile()
        self.process = None
        width, height = size
        try:
            self.process = subprocess.Popen(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
                 "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                 "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-vf",
                 "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 str(self.temporary)], stdin=subprocess.PIPE, stderr=self.log,
            )
        except BaseException:
            self.cleanup()
            raise

    def _error_details(self) -> str:
        self.log.seek(0)
        return self.log.read().decode("utf-8", errors="replace").strip()

    def append(self, rgb: Image.Image) -> None:
        try:
            self.process.stdin.write(rgb.tobytes())
        except BrokenPipeError as error:
            raise RuntimeError(f"FFmpeg stopped while encoding MP4: {self._error_details()}") from error

    def finish(self) -> None:
        try:
            self.process.stdin.close()
            return_code = self.process.wait(timeout=60)
        except (BrokenPipeError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"FFmpeg could not finish MP4: {self._error_details()}") from error
        if return_code:
            raise RuntimeError(f"FFmpeg failed with exit code {return_code}: {self._error_details()}")
        if self.temporary.stat().st_size == 0:
            raise RuntimeError("FFmpeg produced an empty MP4 file")
        self.temporary.replace(self.path)

    def cleanup(self) -> None:
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            if self.process.stdin is not None and not self.process.stdin.closed:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    pass
        self.log.close()
        self.temporary.unlink(missing_ok=True)


def _robust_abs_limit(values: np.ndarray) -> float:
    """Use a deterministic sequence-wide sample to bound percentile memory."""
    flat = values.reshape(-1)
    sample = flat[::max(1, flat.size // 2_000_000)]
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return 1.0
    limit = float(np.percentile(np.abs(finite), 99.0))
    return limit if limit > 0 else 1.0


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    keys = ("x", "y", "times", "time_indices", "u", "v", "vorticity",
            "probability", "mask", "radius", "threshold")
    with np.load(path, allow_pickle=False) as cached:
        missing = set(keys).difference(cached.files)
        if missing:
            raise ValueError(f"Inference cache is missing: {', '.join(sorted(missing))}")
        data = {key: cached[key] for key in keys}
    x, y, times = data["x"], data["y"], data["times"]
    if any(a.ndim != 1 for a in (x, y, times)) or min(x.size, y.size) < 2 or not times.size:
        raise ValueError("x/y/times must be nonempty vectors; x and y need at least two coordinates")
    if not all(np.isfinite(a).all() for a in (x, y, times)):
        raise ValueError("x/y/times must contain finite physical coordinates")
    if not all(np.all(np.diff(a) > 0) for a in (x, y)):
        raise ValueError("x and y coordinates must be strictly increasing")
    shape = (times.size, y.size, x.size)
    for key in ("u", "v", "vorticity", "probability", "mask"):
        if data[key].shape != shape:
            raise ValueError(f"{key} has shape {data[key].shape}; expected {shape}")
    if data["time_indices"].shape != times.shape:
        raise ValueError("time_indices must match times")
    if data["threshold"].size != 1 or not 0 <= float(data["threshold"].item()) <= 1:
        raise ValueError("threshold must be a scalar between 0 and 1")
    if data["radius"].size != 1 or float(data["radius"].item()) < 0:
        raise ValueError("radius must be a nonnegative scalar")
    return data


def _contact_sheet(frames: list[Image.Image], output: Path) -> None:
    width = 960
    height = round(frames[0].height * width / frames[0].width)
    gap, margin, title_height = 20, 28, 42
    columns = 2 if len(frames) > 1 else 1
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width + (columns - 1) * gap + 2 * margin,
                              rows * height + (rows - 1) * gap + 2 * margin + title_height),
                      "#e9edf2")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default(size=24)
    draw.text((margin, 15), "Cylinder flow | sequence overview", fill="#18283d", font=font)
    for index, frame in enumerate(frames):
        sheet.paste(frame.resize((width, height), Image.Resampling.LANCZOS),
                    (margin + (index % columns) * (width + gap),
                     margin + title_height + (index // columns) * (height + gap)))
    sheet.save(output)


def render_animation(input_path: str | Path, output_dir: str | Path,
                     fps: float = 12.0, dpi: int = 240, formats: str = "both",
                     ffmpeg_path: str | Path | None = None,
                     reference_settings: dict | None = None) -> dict:
    """Render every cached frame, returning output paths and rendering metadata."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    if dpi < 40:
        raise ValueError("dpi must be at least 40")
    if formats not in ("gif", "mp4", "both"):
        raise ValueError("formats must be gif, mp4, or both")
    make_gif = formats in ("gif", "both")
    make_mp4 = formats in ("mp4", "both")
    ffmpeg = _resolve_ffmpeg(ffmpeg_path) if make_mp4 else None
    data = _load_cache(Path(input_path))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / "vortex_detection.gif"
    mp4_path = output_dir / "vortex_detection.mp4"
    preview_path = output_dir / "preview.png"
    svg_path = output_dir / "preview.svg"
    contact_path = output_dir / "contact_sheet.png"

    x, y, times = data["x"], data["y"], data["times"]
    threshold = float(data["threshold"].item())
    radius = float(data["radius"].item())
    frame_count = len(times)
    resolved_reference_settings = (default_reference_options()
                                   if reference_settings is None else reference_settings)
    reference = compute_reference(data["vorticity"], x, y, radius,
                                  **resolved_reference_settings)
    reference.save(output_dir, x, y, times, data["time_indices"])
    print(f"IVD reference: fixed threshold={reference.threshold:.6g}; "
          f"regions/frame={reference.settings['region_count_range']}", flush=True)
    ivd_percentile = _robust_abs_limit(reference.ivd)
    # Round the robust limit outward for a legible, sequence-wide color scale.
    magnitude = 10.0 ** np.floor(np.log10(ivd_percentile))
    ivd_limit = float(np.ceil(ivd_percentile / magnitude * 2) * magnitude / 2)
    selected = set(np.linspace(0, frame_count - 1, min(frame_count, 6), dtype=int).tolist())
    preview_index = frame_count // 2
    samples: list[Image.Image] = []
    # Set the physical size and typography for a journal's two-column figure.
    plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 9,
                         "mathtext.fontset": "stix", "svg.fonttype": "none",
                         "axes.labelcolor": "#222222", "text.color": "#222222",
                         "xtick.color": "#333333", "ytick.color": "#333333"})
    figure = plt.figure(figsize=(7.2, 2.65), dpi=dpi, facecolor="white")
    panel_bottoms = (0.59, 0.16)
    axes = [figure.add_axes([0.075, bottom, 0.825, 0.285]) for bottom in panel_bottoms]
    # Coordinates are sample centers; imshow expects outer pixel edges.
    # Keep physical plot limits at the original coordinate range below.
    extent = (float(x[0] - (x[1] - x[0]) / 2), float(x[-1] + (x[-1] - x[-2]) / 2),
              float(y[0] - (y[1] - y[0]) / 2), float(y[-1] + (y[-1] - y[-2]) / 2))
    neutral = plt.get_cmap("RdBu_r")(0.5)
    probability_cmap = LinearSegmentedColormap.from_list(
        "neutral_warm_probability", [neutral, "#fbd9b2", "#f7a66a", "#d85538", "#8e1733"], N=256)
    probability_cmap.set_bad("#e8ecf1")
    images = [
        axes[0].imshow(np.ma.masked_invalid(reference.ivd[0]), origin="lower", extent=extent,
                       cmap=probability_cmap, vmin=0, vmax=ivd_limit,
                       interpolation="nearest", aspect="equal"),
        axes[1].imshow(np.ma.masked_invalid(data["probability"][0]), origin="lower",
                       extent=extent, cmap=probability_cmap, vmin=0, vmax=1,
                       interpolation="nearest", aspect="equal"),
    ]
    panel_titles = ("Ground truth vortices", "Inference vortices")
    colorbars = []
    for index, (axis, title) in enumerate(zip(axes, panel_titles)):
        axis.set_xlim(x[0], x[-1])
        axis.set_ylim(y[0], y[-1])
        axis.set_facecolor(neutral)
        axis.set_title(title, loc="left", fontsize=10, fontweight="normal", pad=5)
        axis.set_ylabel(r"$y$", labelpad=3, fontsize=10)
        if index == 1:
            axis.set_xlabel(r"$x$", labelpad=2, fontsize=10)
        axis.set_yticks(np.linspace(y[0], y[-1], 3))
        axis.tick_params(labelsize=8, length=2.5, width=0.55, direction="in",
                         labelbottom=index == 1)
        for spine in axis.spines.values():
            spine.set_color("#555555")
            spine.set_linewidth(0.55)
        if radius > 0:
            axis.add_patch(Circle((0, 0), radius, facecolor="white",
                                   edgecolor="#222222", linewidth=0.65, zorder=7))
        colorbar_axis = figure.add_axes([0.918, panel_bottoms[index], 0.012, 0.285])
        colorbar = figure.colorbar(images[index], cax=colorbar_axis,
                                  extend="max" if index == 0 else "neither",
                                  extendfrac=0.04)
        colorbars.append(colorbar)
        colorbar.ax.tick_params(labelsize=8, length=2, width=0.45, pad=2)
        colorbar.set_label("IVD" if index == 0 else r"$p$", fontsize=10, labelpad=4)
        colorbar.outline.set_linewidth(0.45)
        colorbar.outline.set_edgecolor("#555555")
        if index == 0:
            colorbar.set_ticks([0, ivd_limit / 2, ivd_limit])
            colorbar.set_ticklabels(["0", f"{ivd_limit / 2:g}", f"{ivd_limit:g}"])
        else:
            colorbar.set_ticks([0, .5, 1])
    # Equal aspect can shrink the axes vertically; align the colorbars to the real panels.
    figure.canvas.draw()
    for axis, colorbar in zip(axes, colorbars):
        box = axis.get_position()
        # The upper extension triangle indicates clipped high IVD values.
        colorbar.ax.set_position([0.918, box.y0, 0.012, box.height])

    y_step = max(1, len(y) // 5)
    x_step = max(1, len(x) // 44)
    yy, xx = np.meshgrid(y[::y_step], x[::x_step], indexing="ij")
    arrows_inside = xx * xx + yy * yy < radius * radius
    quiver_u = np.ma.array(data["u"][0, ::y_step, ::x_step], mask=arrows_inside)
    quiver_v = np.ma.array(data["v"][0, ::y_step, ::x_step], mask=arrows_inside)
    speed_scale = max(_robust_abs_limit(data["u"]), _robust_abs_limit(data["v"]))
    quiver = axes[0].quiver(xx, yy, quiver_u, quiver_v, color="#444444", alpha=.28,
                          angles="xy", scale_units="xy", scale=speed_scale / ((x[-1] - x[0]) / 95),
                          width=.00065, headwidth=3.3, headlength=4, pivot="mid", zorder=4)
    contours = []
    gif = None
    video = None
    try:
        if make_gif:
            gif = StreamingGifWriter(gif_path, fps)
        if make_mp4:
            video = StreamingMp4Writer(mp4_path, ffmpeg, figure.canvas.get_width_height(), fps)
        for index in range(frame_count):
            images[0].set_data(np.ma.masked_invalid(reference.ivd[index]))
            images[1].set_data(np.ma.masked_invalid(data["probability"][index]))
            quiver.set_UVC(
                np.ma.array(data["u"][index, ::y_step, ::x_step], mask=arrows_inside),
                np.ma.array(data["v"][index, ::y_step, ::x_step], mask=arrows_inside),
            )
            for contour in contours:
                contour.remove()
            contours.clear()
            # Recompute regions from IVD using the selected cutoff, not a copied model mask.
            for sign in (1, -1):
                reference_region = np.ma.masked_where(
                    ~reference.valid[index],
                    (reference.signed_mask[index] == sign).astype(np.float32))
                if np.any(reference_region == 1) and np.any(reference_region == 0):
                    contour = axes[0].contour(x, y, reference_region, levels=[.5],
                                              colors=["#222222"], linewidths=0.6, zorder=6)
                    contour.set_path_effects([
                        path_effects.Stroke(linewidth=1.1, foreground="white", alpha=.85),
                        path_effects.Normal(),
                    ])
                    contours.append(contour)
            region = data["mask"][index].astype(np.float32)
            valid = np.isfinite(data["probability"][index])
            region = np.ma.masked_where(~valid, region)
            if np.any(region == 1) and np.any(region == 0):
                # Only the inference panel receives a model-derived boundary.
                contour = axes[1].contour(x, y, region, levels=[.5], colors=["#222222"],
                                          linewidths=0.6, zorder=6)
                contour.set_path_effects([
                    path_effects.Stroke(linewidth=1.1, foreground="white", alpha=.85),
                    path_effects.Normal(),
                ])
                contours.append(contour)
            figure.canvas.draw()
            rgb = Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
            if gif is not None:
                gif.append(rgb)
            if index == preview_index:
                figure.savefig(preview_path, dpi=300, facecolor="white")
                figure.savefig(svg_path, dpi=300, facecolor="white")
            if index in selected:
                samples.append(rgb.copy())
            if video is not None:
                video.append(rgb)
            if index == 0 or (index + 1) % 25 == 0 or index + 1 == frame_count:
                print(f"Rendered {index + 1}/{frame_count} frames", flush=True)
        if video is not None:
            video.finish()
    finally:
        if gif is not None:
            gif.close()
        if video is not None:
            video.cleanup()
        plt.close(figure)
    _contact_sheet(samples, contact_path)
    metadata = {
        "input": str(Path(input_path).resolve()), "frames": frame_count,
        "first_time": float(times[0]), "last_time": float(times[-1]),
        "requested_fps": fps, "formats": formats,
        "gif_frame_durations_ms": gif.durations_ms if gif is not None else None,
        "gif_duration_seconds": sum(gif.durations_ms) / 1000 if gif is not None else None,
        "mp4_duration_seconds": frame_count / fps if make_mp4 else None,
        "ivd_color_limits": [0.0, ivd_limit], "probability_threshold": threshold,
        "ivd_99th_percentile": ivd_percentile,
        "style": "journal", "figure_inches": [7.2, 2.65], "font_family": "STIXGeneral",
        "preview_dpi": 300, "inference_contours_on": "lower panel only",
        "upper_panel_content": "IVD intensity and velocity with vorticity-derived IVD reference boundaries.",
        "upper_colorbar_label": "IVD",
        "reference": reference.settings,
        "reference_data": str((output_dir / "vorticity_reference.npz").resolve()),
        "reference_regions": str((output_dir / "vorticity_reference_regions.csv").resolve()),
        "preview_frame": preview_index, "contact_sheet_frames": sorted(selected),
        "gif": str(gif_path.resolve()) if make_gif else None, "preview": str(preview_path.resolve()),
        "svg": str(svg_path.resolve()),
        "contact_sheet": str(contact_path.resolve()),
        "mp4": str(mp4_path.resolve()) if make_mp4 else None,
        "ffmpeg": ffmpeg,
    }
    (output_dir / "render_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({key: value for key, value in metadata.items()
                      if key != "gif_frame_durations_ms"}, indent=2), flush=True)
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Inference NPZ cache")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=12)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--formats", choices=("gif", "mp4", "both"), default="both")
    parser.add_argument("--ffmpeg", type=Path, help="FFmpeg executable (overrides environment and discovery)")
    add_reference_arguments(parser)
    args = parser.parse_args(argv)
    render_animation(args.input, args.output_dir, args.fps, args.dpi, args.formats, args.ffmpeg,
                     reference_settings=options_from_args(args))


if __name__ == "__main__":
    main()
