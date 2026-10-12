#!/usr/bin/env python3
"""Generate a vorticity-adapted triangular mesh for ``cylinder2d.nc``.

The script computes the scalar vorticity

    omega = dv/dx - du/dy

at one time step, turns its (smoothed) magnitude into a target edge-length
field, samples that field with a variable-radius Poisson sampler, and finally
builds an unstructured Delaunay triangulation.  The cylinder is represented as
an explicit polygonal hole, rather than being meshed over.

Examples
--------
Generate the default mesh from the final snapshot::

    MPLCONFIGDIR=/tmp/pyflowvis-mpl .venv/bin/python generate_mesh.py

Use a different snapshot or request stronger refinement::

    .venv/bin/python generate_mesh.py --time-index 1000 \
        --min-size 0.008 --max-size 0.12

The default outputs are an NPZ archive for Python, an ASCII VTK file for
ParaView, and a PNG preview under ``outputs/``.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Optional, Sequence, Tuple

import matplotlib

# This program is commonly run on a headless machine.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay, QhullError


@dataclass(frozen=True)
class FlowSnapshot:
    """One velocity snapshot on a Cartesian grid."""

    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    time: float
    time_index: int
    cylinder_radius: Optional[float]


@dataclass(frozen=True)
class Mesh:
    """The generated mesh and point-wise adaptation fields."""

    points: np.ndarray
    triangles: np.ndarray
    boundary_edges: np.ndarray
    boundary_markers: np.ndarray
    vorticity: np.ndarray
    target_size: np.ndarray


def load_snapshot(path: Path, time_index: int) -> FlowSnapshot:
    """Load one snapshot without reading the full time-dependent dataset."""

    with nc.Dataset(path, "r") as dataset:
        required = {"u", "v", "xdim", "ydim", "tdim"}
        missing = sorted(required.difference(dataset.variables))
        if missing:
            raise ValueError(f"{path} is missing variables: {', '.join(missing)}")

        x = np.asarray(dataset.variables["xdim"][:], dtype=np.float64)
        y = np.asarray(dataset.variables["ydim"][:], dtype=np.float64)
        times = np.asarray(dataset.variables["tdim"][:], dtype=np.float64)
        resolved_index = time_index if time_index >= 0 else len(times) + time_index
        if not 0 <= resolved_index < len(times):
            raise ValueError(
                f"time index {time_index} is outside [-{len(times)}, {len(times) - 1}]"
            )

        u = np.ma.filled(
            dataset.variables["u"][resolved_index, :, :], np.nan
        ).astype(np.float64)
        v = np.ma.filled(
            dataset.variables["v"][resolved_index, :, :], np.nan
        ).astype(np.float64)
        radius = None
        if "radius" in dataset.variables:
            radius = float(np.asarray(dataset.variables["radius"][:]).ravel()[0])

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("xdim and ydim must be one-dimensional coordinate arrays")
    if u.shape != (len(y), len(x)) or v.shape != u.shape:
        raise ValueError(
            f"expected u and v to have shape {(len(y), len(x))}, got {u.shape}"
        )
    if not np.all(np.diff(x) > 0.0) or not np.all(np.diff(y) > 0.0):
        raise ValueError("xdim and ydim must be strictly increasing")
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
        raise ValueError("the selected velocity snapshot contains missing values")

    return FlowSnapshot(
        x=x,
        y=y,
        u=u,
        v=v,
        time=float(times[resolved_index]),
        time_index=resolved_index,
        cylinder_radius=radius,
    )


def build_monitor(
    snapshot: FlowSnapshot,
    min_size: float,
    max_size: float,
    smooth_sigma: float,
    lower_percentile: float,
    upper_percentile: float,
    exponent: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return vorticity and a robust target edge-length field."""

    dvdx = np.gradient(snapshot.v, snapshot.x, axis=1, edge_order=2)
    dudy = np.gradient(snapshot.u, snapshot.y, axis=0, edge_order=2)
    vorticity = dvdx - dudy
    magnitude = np.abs(vorticity)
    if smooth_sigma > 0.0:
        magnitude = gaussian_filter(magnitude, sigma=smooth_sigma, mode="nearest")

    low, high = np.percentile(magnitude, [lower_percentile, upper_percentile])
    if high <= low + np.finfo(np.float64).eps:
        normalized = np.zeros_like(magnitude)
    else:
        normalized = np.clip((magnitude - low) / (high - low), 0.0, 1.0)

    # exponent > 1 reserves the strongest refinement for high-vorticity zones.
    refinement = normalized**exponent
    target_size = max_size - (max_size - min_size) * refinement
    return vorticity, target_size


def _adaptive_segment(
    start: Sequence[float],
    end: Sequence[float],
    size_at,
    min_size: float,
) -> np.ndarray:
    """Sample a straight boundary using the local target edge length."""

    start_array = np.asarray(start, dtype=np.float64)
    end_array = np.asarray(end, dtype=np.float64)
    direction = end_array - start_array
    length = float(np.linalg.norm(direction))
    if length == 0.0:
        return start_array[None, :]
    direction /= length

    points = [start_array]
    distance = 0.0
    while distance < length:
        local_size = float(size_at(points[-1][None, :])[0])
        distance = min(length, distance + max(0.75 * local_size, 0.5 * min_size))
        points.append(start_array + distance * direction)
    return np.asarray(points)


def make_boundary_points(
    bounds: Tuple[float, float, float, float],
    size_at,
    min_size: float,
    cylinder_center: Tuple[float, float],
    cylinder_radius: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Create outer-boundary and ordered cylinder-boundary vertices."""

    xmin, xmax, ymin, ymax = bounds
    sides = [
        ((xmin, ymin), (xmax, ymin)),
        ((xmax, ymin), (xmax, ymax)),
        ((xmax, ymax), (xmin, ymax)),
        ((xmin, ymax), (xmin, ymin)),
    ]
    # Omit each segment's last vertex to avoid duplicating rectangle corners.
    outer_parts = [
        _adaptive_segment(a, b, size_at, min_size)[:-1] for a, b in sides
    ]
    outer = np.vstack(outer_parts)

    if cylinder_radius is None or cylinder_radius <= 0.0:
        return outer, np.empty((0, 2), dtype=np.float64)

    circumference = 2.0 * math.pi * cylinder_radius
    # A finer inner boundary accurately represents the curved no-slip obstacle.
    count = max(24, int(math.ceil(circumference / (0.55 * min_size))))
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    cx, cy = cylinder_center
    circle = np.column_stack(
        (cx + cylinder_radius * np.cos(angles), cy + cylinder_radius * np.sin(angles))
    )
    return outer, circle


def make_candidates(
    bounds: Tuple[float, float, float, float],
    min_size: float,
    cylinder_center: Tuple[float, float],
    cylinder_radius: Optional[float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a jittered triangular lattice fine enough for the minimum size."""

    xmin, xmax, ymin, ymax = bounds
    step = 0.70 * min_size
    row_step = math.sqrt(3.0) * 0.5 * step
    rows = []
    y_value = ymin + row_step
    row = 0
    while y_value < ymax - row_step:
        offset = 0.5 * step if row % 2 else 0.0
        xs = np.arange(xmin + step + offset, xmax - step, step)
        if len(xs):
            ys = np.full_like(xs, y_value)
            rows.append(np.column_stack((xs, ys)))
        row += 1
        y_value += row_step

    if not rows:
        raise ValueError("min-size is too large for the dataset domain")
    candidates = np.vstack(rows)
    candidates += rng.uniform(-0.12 * step, 0.12 * step, candidates.shape)

    inside = (
        (candidates[:, 0] > xmin)
        & (candidates[:, 0] < xmax)
        & (candidates[:, 1] > ymin)
        & (candidates[:, 1] < ymax)
    )
    if cylinder_radius is not None and cylinder_radius > 0.0:
        center = np.asarray(cylinder_center)
        radius_squared = np.sum((candidates - center) ** 2, axis=1)
        inside &= radius_squared > (1.01 * cylinder_radius) ** 2
    return candidates[inside]


def variable_poisson_sample(
    boundary_points: np.ndarray,
    boundary_sizes: np.ndarray,
    candidates: np.ndarray,
    candidate_sizes: np.ndarray,
    min_size: float,
    max_size: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Greedily enforce a spatially varying minimum point separation."""

    cell_size = min_size / math.sqrt(2.0)
    buckets: DefaultDict[Tuple[int, int], list] = defaultdict(list)
    accepted = [point.copy() for point in boundary_points]
    accepted_sizes = [float(value) for value in boundary_sizes]

    def cell_of(point: np.ndarray) -> Tuple[int, int]:
        return (math.floor(point[0] / cell_size), math.floor(point[1] / cell_size))

    for index, point in enumerate(accepted):
        buckets[cell_of(point)].append(index)

    # Refined locations are considered first.  A small random term avoids
    # directional artefacts between candidates with almost equal sizes.
    scale = max(max_size - min_size, np.finfo(np.float64).eps)
    priority = candidate_sizes + rng.uniform(0.0, 0.04 * scale, len(candidates))
    order = np.argsort(priority, kind="stable")

    for candidate_index in order:
        point = candidates[candidate_index]
        local_size = float(candidate_sizes[candidate_index])
        cell_x, cell_y = cell_of(point)
        search_distance = 0.5 * (local_size + max_size)
        cell_radius = int(math.ceil(search_distance / cell_size))
        valid = True
        for ix in range(cell_x - cell_radius, cell_x + cell_radius + 1):
            if not valid:
                break
            for iy in range(cell_y - cell_radius, cell_y + cell_radius + 1):
                for neighbor_index in buckets.get((ix, iy), ()):
                    neighbor = accepted[neighbor_index]
                    required = 0.82 * 0.5 * (
                        local_size + accepted_sizes[neighbor_index]
                    )
                    if np.dot(point - neighbor, point - neighbor) < required**2:
                        valid = False
                        break
                if not valid:
                    break

        if valid:
            new_index = len(accepted)
            accepted.append(point.copy())
            accepted_sizes.append(local_size)
            buckets[(cell_x, cell_y)].append(new_index)

    return np.asarray(accepted, dtype=np.float64)


def _segment_distance_to_point(
    a: np.ndarray, b: np.ndarray, point: np.ndarray
) -> np.ndarray:
    """Vectorized distance from a point to line segments a--b."""

    direction = b - a
    denominator = np.einsum("ij,ij->i", direction, direction)
    parameter = np.einsum("ij,ij->i", point - a, direction) / np.maximum(
        denominator, np.finfo(np.float64).eps
    )
    parameter = np.clip(parameter, 0.0, 1.0)
    nearest = a + parameter[:, None] * direction
    return np.linalg.norm(nearest - point, axis=1)


def _cross_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Two-dimensional cross product, compatible with NumPy >= 2.0."""

    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def triangulate(
    points: np.ndarray,
    circle_start: int,
    circle_count: int,
    cylinder_center: Tuple[float, float],
    cylinder_radius: Optional[float],
) -> np.ndarray:
    """Delaunay-triangulate points and remove elements crossing the cylinder."""

    try:
        triangles = Delaunay(points).simplices.copy()
    except QhullError:
        # QJ is a robust fallback for nearly collinear boundary configurations.
        triangles = Delaunay(points, qhull_options="QJ").simplices.copy()

    vertices = points[triangles]
    signed_twice_area = _cross_2d(
        vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
    )
    keep = np.abs(signed_twice_area) > 1.0e-14

    if cylinder_radius is not None and cylinder_radius > 0.0 and circle_count:
        center = np.asarray(cylinder_center, dtype=np.float64)
        centroids = vertices.mean(axis=1)
        polygon_radius = cylinder_radius * math.cos(math.pi / circle_count)
        keep &= np.linalg.norm(centroids - center, axis=1) >= polygon_radius

        circle_end = circle_start + circle_count
        circle_local = np.full(len(points), -1, dtype=np.int64)
        circle_local[circle_start:circle_end] = np.arange(circle_count)
        tolerance = max(1.0e-12, 1.0e-6 * cylinder_radius)
        for edge_a, edge_b in ((0, 1), (1, 2), (2, 0)):
            ids_a = triangles[:, edge_a]
            ids_b = triangles[:, edge_b]
            local_a = circle_local[ids_a]
            local_b = circle_local[ids_b]
            difference = np.abs(local_a - local_b)
            adjacent_circle_edge = (
                (local_a >= 0)
                & (local_b >= 0)
                & ((difference == 1) | (difference == circle_count - 1))
            )
            edge_distance = _segment_distance_to_point(
                points[ids_a], points[ids_b], center
            )
            keep &= adjacent_circle_edge | (
                edge_distance >= polygon_radius - tolerance
            )

    triangles = triangles[keep]
    # Consistent counter-clockwise orientation is convenient downstream.
    selected_points = points[triangles]
    clockwise = (
        _cross_2d(
            selected_points[:, 1] - selected_points[:, 0],
            selected_points[:, 2] - selected_points[:, 0],
        )
        < 0.0
    )
    triangles[clockwise, 1], triangles[clockwise, 2] = (
        triangles[clockwise, 2].copy(),
        triangles[clockwise, 1].copy(),
    )
    return triangles.astype(np.int32)


def extract_boundary_edges(
    triangles: np.ndarray, circle_start: int, circle_count: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return exterior edges with marker 1 (far field) or 2 (cylinder)."""

    edges = np.sort(
        np.vstack(
            (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])
        ),
        axis=1,
    )
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1].astype(np.int32)
    boundary_markers = np.ones(len(boundary_edges), dtype=np.int8)
    if circle_count:
        circle_end = circle_start + circle_count
        on_circle = (
            (boundary_edges[:, 0] >= circle_start)
            & (boundary_edges[:, 0] < circle_end)
            & (boundary_edges[:, 1] >= circle_start)
            & (boundary_edges[:, 1] < circle_end)
        )
        boundary_markers[on_circle] = 2
    return boundary_edges, boundary_markers


def generate_mesh(
    snapshot: FlowSnapshot,
    min_size: float,
    max_size: float,
    smooth_sigma: float,
    lower_percentile: float,
    upper_percentile: float,
    exponent: float,
    seed: int,
    cylinder_center: Tuple[float, float],
    cylinder_radius: Optional[float],
) -> Tuple[Mesh, np.ndarray, np.ndarray]:
    """Run the complete vorticity-based mesh adaptation pipeline."""

    bounds = (
        float(snapshot.x[0]),
        float(snapshot.x[-1]),
        float(snapshot.y[0]),
        float(snapshot.y[-1]),
    )
    vorticity_grid, size_grid = build_monitor(
        snapshot,
        min_size,
        max_size,
        smooth_sigma,
        lower_percentile,
        upper_percentile,
        exponent,
    )
    size_interpolator = RegularGridInterpolator(
        (snapshot.y, snapshot.x), size_grid, bounds_error=False, fill_value=max_size
    )

    def size_at(xy: np.ndarray) -> np.ndarray:
        return size_interpolator(np.column_stack((xy[:, 1], xy[:, 0])))

    outer, circle = make_boundary_points(
        bounds, size_at, min_size, cylinder_center, cylinder_radius
    )
    boundary = np.vstack((outer, circle))
    boundary_sizes = size_at(boundary)
    if len(circle):
        boundary_sizes[len(outer) :] = min_size
    rng = np.random.default_rng(seed)
    candidates = make_candidates(
        bounds, min_size, cylinder_center, cylinder_radius, rng
    )
    candidate_sizes = size_at(candidates)
    points = variable_poisson_sample(
        boundary,
        boundary_sizes,
        candidates,
        candidate_sizes,
        min_size,
        max_size,
        rng,
    )

    triangles = triangulate(
        points,
        circle_start=len(outer),
        circle_count=len(circle),
        cylinder_center=cylinder_center,
        cylinder_radius=cylinder_radius,
    )
    boundary_edges, boundary_markers = extract_boundary_edges(
        triangles, circle_start=len(outer), circle_count=len(circle)
    )
    vorticity_interpolator = RegularGridInterpolator(
        (snapshot.y, snapshot.x),
        vorticity_grid,
        bounds_error=False,
        fill_value=0.0,
    )
    query = np.column_stack((points[:, 1], points[:, 0]))
    mesh = Mesh(
        points=points,
        triangles=triangles,
        boundary_edges=boundary_edges,
        boundary_markers=boundary_markers,
        vorticity=np.asarray(vorticity_interpolator(query)),
        target_size=np.asarray(size_interpolator(query)),
    )
    return mesh, vorticity_grid, size_grid


def save_npz(
    path: Path,
    mesh: Mesh,
    snapshot: FlowSnapshot,
    cylinder_center: Tuple[float, float],
    cylinder_radius: Optional[float],
) -> None:
    """Save mesh arrays and metadata in a compact NumPy archive."""

    path.parent.mkdir(parents=True, exist_ok=True)
    radius_value = np.nan if cylinder_radius is None else cylinder_radius
    np.savez_compressed(
        path,
        points=mesh.points,
        triangles=mesh.triangles,
        boundary_edges=mesh.boundary_edges,
        boundary_markers=mesh.boundary_markers,
        vorticity=mesh.vorticity,
        target_size=mesh.target_size,
        time=np.float64(snapshot.time),
        time_index=np.int64(snapshot.time_index),
        bounds=np.asarray(
            [snapshot.x[0], snapshot.x[-1], snapshot.y[0], snapshot.y[-1]]
        ),
        cylinder_center=np.asarray(cylinder_center),
        cylinder_radius=np.float64(radius_value),
    )


def save_legacy_vtk(path: Path, mesh: Mesh) -> None:
    """Write an ASCII VTK unstructured grid without a meshio dependency."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# vtk DataFile Version 3.0\n")
        stream.write("Vorticity-adapted cylinder mesh\n")
        stream.write("ASCII\nDATASET UNSTRUCTURED_GRID\n")
        stream.write(f"POINTS {len(mesh.points)} double\n")
        for x_value, y_value in mesh.points:
            stream.write(f"{x_value:.16g} {y_value:.16g} 0\n")
        stream.write(f"CELLS {len(mesh.triangles)} {4 * len(mesh.triangles)}\n")
        for a, b, c in mesh.triangles:
            stream.write(f"3 {a} {b} {c}\n")
        stream.write(f"CELL_TYPES {len(mesh.triangles)}\n")
        stream.writelines("5\n" for _ in mesh.triangles)  # VTK_TRIANGLE
        stream.write(f"POINT_DATA {len(mesh.points)}\n")
        stream.write("SCALARS vorticity double 1\nLOOKUP_TABLE default\n")
        stream.writelines(f"{value:.16g}\n" for value in mesh.vorticity)
        stream.write("SCALARS abs_vorticity double 1\nLOOKUP_TABLE default\n")
        stream.writelines(f"{abs(value):.16g}\n" for value in mesh.vorticity)
        stream.write("SCALARS target_size double 1\nLOOKUP_TABLE default\n")
        stream.writelines(f"{value:.16g}\n" for value in mesh.target_size)


def save_preview(
    path: Path,
    mesh: Mesh,
    snapshot: FlowSnapshot,
    vorticity_grid: np.ndarray,
    cylinder_center: Tuple[float, float],
    cylinder_radius: Optional[float],
) -> None:
    """Render the source vorticity and adapted mesh for quick visual QA."""

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(16, 3.2), constrained_layout=True)
    color_limit = float(np.percentile(np.abs(vorticity_grid), 99.0))
    if color_limit <= 0.0:
        color_limit = 1.0
    image = axis.imshow(
        vorticity_grid,
        extent=(snapshot.x[0], snapshot.x[-1], snapshot.y[0], snapshot.y[-1]),
        origin="lower",
        cmap="RdBu_r",
        vmin=-color_limit,
        vmax=color_limit,
        interpolation="bilinear",
        aspect="equal",
    )
    axis.triplot(
        mesh.points[:, 0],
        mesh.points[:, 1],
        mesh.triangles,
        color="black",
        linewidth=0.18,
        alpha=0.72,
    )
    if cylinder_radius is not None and cylinder_radius > 0.0:
        axis.add_patch(
            plt.Circle(cylinder_center, cylinder_radius, color="white", zorder=4)
        )
        axis.add_patch(
            plt.Circle(
                cylinder_center,
                cylinder_radius,
                fill=False,
                color="black",
                linewidth=0.8,
                zorder=5,
            )
        )
    axis.set(
        xlabel="x",
        ylabel="y",
        title=(
            f"Vorticity-adapted mesh at t={snapshot.time:g}  "
            f"({len(mesh.points):,} vertices, {len(mesh.triangles):,} triangles)"
        ),
    )
    figure.colorbar(image, ax=axis, label=r"vorticity $\omega$", shrink=0.85)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def print_quality_summary(mesh: Mesh) -> None:
    """Report element sizes in high- and low-vorticity regions."""

    vertices = mesh.points[mesh.triangles]
    twice_area = np.abs(
        _cross_2d(
            vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
        )
    )
    areas = 0.5 * twice_area
    cell_vorticity = np.mean(np.abs(mesh.vorticity[mesh.triangles]), axis=1)
    low_threshold, high_threshold = np.percentile(cell_vorticity, [50.0, 90.0])
    low_areas = areas[cell_vorticity <= low_threshold]
    high_areas = areas[cell_vorticity >= high_threshold]
    ratio = float(np.median(high_areas) / np.median(low_areas))
    print(
        f"Mesh: {len(mesh.points):,} vertices, {len(mesh.triangles):,} triangles"
    )
    print(
        "Median triangle area: "
        f"low-vorticity={np.median(low_areas):.4g}, "
        f"high-vorticity={np.median(high_areas):.4g}, ratio={ratio:.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a vorticity-adapted unstructured triangular mesh."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("dataset/cylinder2d.nc"),
        help="input NetCDF file (default: dataset/cylinder2d.nc)",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=-1,
        help="snapshot index; negative indices count from the end (default: -1)",
    )
    parser.add_argument(
        "--min-size",
        type=float,
        default=0.0125,
        help="target edge length in strongest-vorticity regions (default: 0.0125)",
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=0.10,
        help="target edge length in nearly irrotational regions (default: 0.10)",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.0,
        help="Gaussian smoothing in source-grid cells (default: 1.0)",
    )
    parser.add_argument(
        "--lower-percentile",
        type=float,
        default=50.0,
        help="vorticity percentile mapped to max-size (default: 50)",
    )
    parser.add_argument(
        "--upper-percentile",
        type=float,
        default=99.5,
        help="vorticity percentile mapped to min-size (default: 99.5)",
    )
    parser.add_argument(
        "--monitor-exponent",
        type=float,
        default=1.25,
        help="greater values concentrate refinement near vorticity peaks (default: 1.25)",
    )
    parser.add_argument("--seed", type=int, default=7, help="sampling RNG seed")
    parser.add_argument(
        "--cylinder-x", type=float, default=0.0, help="cylinder center x"
    )
    parser.add_argument(
        "--cylinder-y", type=float, default=0.0, help="cylinder center y"
    )
    parser.add_argument(
        "--cylinder-radius",
        type=float,
        default=None,
        help="override the radius stored in the NetCDF file",
    )
    parser.add_argument(
        "--no-cylinder",
        action="store_true",
        help="mesh the entire rectangular domain without an obstacle hole",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/cylinder2d_adaptive_mesh.npz"),
        help="NPZ output path",
    )
    parser.add_argument(
        "--vtk",
        type=lambda value: Path(value) if value else None,
        default=Path("outputs/cylinder2d_adaptive_mesh.vtk"),
        help="VTK output path; use an empty string to disable",
    )
    parser.add_argument(
        "--plot",
        type=lambda value: Path(value) if value else None,
        default=Path("outputs/cylinder2d_adaptive_mesh.png"),
        help="preview PNG path; use an empty string to disable",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_size <= 0.0 or args.max_size <= 0.0:
        raise ValueError("min-size and max-size must be positive")
    if args.min_size >= args.max_size:
        raise ValueError("min-size must be smaller than max-size")
    if not 0.0 <= args.lower_percentile < args.upper_percentile <= 100.0:
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")
    if args.monitor_exponent <= 0.0:
        raise ValueError("monitor-exponent must be positive")

    snapshot = load_snapshot(args.input, args.time_index)
    if args.no_cylinder:
        cylinder_radius = None
    elif args.cylinder_radius is not None:
        cylinder_radius = args.cylinder_radius
    else:
        cylinder_radius = snapshot.cylinder_radius
    cylinder_center = (args.cylinder_x, args.cylinder_y)
    if cylinder_radius is not None and cylinder_radius <= 0.0:
        raise ValueError("cylinder-radius must be positive; use --no-cylinder to disable it")

    print(
        f"Loaded {args.input}: t={snapshot.time:g} (index {snapshot.time_index}), "
        f"grid={len(snapshot.x)}x{len(snapshot.y)}"
    )
    mesh, vorticity_grid, _ = generate_mesh(
        snapshot=snapshot,
        min_size=args.min_size,
        max_size=args.max_size,
        smooth_sigma=args.smooth_sigma,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        exponent=args.monitor_exponent,
        seed=args.seed,
        cylinder_center=cylinder_center,
        cylinder_radius=cylinder_radius,
    )
    print_quality_summary(mesh)

    save_npz(
        args.output, mesh, snapshot, cylinder_center, cylinder_radius
    )
    print(f"Wrote {args.output}")
    if args.vtk is not None:
        save_legacy_vtk(args.vtk, mesh)
        print(f"Wrote {args.vtk}")
    if args.plot is not None:
        save_preview(
            args.plot,
            mesh,
            snapshot,
            vorticity_grid,
            cylinder_center,
            cylinder_radius,
        )
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
