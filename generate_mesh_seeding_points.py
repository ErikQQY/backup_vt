#!/usr/bin/env python3
"""Generate a vorticity-adapted triangular mesh and pathline seed points.

The script computes the scalar vorticity

    omega = dv/dx - du/dy

at one time step, turns its smoothed magnitude into a target edge-length
field, samples that field with a variable-radius Poisson sampler, and finally
builds an unstructured Delaunay triangulation.

Pathline seed points are then generated directly from the vertices of the
vorticity-adapted mesh. Because high-vorticity regions contain finer mesh
elements and therefore more mesh vertices, this automatically produces a
larger number of seed points in high-vorticity regions and fewer seed points
in low-vorticity regions.

The cylinder is represented as an explicit polygonal hole.

Examples
--------
Generate the default mesh and seed points::

    MPLCONFIGDIR=/tmp/pyflowvis-mpl .venv/bin/python generate_mesh.py

Use a different snapshot::

    .venv/bin/python generate_mesh.py --time-index 1000

Use every second eligible mesh vertex as a seed::

    .venv/bin/python generate_mesh.py --seed-stride 2
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay, QhullError


# ============================================================
# Data structures
# ============================================================

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
    """Generated vorticity-adapted triangular mesh."""

    points: np.ndarray
    triangles: np.ndarray
    boundary_edges: np.ndarray
    boundary_markers: np.ndarray
    vorticity: np.ndarray
    target_size: np.ndarray


@dataclass(frozen=True)
class SeedPoints:
    """Seed points selected from adaptive-mesh vertices."""

    points: np.ndarray
    mesh_point_ids: np.ndarray
    vorticity: np.ndarray
    target_size: np.ndarray


# ============================================================
# Load flow field
# ============================================================

def load_snapshot(path: Path, time_index: int) -> FlowSnapshot:
    """Load one snapshot without reading the full time-dependent dataset."""

    with nc.Dataset(path, "r") as dataset:

        required = {"u", "v", "xdim", "ydim", "tdim"}
        missing = sorted(required.difference(dataset.variables))

        if missing:
            raise ValueError(
                f"{path} is missing variables: {', '.join(missing)}"
            )

        x = np.asarray(
            dataset.variables["xdim"][:],
            dtype=np.float64,
        )

        y = np.asarray(
            dataset.variables["ydim"][:],
            dtype=np.float64,
        )

        times = np.asarray(
            dataset.variables["tdim"][:],
            dtype=np.float64,
        )

        resolved_index = (
            time_index
            if time_index >= 0
            else len(times) + time_index
        )

        if not 0 <= resolved_index < len(times):
            raise ValueError(
                f"time index {time_index} is outside "
                f"[-{len(times)}, {len(times) - 1}]"
            )

        u = np.ma.filled(
            dataset.variables["u"][resolved_index, :, :],
            np.nan,
        ).astype(np.float64)

        v = np.ma.filled(
            dataset.variables["v"][resolved_index, :, :],
            np.nan,
        ).astype(np.float64)

        radius = None

        if "radius" in dataset.variables:
            radius = float(
                np.asarray(dataset.variables["radius"][:]).ravel()[0]
            )

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(
            "xdim and ydim must be one-dimensional coordinate arrays"
        )

    if u.shape != (len(y), len(x)) or v.shape != u.shape:
        raise ValueError(
            f"expected u and v to have shape {(len(y), len(x))}, "
            f"got {u.shape}"
        )

    if not np.all(np.diff(x) > 0.0):
        raise ValueError("xdim must be strictly increasing")

    if not np.all(np.diff(y) > 0.0):
        raise ValueError("ydim must be strictly increasing")

    if not np.all(np.isfinite(u)):
        raise ValueError("selected u snapshot contains missing values")

    if not np.all(np.isfinite(v)):
        raise ValueError("selected v snapshot contains missing values")

    return FlowSnapshot(
        x=x,
        y=y,
        u=u,
        v=v,
        time=float(times[resolved_index]),
        time_index=resolved_index,
        cylinder_radius=radius,
    )


# ============================================================
# Vorticity monitor
# ============================================================

def build_monitor(
    snapshot: FlowSnapshot,
    min_size: float,
    max_size: float,
    smooth_sigma: float,
    lower_percentile: float,
    upper_percentile: float,
    exponent: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return vorticity and target edge-length field."""

    dvdx = np.gradient(
        snapshot.v,
        snapshot.x,
        axis=1,
        edge_order=2,
    )

    dudy = np.gradient(
        snapshot.u,
        snapshot.y,
        axis=0,
        edge_order=2,
    )

    vorticity = dvdx - dudy

    magnitude = np.abs(vorticity)

    if smooth_sigma > 0.0:
        magnitude = gaussian_filter(
            magnitude,
            sigma=smooth_sigma,
            mode="nearest",
        )

    low, high = np.percentile(
        magnitude,
        [lower_percentile, upper_percentile],
    )

    if high <= low + np.finfo(np.float64).eps:

        normalized = np.zeros_like(magnitude)

    else:

        normalized = np.clip(
            (magnitude - low) / (high - low),
            0.0,
            1.0,
        )

    refinement = normalized ** exponent

    target_size = (
        max_size
        - (max_size - min_size) * refinement
    )

    return vorticity, target_size


# ============================================================
# Boundary sampling
# ============================================================

def _adaptive_segment(
    start: Sequence[float],
    end: Sequence[float],
    size_at,
    min_size: float,
) -> np.ndarray:
    """Sample a straight boundary using local target size."""

    start_array = np.asarray(
        start,
        dtype=np.float64,
    )

    end_array = np.asarray(
        end,
        dtype=np.float64,
    )

    direction = end_array - start_array

    length = float(
        np.linalg.norm(direction)
    )

    if length == 0.0:
        return start_array[None, :]

    direction /= length

    points = [start_array]

    distance = 0.0

    while distance < length:

        local_size = float(
            size_at(points[-1][None, :])[0]
        )

        distance = min(
            length,
            distance
            + max(
                0.75 * local_size,
                0.5 * min_size,
            ),
        )

        points.append(
            start_array + distance * direction
        )

    return np.asarray(points)


def make_boundary_points(
    bounds: Tuple[float, float, float, float],
    size_at,
    min_size: float,
    cylinder_center: Tuple[float, float],
    cylinder_radius: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Create outer and cylinder boundary vertices."""

    xmin, xmax, ymin, ymax = bounds

    sides = [
        ((xmin, ymin), (xmax, ymin)),
        ((xmax, ymin), (xmax, ymax)),
        ((xmax, ymax), (xmin, ymax)),
        ((xmin, ymax), (xmin, ymin)),
    ]

    outer_parts = [
        _adaptive_segment(
            a,
            b,
            size_at,
            min_size,
        )[:-1]
        for a, b in sides
    ]

    outer = np.vstack(outer_parts)

    if (
        cylinder_radius is None
        or cylinder_radius <= 0.0
    ):
        return (
            outer,
            np.empty(
                (0, 2),
                dtype=np.float64,
            ),
        )

    circumference = (
        2.0
        * math.pi
        * cylinder_radius
    )

    count = max(
        24,
        int(
            math.ceil(
                circumference
                / (0.55 * min_size)
            )
        ),
    )

    angles = np.linspace(
        0.0,
        2.0 * math.pi,
        count,
        endpoint=False,
    )

    cx, cy = cylinder_center

    circle = np.column_stack(
        (
            cx
            + cylinder_radius
            * np.cos(angles),

            cy
            + cylinder_radius
            * np.sin(angles),
        )
    )

    return outer, circle


# ============================================================
# Candidate points
# ============================================================

def make_candidates(
    bounds: Tuple[float, float, float, float],
    min_size: float,
    cylinder_center: Tuple[float, float],
    cylinder_radius: Optional[float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Create jittered triangular-lattice candidates."""

    xmin, xmax, ymin, ymax = bounds

    step = 0.70 * min_size

    row_step = (
        math.sqrt(3.0)
        * 0.5
        * step
    )

    rows = []

    y_value = ymin + row_step

    row = 0

    while y_value < ymax - row_step:

        offset = (
            0.5 * step
            if row % 2
            else 0.0
        )

        xs = np.arange(
            xmin + step + offset,
            xmax - step,
            step,
        )

        if len(xs):

            ys = np.full_like(
                xs,
                y_value,
            )

            rows.append(
                np.column_stack(
                    (xs, ys)
                )
            )

        row += 1

        y_value += row_step

    if not rows:
        raise ValueError(
            "min-size is too large for the dataset domain"
        )

    candidates = np.vstack(rows)

    candidates += rng.uniform(
        -0.12 * step,
        0.12 * step,
        candidates.shape,
    )

    inside = (
        (candidates[:, 0] > xmin)
        & (candidates[:, 0] < xmax)
        & (candidates[:, 1] > ymin)
        & (candidates[:, 1] < ymax)
    )

    if (
        cylinder_radius is not None
        and cylinder_radius > 0.0
    ):

        center = np.asarray(
            cylinder_center
        )

        radius_squared = np.sum(
            (candidates - center) ** 2,
            axis=1,
        )

        inside &= (
            radius_squared
            >
            (1.01 * cylinder_radius) ** 2
        )

    return candidates[inside]


# ============================================================
# Variable-radius Poisson sampling
# ============================================================

def variable_poisson_sample(
    boundary_points: np.ndarray,
    boundary_sizes: np.ndarray,
    candidates: np.ndarray,
    candidate_sizes: np.ndarray,
    min_size: float,
    max_size: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Enforce spatially varying minimum separation."""

    cell_size = (
        min_size
        / math.sqrt(2.0)
    )

    buckets: DefaultDict[
        Tuple[int, int],
        list,
    ] = defaultdict(list)

    accepted = [
        point.copy()
        for point
        in boundary_points
    ]

    accepted_sizes = [
        float(value)
        for value
        in boundary_sizes
    ]

    def cell_of(
        point: np.ndarray
    ) -> Tuple[int, int]:

        return (
            math.floor(
                point[0] / cell_size
            ),
            math.floor(
                point[1] / cell_size
            ),
        )

    for index, point in enumerate(accepted):

        buckets[
            cell_of(point)
        ].append(index)

    scale = max(
        max_size - min_size,
        np.finfo(np.float64).eps,
    )

    priority = (
        candidate_sizes
        + rng.uniform(
            0.0,
            0.04 * scale,
            len(candidates),
        )
    )

    order = np.argsort(
        priority,
        kind="stable",
    )

    for candidate_index in order:

        point = candidates[
            candidate_index
        ]

        local_size = float(
            candidate_sizes[
                candidate_index
            ]
        )

        cell_x, cell_y = cell_of(
            point
        )

        search_distance = (
            0.5
            * (
                local_size
                + max_size
            )
        )

        cell_radius = int(
            math.ceil(
                search_distance
                / cell_size
            )
        )

        valid = True

        for ix in range(
            cell_x - cell_radius,
            cell_x + cell_radius + 1,
        ):

            if not valid:
                break

            for iy in range(
                cell_y - cell_radius,
                cell_y + cell_radius + 1,
            ):

                for neighbor_index in buckets.get(
                    (ix, iy),
                    (),
                ):

                    neighbor = accepted[
                        neighbor_index
                    ]

                    required = (
                        0.82
                        * 0.5
                        * (
                            local_size
                            + accepted_sizes[
                                neighbor_index
                            ]
                        )
                    )

                    delta = (
                        point
                        - neighbor
                    )

                    if (
                        np.dot(
                            delta,
                            delta,
                        )
                        <
                        required ** 2
                    ):

                        valid = False
                        break

                if not valid:
                    break

        if valid:

            new_index = len(
                accepted
            )

            accepted.append(
                point.copy()
            )

            accepted_sizes.append(
                local_size
            )

            buckets[
                (cell_x, cell_y)
            ].append(
                new_index
            )

    return np.asarray(
        accepted,
        dtype=np.float64,
    )


# ============================================================
# Geometry helpers
# ============================================================

def _segment_distance_to_point(
    a: np.ndarray,
    b: np.ndarray,
    point: np.ndarray,
) -> np.ndarray:
    """Vectorized distance from point to segments."""

    direction = b - a

    denominator = np.einsum(
        "ij,ij->i",
        direction,
        direction,
    )

    parameter = np.einsum(
        "ij,ij->i",
        point - a,
        direction,
    ) / np.maximum(
        denominator,
        np.finfo(np.float64).eps,
    )

    parameter = np.clip(
        parameter,
        0.0,
        1.0,
    )

    nearest = (
        a
        + parameter[:, None]
        * direction
    )

    return np.linalg.norm(
        nearest - point,
        axis=1,
    )


def _cross_2d(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """2D cross product."""

    return (
        a[..., 0] * b[..., 1]
        -
        a[..., 1] * b[..., 0]
    )


# ============================================================
# Delaunay triangulation
# ============================================================

def triangulate(
    points: np.ndarray,
    circle_start: int,
    circle_count: int,
    cylinder_center: Tuple[float, float],
    cylinder_radius: Optional[float],
) -> np.ndarray:
    """Delaunay triangulate and remove cylinder-crossing cells."""

    try:

        triangles = (
            Delaunay(points)
            .simplices
            .copy()
        )

    except QhullError:

        triangles = (
            Delaunay(
                points,
                qhull_options="QJ",
            )
            .simplices
            .copy()
        )

    vertices = points[
        triangles
    ]

    signed_twice_area = _cross_2d(
        vertices[:, 1]
        - vertices[:, 0],

        vertices[:, 2]
        - vertices[:, 0],
    )

    keep = (
        np.abs(
            signed_twice_area
        )
        > 1.0e-14
    )

    if (
        cylinder_radius is not None
        and cylinder_radius > 0.0
        and circle_count
    ):

        center = np.asarray(
            cylinder_center,
            dtype=np.float64,
        )

        centroids = (
            vertices.mean(
                axis=1
            )
        )

        polygon_radius = (
            cylinder_radius
            * math.cos(
                math.pi
                / circle_count
            )
        )

        keep &= (
            np.linalg.norm(
                centroids - center,
                axis=1,
            )
            >= polygon_radius
        )

        circle_end = (
            circle_start
            + circle_count
        )

        circle_local = np.full(
            len(points),
            -1,
            dtype=np.int64,
        )

        circle_local[
            circle_start:circle_end
        ] = np.arange(
            circle_count
        )

        tolerance = max(
            1.0e-12,
            1.0e-6 * cylinder_radius,
        )

        for edge_a, edge_b in (
            (0, 1),
            (1, 2),
            (2, 0),
        ):

            ids_a = triangles[
                :, edge_a
            ]

            ids_b = triangles[
                :, edge_b
            ]

            local_a = circle_local[
                ids_a
            ]

            local_b = circle_local[
                ids_b
            ]

            difference = np.abs(
                local_a
                - local_b
            )

            adjacent_circle_edge = (
                (local_a >= 0)
                & (local_b >= 0)
                & (
                    (difference == 1)
                    |
                    (
                        difference
                        == circle_count - 1
                    )
                )
            )

            edge_distance = (
                _segment_distance_to_point(
                    points[ids_a],
                    points[ids_b],
                    center,
                )
            )

            keep &= (
                adjacent_circle_edge
                |
                (
                    edge_distance
                    >= polygon_radius
                    - tolerance
                )
            )

    triangles = triangles[
        keep
    ]

    selected_points = points[
        triangles
    ]

    clockwise = (
        _cross_2d(
            selected_points[:, 1]
            - selected_points[:, 0],

            selected_points[:, 2]
            - selected_points[:, 0],
        )
        < 0.0
    )

    triangles[
        clockwise,
        1
    ], triangles[
        clockwise,
        2
    ] = (
        triangles[
            clockwise,
            2
        ].copy(),

        triangles[
            clockwise,
            1
        ].copy(),
    )

    return triangles.astype(
        np.int32
    )


# ============================================================
# Boundary extraction
# ============================================================

def extract_boundary_edges(
    triangles: np.ndarray,
    circle_start: int,
    circle_count: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """Return exterior edges with markers."""

    edges = np.sort(
        np.vstack(
            (
                triangles[:, [0, 1]],
                triangles[:, [1, 2]],
                triangles[:, [2, 0]],
            )
        ),
        axis=1,
    )

    unique_edges, counts = np.unique(
        edges,
        axis=0,
        return_counts=True,
    )

    boundary_edges = (
        unique_edges[
            counts == 1
        ]
        .astype(
            np.int32
        )
    )

    boundary_markers = np.ones(
        len(
            boundary_edges
        ),
        dtype=np.int8,
    )

    if circle_count:

        circle_end = (
            circle_start
            + circle_count
        )

        on_circle = (
            (
                boundary_edges[:, 0]
                >= circle_start
            )
            &
            (
                boundary_edges[:, 0]
                < circle_end
            )
            &
            (
                boundary_edges[:, 1]
                >= circle_start
            )
            &
            (
                boundary_edges[:, 1]
                < circle_end
            )
        )

        boundary_markers[
            on_circle
        ] = 2

    return (
        boundary_edges,
        boundary_markers,
    )


# ============================================================
# Mesh generation
# ============================================================

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
) -> Tuple[
    Mesh,
    np.ndarray,
    np.ndarray,
]:
    """Run vorticity-based mesh adaptation."""

    bounds = (
        float(snapshot.x[0]),
        float(snapshot.x[-1]),
        float(snapshot.y[0]),
        float(snapshot.y[-1]),
    )

    vorticity_grid, size_grid = (
        build_monitor(
            snapshot,
            min_size,
            max_size,
            smooth_sigma,
            lower_percentile,
            upper_percentile,
            exponent,
        )
    )

    size_interpolator = (
        RegularGridInterpolator(
            (
                snapshot.y,
                snapshot.x,
            ),
            size_grid,
            bounds_error=False,
            fill_value=max_size,
        )
    )

    def size_at(
        xy: np.ndarray
    ) -> np.ndarray:

        return size_interpolator(
            np.column_stack(
                (
                    xy[:, 1],
                    xy[:, 0],
                )
            )
        )

    outer, circle = (
        make_boundary_points(
            bounds,
            size_at,
            min_size,
            cylinder_center,
            cylinder_radius,
        )
    )

    boundary = np.vstack(
        (
            outer,
            circle,
        )
    )

    boundary_sizes = (
        size_at(
            boundary
        )
    )

    if len(circle):

        boundary_sizes[
            len(outer):
        ] = min_size

    rng = np.random.default_rng(
        seed
    )

    candidates = make_candidates(
        bounds,
        min_size,
        cylinder_center,
        cylinder_radius,
        rng,
    )

    candidate_sizes = (
        size_at(
            candidates
        )
    )

    points = (
        variable_poisson_sample(
            boundary,
            boundary_sizes,
            candidates,
            candidate_sizes,
            min_size,
            max_size,
            rng,
        )
    )

    triangles = triangulate(
        points,
        circle_start=len(outer),
        circle_count=len(circle),
        cylinder_center=cylinder_center,
        cylinder_radius=cylinder_radius,
    )

    (
        boundary_edges,
        boundary_markers,
    ) = extract_boundary_edges(
        triangles,
        circle_start=len(outer),
        circle_count=len(circle),
    )

    vorticity_interpolator = (
        RegularGridInterpolator(
            (
                snapshot.y,
                snapshot.x,
            ),
            vorticity_grid,
            bounds_error=False,
            fill_value=0.0,
        )
    )

    query = np.column_stack(
        (
            points[:, 1],
            points[:, 0],
        )
    )

    mesh = Mesh(
        points=points,
        triangles=triangles,
        boundary_edges=boundary_edges,
        boundary_markers=boundary_markers,
        vorticity=np.asarray(
            vorticity_interpolator(
                query
            )
        ),
        target_size=np.asarray(
            size_interpolator(
                query
            )
        ),
    )

    return (
        mesh,
        vorticity_grid,
        size_grid,
    )


# ============================================================
# Seed-point generation
# ============================================================

def generate_seed_points(
    mesh: Mesh,
    include_boundary: bool = False,
    stride: int = 1,
) -> SeedPoints:
    """Generate pathline seeds from adaptive-mesh vertices.

    Because mesh density is already controlled by vorticity magnitude,
    directly using mesh vertices naturally yields more seed points in
    high-vorticity regions and fewer seed points in low-vorticity regions.
    """

    if stride <= 0:
        raise ValueError(
            "seed stride must be a positive integer"
        )

    number_of_points = len(
        mesh.points
    )

    eligible = np.ones(
        number_of_points,
        dtype=bool,
    )

    # Remove all geometrical boundary points by default.
    if (
        not include_boundary
        and len(mesh.boundary_edges)
    ):

        boundary_ids = np.unique(
            mesh.boundary_edges.ravel()
        )

        eligible[
            boundary_ids
        ] = False

    mesh_point_ids = (
        np.flatnonzero(
            eligible
        )
    )

    # Optional downsampling.
    mesh_point_ids = (
        mesh_point_ids[
            ::stride
        ]
    )

    return SeedPoints(
        points=(
            mesh.points[
                mesh_point_ids
            ].copy()
        ),
        mesh_point_ids=(
            mesh_point_ids.astype(
                np.int32
            )
        ),
        vorticity=(
            mesh.vorticity[
                mesh_point_ids
            ].copy()
        ),
        target_size=(
            mesh.target_size[
                mesh_point_ids
            ].copy()
        ),
    )


# ============================================================
# Save mesh
# ============================================================

def save_npz(
    path: Path,
    mesh: Mesh,
    snapshot: FlowSnapshot,
    cylinder_center: Tuple[float, float],
    cylinder_radius: Optional[float],
) -> None:
    """Save mesh arrays and metadata."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    radius_value = (
        np.nan
        if cylinder_radius is None
        else cylinder_radius
    )

    np.savez_compressed(
        path,
        points=mesh.points,
        triangles=mesh.triangles,
        boundary_edges=mesh.boundary_edges,
        boundary_markers=mesh.boundary_markers,
        vorticity=mesh.vorticity,
        target_size=mesh.target_size,
        time=np.float64(
            snapshot.time
        ),
        time_index=np.int64(
            snapshot.time_index
        ),
        bounds=np.asarray(
            [
                snapshot.x[0],
                snapshot.x[-1],
                snapshot.y[0],
                snapshot.y[-1],
            ]
        ),
        cylinder_center=np.asarray(
            cylinder_center
        ),
        cylinder_radius=np.float64(
            radius_value
        ),
    )


def save_legacy_vtk(
    path: Path,
    mesh: Mesh,
) -> None:
    """Write an ASCII VTK unstructured grid."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as stream:

        stream.write(
            "# vtk DataFile Version 3.0\n"
        )

        stream.write(
            "Vorticity-adapted cylinder mesh\n"
        )

        stream.write(
            "ASCII\n"
        )

        stream.write(
            "DATASET UNSTRUCTURED_GRID\n"
        )

        stream.write(
            f"POINTS {len(mesh.points)} double\n"
        )

        for x_value, y_value in mesh.points:

            stream.write(
                f"{x_value:.16g} "
                f"{y_value:.16g} "
                "0\n"
            )

        stream.write(
            f"CELLS {len(mesh.triangles)} "
            f"{4 * len(mesh.triangles)}\n"
        )

        for a, b, c in mesh.triangles:

            stream.write(
                f"3 {a} {b} {c}\n"
            )

        stream.write(
            f"CELL_TYPES {len(mesh.triangles)}\n"
        )

        stream.writelines(
            "5\n"
            for _ in mesh.triangles
        )

        stream.write(
            f"POINT_DATA {len(mesh.points)}\n"
        )

        stream.write(
            "SCALARS vorticity double 1\n"
        )

        stream.write(
            "LOOKUP_TABLE default\n"
        )

        stream.writelines(
            f"{value:.16g}\n"
            for value in mesh.vorticity
        )

        stream.write(
            "SCALARS abs_vorticity double 1\n"
        )

        stream.write(
            "LOOKUP_TABLE default\n"
        )

        stream.writelines(
            f"{abs(value):.16g}\n"
            for value in mesh.vorticity
        )

        stream.write(
            "SCALARS target_size double 1\n"
        )

        stream.write(
            "LOOKUP_TABLE default\n"
        )

        stream.writelines(
            f"{value:.16g}\n"
            for value in mesh.target_size
        )


# ============================================================
# Save seeds
# ============================================================

def save_seed_points_npz(
    path: Path,
    seeds: SeedPoints,
    snapshot: FlowSnapshot,
) -> None:
    """Save seed points."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,
        points=seeds.points,
        mesh_point_ids=seeds.mesh_point_ids,
        vorticity=seeds.vorticity,
        abs_vorticity=np.abs(
            seeds.vorticity
        ),
        target_size=seeds.target_size,
        time=np.float64(
            snapshot.time
        ),
        time_index=np.int64(
            snapshot.time_index
        ),
    )


def save_seed_points_vtk(
    path: Path,
    seeds: SeedPoints,
) -> None:
    """Write seed points as VTK POLYDATA."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as stream:

        stream.write(
            "# vtk DataFile Version 3.0\n"
        )

        stream.write(
            "Vorticity-guided pathline seed points\n"
        )

        stream.write(
            "ASCII\n"
        )

        stream.write(
            "DATASET POLYDATA\n"
        )

        stream.write(
            f"POINTS {len(seeds.points)} double\n"
        )

        for x_value, y_value in seeds.points:

            stream.write(
                f"{x_value:.16g} "
                f"{y_value:.16g} "
                "0\n"
            )

        stream.write(
            f"VERTICES {len(seeds.points)} "
            f"{2 * len(seeds.points)}\n"
        )

        for index in range(
            len(seeds.points)
        ):

            stream.write(
                f"1 {index}\n"
            )

        stream.write(
            f"POINT_DATA {len(seeds.points)}\n"
        )

        stream.write(
            "COLOR_SCALARS seeding_point_color 3\n"
        )

        stream.writelines(
            "0 0 0\n"
            for _ in seeds.points
        )

        stream.write(
            "SCALARS vorticity double 1\n"
        )

        stream.write(
            "LOOKUP_TABLE default\n"
        )

        for value in seeds.vorticity:

            stream.write(
                f"{value:.16g}\n"
            )

        stream.write(
            "SCALARS abs_vorticity double 1\n"
        )

        stream.write(
            "LOOKUP_TABLE default\n"
        )

        for value in np.abs(
            seeds.vorticity
        ):

            stream.write(
                f"{value:.16g}\n"
            )

        stream.write(
            "SCALARS target_size double 1\n"
        )

        stream.write(
            "LOOKUP_TABLE default\n"
        )

        for value in seeds.target_size:

            stream.write(
                f"{value:.16g}\n"
            )

        stream.write(
            "SCALARS mesh_point_id int 1\n"
        )

        stream.write(
            "LOOKUP_TABLE default\n"
        )

        for value in seeds.mesh_point_ids:

            stream.write(
                f"{int(value)}\n"
            )


# ============================================================
# Preview
# ============================================================

def save_preview(
    path: Path,
    mesh: Mesh,
    seeds: SeedPoints,
    snapshot: FlowSnapshot,
    vorticity_grid: np.ndarray,
    cylinder_center: Tuple[float, float],
    cylinder_radius: Optional[float],
) -> None:
    """Render vorticity, adaptive mesh and seed points."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(16, 3.2),
        constrained_layout=True,
    )

    color_limit = float(
        np.percentile(
            np.abs(vorticity_grid),
            99.0,
        )
    )

    if color_limit <= 0.0:
        color_limit = 1.0

    image = axis.imshow(
        vorticity_grid,
        extent=(
            snapshot.x[0],
            snapshot.x[-1],
            snapshot.y[0],
            snapshot.y[-1],
        ),
        origin="lower",
        cmap="RdBu_r",
        vmin=-color_limit,
        vmax=color_limit,
        interpolation="bilinear",
        aspect="equal",
    )

    # Adaptive triangular mesh.
    axis.triplot(
        mesh.points[:, 0],
        mesh.points[:, 1],
        mesh.triangles,
        color="black",
        linewidth=0.18,
        alpha=0.35,
        zorder=2,
    )

    # Seed points.
    axis.scatter(
        seeds.points[:, 0],
        seeds.points[:, 1],
        s=5.0,
        c="black",
        marker="o",
        linewidths=0.0,
        alpha=0.95,
        zorder=3,
        label="Seed points",
    )

    # Cylinder.
    if (
        cylinder_radius is not None
        and cylinder_radius > 0.0
    ):

        axis.add_patch(
            plt.Circle(
                cylinder_center,
                cylinder_radius,
                color="white",
                zorder=4,
            )
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
            f"Vorticity-adapted mesh and seed points "
            f"at t={snapshot.time:g}\n"
            f"{len(mesh.points):,} vertices, "
            f"{len(mesh.triangles):,} triangles, "
            f"{len(seeds.points):,} seeds"
        ),
    )

    axis.legend(
        loc="upper right",
        frameon=True,
    )

    figure.colorbar(
        image,
        ax=axis,
        label=r"vorticity $\omega$",
        shrink=0.85,
    )

    figure.savefig(
        path,
        dpi=220,
    )

    plt.close(
        figure
    )


# ============================================================
# Quality diagnostics
# ============================================================

def print_quality_summary(
    mesh: Mesh,
) -> None:
    """Report mesh-area statistics."""

    vertices = mesh.points[
        mesh.triangles
    ]

    twice_area = np.abs(
        _cross_2d(
            vertices[:, 1]
            - vertices[:, 0],

            vertices[:, 2]
            - vertices[:, 0],
        )
    )

    areas = (
        0.5
        * twice_area
    )

    cell_vorticity = np.mean(
        np.abs(
            mesh.vorticity[
                mesh.triangles
            ]
        ),
        axis=1,
    )

    (
        low_threshold,
        high_threshold,
    ) = np.percentile(
        cell_vorticity,
        [50.0, 90.0],
    )

    low_areas = areas[
        cell_vorticity
        <= low_threshold
    ]

    high_areas = areas[
        cell_vorticity
        >= high_threshold
    ]

    ratio = float(
        np.median(
            high_areas
        )
        /
        np.median(
            low_areas
        )
    )

    print(
        f"Mesh: "
        f"{len(mesh.points):,} vertices, "
        f"{len(mesh.triangles):,} triangles"
    )

    print(
        "Median triangle area: "
        f"low-vorticity="
        f"{np.median(low_areas):.4g}, "
        f"high-vorticity="
        f"{np.median(high_areas):.4g}, "
        f"ratio={ratio:.3f}"
    )


# ============================================================
# Command-line arguments
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Generate a vorticity-adapted triangular mesh "
            "and pathline seed points."
        )
    )

    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path(
            "dataset/cylinder2d.nc"
        ),
        help=(
            "input NetCDF file "
            "(default: dataset/cylinder2d.nc)"
        ),
    )

    parser.add_argument(
        "--time-index",
        type=int,
        default=-1,
        help=(
            "snapshot index; negative indices count "
            "from the end (default: -1)"
        ),
    )

    parser.add_argument(
        "--min-size",
        type=float,
        default=0.0125,
        help=(
            "target edge length in strongest-vorticity "
            "regions (default: 0.0125)"
        ),
    )

    parser.add_argument(
        "--max-size",
        type=float,
        default=0.10,
        help=(
            "target edge length in low-vorticity "
            "regions (default: 0.10)"
        ),
    )

    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.0,
        help=(
            "Gaussian smoothing in source-grid cells "
            "(default: 1.0)"
        ),
    )

    parser.add_argument(
        "--lower-percentile",
        type=float,
        default=50.0,
        help=(
            "vorticity percentile mapped to max-size "
            "(default: 50)"
        ),
    )

    parser.add_argument(
        "--upper-percentile",
        type=float,
        default=99.5,
        help=(
            "vorticity percentile mapped to min-size "
            "(default: 99.5)"
        ),
    )

    parser.add_argument(
        "--monitor-exponent",
        type=float,
        default=1.25,
        help=(
            "greater values concentrate refinement "
            "near vorticity peaks (default: 1.25)"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="mesh sampling RNG seed",
    )

    parser.add_argument(
        "--seed-stride",
        type=int,
        default=1,
        help=(
            "use every N-th eligible mesh vertex as "
            "a pathline seed (default: 1)"
        ),
    )

    parser.add_argument(
        "--include-boundary-seeds",
        action="store_true",
        help=(
            "include outer- and cylinder-boundary "
            "vertices as seed points"
        ),
    )

    parser.add_argument(
        "--cylinder-x",
        type=float,
        default=0.0,
        help="cylinder center x",
    )

    parser.add_argument(
        "--cylinder-y",
        type=float,
        default=0.0,
        help="cylinder center y",
    )

    parser.add_argument(
        "--cylinder-radius",
        type=float,
        default=None,
        help=(
            "override radius stored in NetCDF file"
        ),
    )

    parser.add_argument(
        "--no-cylinder",
        action="store_true",
        help=(
            "mesh entire rectangular domain "
            "without cylinder"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/cylinder2d_adaptive_mesh.npz"
        ),
        help="mesh NPZ output path",
    )

    parser.add_argument(
        "--vtk",
        type=lambda value: (
            Path(value)
            if value
            else None
        ),
        default=Path(
            "outputs/cylinder2d_adaptive_mesh.vtk"
        ),
        help=(
            "mesh VTK output path; "
            "use empty string to disable"
        ),
    )

    parser.add_argument(
        "--seed-output",
        type=Path,
        default=Path(
            "outputs/cylinder2d_seeding_points.npz"
        ),
        help=(
            "seed-point NPZ output path"
        ),
    )

    parser.add_argument(
        "--seed-vtk",
        type=lambda value: (
            Path(value)
            if value
            else None
        ),
        default=Path(
            "outputs/cylinder2d_seeding_points.vtk"
        ),
        help=(
            "seed-point VTK output path; "
            "use empty string to disable"
        ),
    )

    parser.add_argument(
        "--plot",
        type=lambda value: (
            Path(value)
            if value
            else None
        ),
        default=Path(
            "outputs/cylinder2d_adaptive_mesh_seeds.png"
        ),
        help=(
            "preview PNG path; "
            "use empty string to disable"
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    if (
        args.min_size <= 0.0
        or args.max_size <= 0.0
    ):
        raise ValueError(
            "min-size and max-size must be positive"
        )

    if (
        args.min_size
        >= args.max_size
    ):
        raise ValueError(
            "min-size must be smaller than max-size"
        )

    if not (
        0.0
        <= args.lower_percentile
        < args.upper_percentile
        <= 100.0
    ):
        raise ValueError(
            "percentiles must satisfy "
            "0 <= lower < upper <= 100"
        )

    if (
        args.monitor_exponent
        <= 0.0
    ):
        raise ValueError(
            "monitor-exponent must be positive"
        )

    if (
        args.seed_stride
        <= 0
    ):
        raise ValueError(
            "seed-stride must be positive"
        )

    # --------------------------------------------------------
    # Load one velocity snapshot
    # --------------------------------------------------------

    snapshot = load_snapshot(
        args.input,
        args.time_index,
    )

    # --------------------------------------------------------
    # Cylinder configuration
    # --------------------------------------------------------

    if args.no_cylinder:

        cylinder_radius = None

    elif (
        args.cylinder_radius
        is not None
    ):

        cylinder_radius = (
            args.cylinder_radius
        )

    else:

        cylinder_radius = (
            snapshot.cylinder_radius
        )

    cylinder_center = (
        args.cylinder_x,
        args.cylinder_y,
    )

    if (
        cylinder_radius is not None
        and cylinder_radius <= 0.0
    ):
        raise ValueError(
            "cylinder-radius must be positive; "
            "use --no-cylinder to disable it"
        )

    print(
        f"Loaded {args.input}: "
        f"t={snapshot.time:g} "
        f"(index {snapshot.time_index}), "
        f"grid={len(snapshot.x)}x{len(snapshot.y)}"
    )

    # --------------------------------------------------------
    # Generate vorticity-adapted mesh
    # --------------------------------------------------------

    (
        mesh,
        vorticity_grid,
        _,
    ) = generate_mesh(
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

    print_quality_summary(
        mesh
    )

    # --------------------------------------------------------
    # Generate pathline seed points from mesh vertices
    # --------------------------------------------------------

    seeds = generate_seed_points(
        mesh,
        include_boundary=(
            args.include_boundary_seeds
        ),
        stride=args.seed_stride,
    )

    print(
        f"Seed points: "
        f"{len(seeds.points):,} "
        f"from "
        f"{len(mesh.points):,} "
        f"mesh vertices"
    )

    # --------------------------------------------------------
    # Save adaptive mesh
    # --------------------------------------------------------

    save_npz(
        args.output,
        mesh,
        snapshot,
        cylinder_center,
        cylinder_radius,
    )

    print(
        f"Wrote {args.output}"
    )

    if (
        args.vtk
        is not None
    ):

        save_legacy_vtk(
            args.vtk,
            mesh,
        )

        print(
            f"Wrote {args.vtk}"
        )

    # --------------------------------------------------------
    # Save seeding points
    # --------------------------------------------------------

    save_seed_points_npz(
        args.seed_output,
        seeds,
        snapshot,
    )

    print(
        f"Wrote {args.seed_output}"
    )

    if (
        args.seed_vtk
        is not None
    ):

        save_seed_points_vtk(
            args.seed_vtk,
            seeds,
        )

        print(
            f"Wrote {args.seed_vtk}"
        )

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    if (
        args.plot
        is not None
    ):

        save_preview(
            args.plot,
            mesh,
            seeds,
            snapshot,
            vorticity_grid,
            cylinder_center,
            cylinder_radius,
        )

        print(
            f"Wrote {args.plot}"
        )


if __name__ == "__main__":
    main()
