import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.io import loadmat

DEFAULT_FOLDERS = ("General",)
ELEVATION_KEY = "sZelev"
LINEAR_FEATURES_KEY = "BWLF"
INACCESSIBLE_KEY = "BWInac"
EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class InitialCondition:
    incident_index: int
    ipp_lat: float
    ipp_lon: float
    find_lat: float
    find_lon: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render SAR .mat maps in the style of SAR Map Data/Figure1.png."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("SAR Map Data"),
        help="Folder that contains the SAR map folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots") / "SAR Maps Data",
        help="Folder where PNG images will be written.",
    )
    parser.add_argument(
        "--initial-conditions",
        type=Path,
        default=None,
        help="CSV with incident_index, IPP_lat/lon, and find_lat/lon.",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=list(DEFAULT_FOLDERS),
        help="SAR Map Data subfolders to render. Defaults to the 65 unique maps.",
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=None,
        help="Optional incident indices to render, e.g. --indices 39 48.",
    )
    parser.add_argument(
        "--meters-per-cell",
        type=float,
        default=10.0,
        help="Map resolution used to convert lat/lon offsets into grid cells.",
    )
    parser.add_argument(
        "--contour-count",
        type=int,
        default=32,
        help="Number of elevation contour levels to draw.",
    )
    parser.add_argument(
        "--contour-stride",
        type=int,
        default=6,
        help="Downsampling stride for elevation contours.",
    )
    parser.add_argument(
        "--linear-color",
        default="#b2183b",
        help="Color for BWLF linear-feature pixels.",
    )
    parser.add_argument(
        "--linear-width",
        type=int,
        default=2,
        help="Display-only dilation radius for BWLF lines.",
    )
    parser.add_argument(
        "--inaccessible-color",
        default="#1f78b4",
        help="Color for BWInac inaccessible-area pixels.",
    )
    return parser.parse_args()


def get_2d_array(
    mat_data: dict[str, np.ndarray], key: str, mat_path: Path
) -> np.ndarray:
    if key not in mat_data:
        raise KeyError(f"{mat_path} is missing required object {key!r}")

    array = np.asarray(mat_data[key]).squeeze()
    if array.ndim != 2:
        raise ValueError(f"{mat_path} object {key!r} must be 2D, got {array.shape}")
    return array


def load_map_layers(mat_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mat_data = loadmat(mat_path)
    elevation = get_2d_array(mat_data, ELEVATION_KEY, mat_path).astype(float)
    linear_features = get_2d_array(mat_data, LINEAR_FEATURES_KEY, mat_path) > 0
    inaccessible = get_2d_array(mat_data, INACCESSIBLE_KEY, mat_path) > 0

    if (
        elevation.shape != linear_features.shape
        or elevation.shape != inaccessible.shape
    ):
        raise ValueError(
            f"{mat_path} layer shapes do not match: "
            f"{ELEVATION_KEY}={elevation.shape}, "
            f"{LINEAR_FEATURES_KEY}={linear_features.shape}, "
            f"{INACCESSIBLE_KEY}={inaccessible.shape}"
        )

    return elevation, linear_features, inaccessible


def read_initial_conditions(csv_path: Path) -> dict[int, InitialCondition]:
    with csv_path.open(newline="") as file:
        first_line = file.readline()
        if first_line.startswith("incident_index"):
            file.seek(0)
        rows = csv.DictReader(file)
        conditions = {
            int(row["incident_index"]): InitialCondition(
                incident_index=int(row["incident_index"]),
                ipp_lat=float(row["IPP_lat"]),
                ipp_lon=float(row["IPP_lon"]),
                find_lat=float(row["find_lat"]),
                find_lon=float(row["find_lon"]),
            )
            for row in rows
        }

    if not conditions:
        raise ValueError(f"No initial conditions found in {csv_path}")
    return conditions


def extract_map_index(mat_path: Path) -> int:
    match = re.search(r"_(\d+)$", mat_path.stem)
    if not match:
        raise ValueError(f"Could not extract incident index from {mat_path.name}")
    return int(match.group(1))


def latlon_offset_m(
    lat: float,
    lon: float,
    reference_lat: float,
    reference_lon: float,
) -> tuple[float, float]:
    mean_lat_rad = math.radians((lat + reference_lat) / 2)
    east_m = math.radians(lon - reference_lon) * EARTH_RADIUS_M * math.cos(mean_lat_rad)
    north_m = math.radians(lat - reference_lat) * EARTH_RADIUS_M
    return east_m, north_m


def initial_condition_cells(
    condition: InitialCondition,
    shape: tuple[int, int],
    meters_per_cell: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if meters_per_cell <= 0:
        raise ValueError("--meters-per-cell must be positive")

    rows, cols = shape
    ipp_cell = (cols / 2, rows / 2)
    find_east_m, find_north_m = latlon_offset_m(
        condition.find_lat,
        condition.find_lon,
        condition.ipp_lat,
        condition.ipp_lon,
    )
    find_cell = (
        ipp_cell[0] + find_east_m / meters_per_cell,
        ipp_cell[1] + find_north_m / meters_per_cell,
    )
    return ipp_cell, find_cell


def render_mask_overlay(
    ax: Axes,
    mask: np.ndarray,
    color: str,
    alpha: float,
    zorder: int,
) -> None:
    if not mask.any():
        return

    rows, cols = mask.shape
    overlay = np.ma.masked_where(~mask, mask)
    ax.imshow(
        overlay,
        cmap=ListedColormap([color]),
        origin="lower",
        extent=(0, cols, 0, rows),
        interpolation="nearest",
        vmin=0,
        vmax=1,
        alpha=alpha,
        zorder=zorder,
    )


def dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0 or not mask.any():
        return mask
    return binary_dilation(mask, iterations=iterations)


def render_elevation_contours(
    ax: Axes,
    elevation: np.ndarray,
    contour_count: int,
    contour_stride: int,
) -> None:
    rows, cols = elevation.shape
    stride = max(1, contour_stride)
    sampled = elevation[::stride, ::stride]
    finite = sampled[np.isfinite(sampled)]
    if finite.size == 0:
        return

    low, high = np.nanpercentile(finite, [2, 98])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
    if high <= low:
        return

    x = np.arange(0, cols, stride)
    y = np.arange(0, rows, stride)
    levels = np.linspace(low, high, max(2, contour_count))
    ax.contour(
        x,
        y,
        sampled,
        levels=levels,
        colors="#808080",
        linewidths=0.65,
        alpha=0.85,
        zorder=1,
    )


def legend_handles(
    linear_color: str,
    inaccessible_color: str,
) -> list[Line2D | Patch]:
    return [
        Line2D(
            [0],
            [0],
            color="#808080",
            linewidth=2,
            label="Elevation gradients",
        ),
        Line2D(
            [0],
            [0],
            color=linear_color,
            linewidth=3,
            label="Linear features",
        ),
        Patch(
            facecolor=inaccessible_color,
            edgecolor="none",
            alpha=0.65,
            label="Inaccessible areas",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="black",
            markerfacecolor="yellow",
            markeredgecolor="black",
            markersize=18,
            linestyle="None",
            label="Find location",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="red",
            markeredgecolor="black",
            markersize=13,
            linestyle="None",
            label="IPP",
        ),
    ]


def render_mat_file(
    mat_path: Path,
    output_folder: Path,
    condition: InitialCondition,
    meters_per_cell: float,
    contour_count: int,
    contour_stride: int,
    linear_color: str,
    linear_width: int,
    inaccessible_color: str,
    output_name: str | None = None,
) -> Path:
    elevation, linear_features, inaccessible = load_map_layers(mat_path)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{output_name or mat_path.stem}.png"

    rows, cols = elevation.shape
    ipp_cell, find_cell = initial_condition_cells(
        condition,
        (rows, cols),
        meters_per_cell,
    )

    fig, ax = plt.subplots(figsize=(14, 9), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.72, bottom=0.12, top=0.95)

    ax.set_facecolor("white")
    render_elevation_contours(ax, elevation, contour_count, contour_stride)
    render_mask_overlay(ax, inaccessible, inaccessible_color, alpha=0.65, zorder=2)
    render_mask_overlay(
        ax,
        dilate_mask(linear_features, linear_width),
        linear_color,
        alpha=0.95,
        zorder=3,
    )

    ax.scatter(
        [find_cell[0]],
        [find_cell[1]],
        marker="*",
        s=420,
        facecolor="yellow",
        edgecolor="black",
        linewidth=1.2,
        zorder=5,
    )
    ax.scatter(
        [ipp_cell[0]],
        [ipp_cell[1]],
        marker="o",
        s=220,
        facecolor="red",
        edgecolor="black",
        linewidth=1.2,
        zorder=5,
    )

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("longitude (x cells)", fontsize=20)
    ax.set_ylabel("latitude (y cells)", fontsize=20)
    ax.set_xticks(np.arange(0, cols + 1, 500))
    ax.set_yticks(np.arange(0, rows + 1, 500))
    ax.tick_params(axis="both", labelsize=18)
    ax.legend(
        handles=legend_handles(linear_color, inaccessible_color),
        loc="center left",
        bbox_to_anchor=(1.04, 0.5),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        fontsize=18,
    )

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def sorted_mat_files(input_folder: Path) -> list[Path]:
    return sorted(input_folder.glob("*.mat"), key=lambda path: extract_map_index(path))


def render_folder(
    input_folder: Path,
    output_folder: Path,
    conditions: dict[int, InitialCondition],
    indices: set[int] | None,
    meters_per_cell: float,
    contour_count: int,
    contour_stride: int,
    linear_color: str,
    linear_width: int,
    inaccessible_color: str,
    prefix_output_names: bool,
) -> int:
    mat_files = sorted_mat_files(input_folder)
    if indices is not None:
        mat_files = [
            mat_file for mat_file in mat_files if extract_map_index(mat_file) in indices
        ]
    if not mat_files:
        raise FileNotFoundError(f"No matching .mat files found in {input_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for mat_file in mat_files:
        map_index = extract_map_index(mat_file)
        if map_index not in conditions:
            raise KeyError(f"No initial condition row for incident {map_index}")
        output_name = (
            f"{input_folder.name}_{mat_file.stem}"
            if prefix_output_names
            else mat_file.stem
        )
        output_path = render_mat_file(
            mat_file,
            output_folder,
            conditions[map_index],
            meters_per_cell,
            contour_count,
            contour_stride,
            linear_color,
            linear_width,
            inaccessible_color,
            output_name,
        )
        rendered += 1
        print(f"Saved {output_path}", flush=True)
    return rendered


def main() -> None:
    args = parse_args()
    initial_conditions_path = (
        args.initial_conditions
        if args.initial_conditions is not None
        else args.input_dir / "InitialConditions.csv"
    )
    conditions = read_initial_conditions(initial_conditions_path)

    folder_names = list(dict.fromkeys(args.folders))
    selected_indices = set(args.indices) if args.indices is not None else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix_output_names = len(folder_names) > 1

    total = 0
    for folder_name in folder_names:
        input_folder = args.input_dir / folder_name
        if not input_folder.is_dir():
            raise FileNotFoundError(f"Missing SAR map folder: {input_folder}")
        total += render_folder(
            input_folder,
            args.output_dir,
            conditions,
            selected_indices,
            args.meters_per_cell,
            args.contour_count,
            args.contour_stride,
            args.linear_color,
            args.linear_width,
            args.inaccessible_color,
            prefix_output_names,
        )

    print(f"Rendered {total} image(s) into {args.output_dir}")


if __name__ == "__main__":
    main()
