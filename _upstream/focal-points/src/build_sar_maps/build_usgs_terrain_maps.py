import argparse
import csv
import io
import json
import math
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFile, ImageFont
from scipy.io import savemat
from scipy.ndimage import binary_dilation

try:
    from .map_image_common import Incident, latlon_offset_m, read_initial_conditions
except ImportError:
    from map_image_common import Incident, latlon_offset_m, read_initial_conditions


USGS_3DEP_EXPORT_IMAGE_URL = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/"
    "3DEPElevation/ImageServer/exportImage"
)
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
EARTH_RADIUS_M = 6_371_008.8
RETRYABLE_HTTP_STATUS_CODES = {400, 408, 425, 429, 500, 502, 503, 504}

COLORS = {
    "elevation_gradients": "#666666",
    "streams": "#0089bd",
    "riverbanks": "#8f6500",
    "roads": "#7a4f13",
    "railroads": "#2f7d32",
    "powerlines": "#7b1fa2",
    "lake_shorelines": "#c45100",
    "hiking_trails": "#1d1d1d",
    "river_interiors": "#0034cc",
    "lake_interiors": "#7657e8",
    "find": "#fff200",
    "ipp": "#ff1010",
}

LAYER_LABELS = {
    "elevation_gradients": "Elevation gradients",
    "streams": "Streams",
    "riverbanks": "Riverbanks",
    "roads": "Roads",
    "railroads": "Railroads",
    "powerlines": "Powerline easements",
    "lake_shorelines": "Lake shorelines",
    "hiking_trails": "Hiking trails",
    "river_interiors": "River interiors",
    "lake_interiors": "Lake interiors",
    "find": "Find location",
    "ipp": "IPP",
}

LINEAR_LAYERS = (
    "streams",
    "riverbanks",
    "roads",
    "railroads",
    "powerlines",
    "lake_shorelines",
    "hiking_trails",
)
AREA_LAYERS = ("river_interiors", "lake_interiors")
FEATURE_LEGEND_ORDER = (
    "elevation_gradients",
    *LINEAR_LAYERS,
    *AREA_LAYERS,
)
MARKER_LEGEND_ORDER = ("find", "ipp")
LINE_RENDER_ALPHA = 245
AREA_RENDER_ALPHA = 205
ImageFile.LOAD_TRUNCATED_IMAGES = True
AXIS_LABEL_X = "longitude (x)"
AXIS_LABEL_Y = "latitude (y)"
AXIS_LABELS_PER_SIDE = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build 3000x3000 SAR-style terrain maps centered on IPP coordinates "
            "using USGS 3DEP DEM data plus OpenStreetMap vector layers."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data") / "SAR_maps" / "InitialConditions.csv",
        help="CSV with incident_index, IPP_lat/lon, and find_lat/lon.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "SAR_maps" / "usgs_terrain",
        help="Output root for clean base images, matrices, and metadata.",
    )
    parser.add_argument(
        "--v1-dir",
        type=Path,
        default=Path("data") / "SAR_maps" / "v1" / "usgs_terrain",
        help="Output directory for IPP-only terrain images.",
    )
    parser.add_argument(
        "--v2-dir",
        type=Path,
        default=Path("data") / "SAR_maps" / "v2" / "usgs_terrain",
        help="Output directory for IPP plus find-location terrain images.",
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        default=None,
        help=(
            'Optional incident_index values to process, e.g. "1-5", "7", '
            'or "1-5, 7, 20-30".'
        ),
    )
    parser.add_argument(
        "--failed-from-metadata",
        action="store_true",
        help=(
            "Process only incidents marked failed in the existing metadata.csv "
            "under --output-dir."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=3000,
        help="Square output size in pixels/cells.",
    )
    parser.add_argument(
        "--meters-per-cell",
        type=float,
        default=None,
        help=(
            "Fixed ground resolution. If omitted, each incident is auto-scaled "
            "from the IPP-to-find offset."
        ),
    )
    parser.add_argument(
        "--min-ground-width-m",
        type=float,
        default=5000.0,
        help=(
            "Minimum auto-scaled map width in meters. The default gives close "
            "incidents a 5 km x 5 km map."
        ),
    )
    parser.add_argument(
        "--point-padding-m",
        type=float,
        default=800.0,
        help="Extra auto-scale padding around the find location, in meters.",
    )
    parser.add_argument(
        "--usable-half-width-fraction",
        type=float,
        default=0.88,
        help=(
            "Auto-scale fraction of the center-to-edge distance that may be "
            "used by the IPP-to-find offset plus padding."
        ),
    )
    parser.add_argument(
        "--slope-threshold-deg",
        type=float,
        default=35.0,
        help="Slope threshold used for the BWInac matrix.",
    )
    parser.add_argument(
        "--contour-count",
        type=int,
        default=42,
        help="Number of gray elevation contour levels to draw.",
    )
    parser.add_argument(
        "--contour-stride",
        type=int,
        default=4,
        help="Downsampling stride for contour generation.",
    )
    parser.add_argument(
        "--line-width-px",
        type=int,
        default=5,
        help="Default rendered line width for linear GIS features.",
    )
    parser.add_argument(
        "--include-powerlines",
        action="store_true",
        help="Include OpenStreetMap power=line/minor_line features in maps and legends.",
    )
    parser.add_argument(
        "--matrix-line-width-px",
        type=int,
        default=2,
        help="Dilation radius used when storing thin vector features as binary matrices.",
    )
    parser.add_argument(
        "--marker-radius-px",
        type=int,
        default=34,
        help="IPP red-dot radius in pixels.",
    )
    parser.add_argument(
        "--star-radius-px",
        type=int,
        default=52,
        help="Find-location yellow star outer radius in pixels.",
    )
    parser.add_argument(
        "--no-grid-legend",
        action="store_true",
        help="Save plain square maps without cell-grid axes or legend.",
    )
    parser.add_argument(
        "--legend-min-coverage-pct",
        type=float,
        default=0.1,
        help=(
            "Minimum percent of map pixels a feature layer must cover to appear "
            "in the legend. The layer is still rendered on the map."
        ),
    )
    parser.add_argument(
        "--grid-step-m",
        type=float,
        default=500.0,
        help=(
            "Base ground increment for axis labels. The actual label spacing "
            "is a multiple of this value. Default is 500 m."
        ),
    )
    parser.add_argument(
        "--scale-bar-m",
        type=float,
        default=500.0,
        help=(
            "Deprecated. The scale bar follows the axis-label spacing so its "
            "length matches one jump between neighboring axis labels."
        ),
    )
    parser.add_argument(
        "--decorate-existing",
        action="store_true",
        help=(
            "Add grid/legend frames to existing base/v1/v2 PNGs without "
            "downloading USGS or OSM data. Intended for local redraws."
        ),
    )
    parser.add_argument(
        "--usgs-url",
        default=USGS_3DEP_EXPORT_IMAGE_URL,
        help="USGS 3DEP ArcGIS ImageServer exportImage endpoint.",
    )
    parser.add_argument(
        "--overpass-url",
        default=OVERPASS_URL,
        help="Overpass API endpoint for OSM vector layers.",
    )
    parser.add_argument(
        "--http-timeout-s",
        type=float,
        default=180.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--overpass-timeout-s",
        type=int,
        default=180,
        help="Timeout embedded in the Overpass query.",
    )
    parser.add_argument(
        "--rate-limit-s",
        type=float,
        default=1.0,
        help="Sleep time between incidents.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum attempts for each USGS/Overpass HTTP request.",
    )
    parser.add_argument(
        "--retry-delay-s",
        type=float,
        default=20.0,
        help="Initial sleep before retrying a failed HTTP request.",
    )
    parser.add_argument(
        "--skip-osm",
        action="store_true",
        help="Only fetch/render the USGS elevation layer and markers.",
    )
    parser.add_argument(
        "--no-mat",
        action="store_true",
        help="Do not write MATLAB .mat matrices.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned requests without downloading data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with remaining incidents when one incident fails.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if args.meters_per_cell is not None and args.meters_per_cell <= 0:
        raise ValueError("--meters-per-cell must be positive")
    if args.min_ground_width_m <= 0:
        raise ValueError("--min-ground-width-m must be positive")
    if args.point_padding_m < 0:
        raise ValueError("--point-padding-m must be non-negative")
    if args.usable_half_width_fraction <= 0 or args.usable_half_width_fraction >= 1:
        raise ValueError("--usable-half-width-fraction must be in the range (0, 1)")
    if args.slope_threshold_deg <= 0 or args.slope_threshold_deg >= 90:
        raise ValueError("--slope-threshold-deg must be in the range (0, 90)")
    if args.http_timeout_s <= 0:
        raise ValueError("--http-timeout-s must be positive")
    if args.overpass_timeout_s <= 0:
        raise ValueError("--overpass-timeout-s must be positive")
    if args.rate_limit_s < 0:
        raise ValueError("--rate-limit-s must be non-negative")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    if args.retry_delay_s < 0:
        raise ValueError("--retry-delay-s must be non-negative")
    if args.grid_step_m <= 0:
        raise ValueError("--grid-step-m must be positive")
    if args.scale_bar_m <= 0:
        raise ValueError("--scale-bar-m must be positive")
    if args.legend_min_coverage_pct < 0 or args.legend_min_coverage_pct > 100:
        raise ValueError("--legend-min-coverage-pct must be in the range [0, 100]")


def failed_indices_from_metadata(metadata_path: Path) -> set[int]:
    if not metadata_path.exists():
        raise ValueError(f"Metadata file not found: {metadata_path}")

    failed_indices = set()
    with metadata_path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("status") != "failed":
                continue
            incident_index = row.get("incident_index")
            if not incident_index:
                continue
            failed_indices.add(int(incident_index))
    return failed_indices


def parse_positive_index(value: str) -> int:
    try:
        index = int(value)
    except ValueError as exc:
        raise ValueError(f"Incident index must be a positive integer, got {value!r}") from exc

    if index <= 0:
        raise ValueError(f"Incident index must be positive, got {value!r}")
    return index


def parse_index_specs(specs: list[str]) -> set[int]:
    indices: set[int] = set()
    normalized = " ".join(specs)
    for chunk in normalized.replace(",", " ").split():
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = parse_positive_index(start_text.strip())
            end = parse_positive_index(end_text.strip())
            if end < start:
                raise ValueError(f"Incident index range must be ascending, got {chunk!r}")
            indices.update(range(start, end + 1))
        else:
            indices.add(parse_positive_index(chunk))

    if not indices:
        raise ValueError("--indices did not contain any incident ids")
    return indices


def selected_incidents(args: argparse.Namespace) -> list[Incident]:
    incidents = sorted(
        read_initial_conditions(args.csv),
        key=lambda incident: incident.incident_index,
    )
    wanted = None
    if args.failed_from_metadata:
        wanted = failed_indices_from_metadata(args.output_dir / "metadata.csv")
        if not wanted:
            print(f"No failed incidents found in {args.output_dir / 'metadata.csv'}")
            return []

    if args.indices is not None:
        explicit_indices = parse_index_specs(args.indices)
        wanted = explicit_indices if wanted is None else wanted & explicit_indices

    if wanted is None:
        return incidents

    selected = [incident for incident in incidents if incident.incident_index in wanted]
    missing = wanted - {incident.incident_index for incident in selected}
    if missing:
        raise ValueError(f"Requested incident indices not found: {sorted(missing)}")
    return selected


def bbox_around_ipp(
    ipp_lat: float,
    ipp_lon: float,
    image_size: int,
    meters_per_cell: float,
) -> tuple[float, float, float, float]:
    half_width_m = image_size * meters_per_cell / 2
    lat_delta = math.degrees(half_width_m / EARTH_RADIUS_M)
    lon_delta = math.degrees(
        half_width_m / (EARTH_RADIUS_M * math.cos(math.radians(ipp_lat)))
    )
    west = ipp_lon - lon_delta
    south = ipp_lat - lat_delta
    east = ipp_lon + lon_delta
    north = ipp_lat + lat_delta
    return west, south, east, north


def meters_per_cell_for_incident(
    find_east_m: float,
    find_north_m: float,
    image_size: int,
    args: argparse.Namespace,
) -> float:
    if args.meters_per_cell is not None:
        return float(args.meters_per_cell)

    max_offset_m = max(abs(find_east_m), abs(find_north_m))
    half_width_m = max(
        args.min_ground_width_m / 2,
        (max_offset_m + args.point_padding_m) / args.usable_half_width_fraction,
    )
    axis_snap_m = args.grid_step_m * AXIS_LABELS_PER_SIDE
    half_width_m = math.ceil(half_width_m / axis_snap_m) * axis_snap_m
    return (2 * half_width_m) / image_size


def latlon_to_cell(
    lat: float,
    lon: float,
    bbox: tuple[float, float, float, float],
    image_size: int,
) -> tuple[float, float]:
    west, south, east, north = bbox
    x = (lon - west) / (east - west) * image_size
    y = (lat - south) / (north - south) * image_size
    return x, y


def cell_to_pil_point(x: float, y: float, image_size: int) -> tuple[int, int]:
    return round(x), round(image_size - 1 - y)


def geom_to_pil_points(
    geometry: list[dict[str, float]],
    bbox: tuple[float, float, float, float],
    image_size: int,
) -> list[tuple[int, int]]:
    points = []
    for point in geometry:
        x, y = latlon_to_cell(point["lat"], point["lon"], bbox, image_size)
        points.append(cell_to_pil_point(x, y, image_size))
    return points


def retryable_url_error(error: BaseException) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code in RETRYABLE_HTTP_STATUS_CODES
    return isinstance(
        error,
        (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            OSError,
        ),
    )


def retry_error_text(error: BaseException) -> str:
    if isinstance(error, urllib.error.HTTPError):
        return f"HTTP {error.code}: {error.reason}"
    return f"{type(error).__name__}: {error}"


def urlopen_bytes(
    url: str,
    data: bytes | None,
    timeout_s: float,
    headers: dict[str, str] | None = None,
    max_attempts: int = 1,
    retry_delay_s: float = 0.0,
    request_label: str = "request",
) -> bytes:
    request = urllib.request.Request(url, data=data, headers=headers or {})
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return response.read()
        except Exception as error:
            if attempt >= max_attempts or not retryable_url_error(error):
                raise
            delay_s = retry_delay_s * (2 ** (attempt - 1))
            print(
                f"retry: {request_label} attempt {attempt}/{max_attempts} "
                f"failed ({retry_error_text(error)}); "
                f"sleeping {delay_s:g}s"
            )
            if delay_s:
                time.sleep(delay_s)
    raise RuntimeError(f"{request_label} failed without returning data")


def fetch_json(
    url: str,
    params: dict[str, str],
    timeout_s: float,
    max_attempts: int = 1,
    retry_delay_s: float = 0.0,
    request_label: str = "JSON request",
) -> dict[str, Any]:
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    body = urlopen_bytes(
        full_url,
        None,
        timeout_s,
        max_attempts=max_attempts,
        retry_delay_s=retry_delay_s,
        request_label=request_label,
    )
    payload = json.loads(body.decode("utf-8"))
    if "error" in payload:
        raise RuntimeError(f"Request failed: {payload['error']}")
    return payload


def fetch_usgs_dem(
    bbox: tuple[float, float, float, float],
    image_size: int,
    args: argparse.Namespace,
) -> np.ndarray:
    west, south, east, north = bbox
    params = {
        "f": "json",
        "bbox": f"{west},{south},{east},{north}",
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": f"{image_size},{image_size}",
        "format": "tiff",
        "pixelType": "F32",
        "interpolation": "RSP_BilinearInterpolation",
        "noDataInterpretation": "esriNoDataMatchAny",
        "returnSquarePixels": "false",
    }
    payload = fetch_json(
        args.usgs_url,
        params,
        args.http_timeout_s,
        max_attempts=args.max_attempts,
        retry_delay_s=args.retry_delay_s,
        request_label="USGS exportImage",
    )
    image_url = payload.get("href")
    if not image_url:
        raise RuntimeError(f"USGS response did not include an image href: {payload}")

    image_bytes = urlopen_bytes(
        str(image_url),
        None,
        args.http_timeout_s,
        max_attempts=args.max_attempts,
        retry_delay_s=args.retry_delay_s,
        request_label="USGS image download",
    )
    image = Image.open(io.BytesIO(image_bytes))
    elevation = np.asarray(image).astype(np.float32)
    if elevation.ndim == 3:
        elevation = elevation[:, :, 0]

    elevation = np.flipud(elevation)
    elevation[~np.isfinite(elevation)] = np.nan
    elevation[elevation < -10000] = np.nan
    return elevation


def overpass_query(
    bbox: tuple[float, float, float, float],
    timeout_s: int,
    include_powerlines: bool,
) -> str:
    west, south, east, north = bbox
    bounds = f"({south:.8f},{west:.8f},{north:.8f},{east:.8f})"
    powerline_query = (
        f'  way["power"~"line|minor_line"]{bounds};\n'
        if include_powerlines
        else ""
    )
    return f"""
[out:json][timeout:{timeout_s}];
(
  way["waterway"~"stream|river|canal|ditch|drain"]{bounds};
  way["highway"]{bounds};
  way["railway"~"rail|narrow_gauge|light_rail|tram"]{bounds};
{powerline_query}  way["natural"="water"]{bounds};
  relation["natural"="water"]{bounds};
  way["waterway"="riverbank"]{bounds};
  relation["waterway"="riverbank"]{bounds};
  way["landuse"="reservoir"]{bounds};
  relation["landuse"="reservoir"]{bounds};
);
out body geom;
"""


def fetch_osm_layers(
    bbox: tuple[float, float, float, float],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    query = overpass_query(bbox, args.overpass_timeout_s, args.include_powerlines)
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    body = urlopen_bytes(
        args.overpass_url,
        data,
        args.http_timeout_s,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "focal-points-usgs-terrain-map-builder/1.0",
        },
        max_attempts=args.max_attempts,
        retry_delay_s=args.retry_delay_s,
        request_label="Overpass OSM",
    )
    payload = json.loads(body.decode("utf-8"))
    if "remark" in payload:
        print(f"Overpass remark: {payload['remark']}")
    return list(payload.get("elements", []))


def way_geometry(element: dict[str, Any]) -> list[dict[str, float]]:
    geometry = element.get("geometry")
    if isinstance(geometry, list):
        return geometry
    return []


def relation_member_geometries(element: dict[str, Any]) -> list[list[dict[str, float]]]:
    geometries = []
    for member in element.get("members", []):
        geometry = member.get("geometry")
        if isinstance(geometry, list):
            geometries.append(geometry)
    return geometries


def is_closed_geometry(geometry: list[dict[str, float]]) -> bool:
    if len(geometry) < 4:
        return False
    first = geometry[0]
    last = geometry[-1]
    return first.get("lat") == last.get("lat") and first.get("lon") == last.get("lon")


def is_hiking_trail(tags: dict[str, str]) -> bool:
    highway = tags.get("highway")
    return highway in {
        "path",
        "footway",
        "bridleway",
        "steps",
        "pedestrian",
        "cycleway",
    }


def is_road(tags: dict[str, str]) -> bool:
    return "highway" in tags and not is_hiking_trail(tags)


def is_railroad(tags: dict[str, str]) -> bool:
    return tags.get("railway") in {"rail", "narrow_gauge", "light_rail", "tram"}


def is_powerline(tags: dict[str, str]) -> bool:
    return tags.get("power") in {"line", "minor_line"}


def is_water_line(tags: dict[str, str]) -> bool:
    return tags.get("waterway") in {"stream", "river", "canal", "ditch", "drain"}


def is_river_area(tags: dict[str, str]) -> bool:
    return tags.get("waterway") == "riverbank" or tags.get("water") == "river"


def is_water_area(tags: dict[str, str]) -> bool:
    return (
        tags.get("natural") == "water"
        or tags.get("landuse") == "reservoir"
        or tags.get("waterway") == "riverbank"
    )


def empty_layer_images(image_size: int) -> dict[str, Image.Image]:
    return {
        layer: Image.new("L", (image_size, image_size), 0)
        for layer in (*LINEAR_LAYERS, *AREA_LAYERS)
    }


def draw_line(
    layer_images: dict[str, Image.Image],
    layer: str,
    points: list[tuple[int, int]],
    width: int,
) -> None:
    if len(points) < 2:
        return
    ImageDraw.Draw(layer_images[layer]).line(
        points, fill=255, width=width, joint="curve"
    )


def draw_polygon(
    layer_images: dict[str, Image.Image],
    fill_layer: str,
    outline_layer: str,
    points: list[tuple[int, int]],
    line_width: int,
) -> None:
    if len(points) < 3:
        return
    ImageDraw.Draw(layer_images[fill_layer]).polygon(points, fill=255)
    ImageDraw.Draw(layer_images[outline_layer]).line(
        points,
        fill=255,
        width=line_width,
        joint="curve",
    )


def rasterize_osm_layers(
    elements: list[dict[str, Any]],
    bbox: tuple[float, float, float, float],
    image_size: int,
    line_width: int,
) -> dict[str, np.ndarray]:
    layer_images = empty_layer_images(image_size)
    for element in elements:
        tags = element.get("tags") or {}
        geometries: list[list[dict[str, float]]]
        if element.get("type") == "relation":
            geometries = relation_member_geometries(element)
        else:
            geometry = way_geometry(element)
            geometries = [geometry] if geometry else []

        for geometry in geometries:
            if len(geometry) < 2:
                continue
            points = geom_to_pil_points(geometry, bbox, image_size)
            if is_water_area(tags) and is_closed_geometry(geometry):
                if is_river_area(tags):
                    draw_polygon(
                        layer_images,
                        "river_interiors",
                        "riverbanks",
                        points,
                        line_width,
                    )
                else:
                    draw_polygon(
                        layer_images,
                        "lake_interiors",
                        "lake_shorelines",
                        points,
                        line_width,
                    )
            elif is_water_line(tags):
                draw_line(layer_images, "streams", points, line_width)
            elif is_road(tags):
                draw_line(layer_images, "roads", points, line_width)
            elif is_hiking_trail(tags):
                draw_line(layer_images, "hiking_trails", points, line_width)
            elif is_railroad(tags):
                draw_line(layer_images, "railroads", points, line_width)
            elif is_powerline(tags):
                draw_line(layer_images, "powerlines", points, line_width)

    return {
        layer: np.flipud(np.asarray(image, dtype=np.uint8) > 0)
        for layer, image in layer_images.items()
    }


def blank_masks(image_size: int) -> dict[str, np.ndarray]:
    return {
        layer: np.zeros((image_size, image_size), dtype=bool)
        for layer in (*LINEAR_LAYERS, *AREA_LAYERS)
    }


def slope_mask(
    elevation: np.ndarray,
    meters_per_cell: float,
    slope_threshold_deg: float,
) -> np.ndarray:
    filled = elevation.copy()
    if np.isnan(filled).any():
        finite = filled[np.isfinite(filled)]
        fill_value = float(np.nanmedian(finite)) if finite.size else 0.0
        filled[~np.isfinite(filled)] = fill_value

    grad_y, grad_x = np.gradient(filled, meters_per_cell, meters_per_cell)
    slope_deg = np.degrees(np.arctan(np.hypot(grad_x, grad_y)))
    return slope_deg >= slope_threshold_deg


def matrix_line_masks(
    masks: dict[str, np.ndarray],
    dilation_radius: int,
) -> dict[str, np.ndarray]:
    if dilation_radius <= 0:
        return masks
    out = {}
    for layer, mask in masks.items():
        if layer in LINEAR_LAYERS and mask.any():
            out[layer] = binary_dilation(mask, iterations=dilation_radius)
        else:
            out[layer] = mask
    return out


def render_contours(
    elevation: np.ndarray,
    image_size: int,
    contour_count: int,
    contour_stride: int,
) -> Image.Image:
    stride = max(1, contour_stride)
    sampled = elevation[::stride, ::stride]
    finite = sampled[np.isfinite(sampled)]
    layer = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
    if finite.size == 0:
        return layer

    low, high = np.nanpercentile(finite, [2, 98])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
    if high <= low:
        return layer

    dpi = 100
    fig = plt.figure(figsize=(image_size / dpi, image_size / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, image_size)
    ax.set_ylim(0, image_size)
    ax.axis("off")
    x = np.arange(0, image_size, stride)
    y = np.arange(0, image_size, stride)
    levels = np.linspace(low, high, max(2, contour_count))
    ax.contour(
        x,
        y,
        sampled,
        levels=levels,
        colors=COLORS["elevation_gradients"],
        linewidths=1.0,
        alpha=0.88,
    )

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, transparent=True)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGBA")


def paste_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: str,
    alpha: int = 255,
) -> None:
    if not mask.any():
        return
    mask_image = Image.fromarray(np.flipud(mask).astype(np.uint8) * alpha, mode="L")
    overlay = Image.new("RGBA", image.size, color)
    overlay.putalpha(mask_image)
    image.alpha_composite(overlay)


def rgb_from_hex(color: str) -> tuple[int, int, int]:
    value = color.removeprefix("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def blend_on_white(color: str, alpha: int) -> tuple[int, int, int]:
    rgb = rgb_from_hex(color)
    return tuple(
        round(channel * alpha / 255 + 255 * (1 - alpha / 255)) for channel in rgb
    )


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        ("DejaVuSans-Bold.ttf", "arialbd.ttf")
        if bold
        else ("DejaVuSans.ttf", "arial.ttf")
    )
    search_paths = [
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/truetype/msttcorefonts"),
        Path("C:/Windows/Fonts"),
        Path("/mnt/c/Windows/Fonts"),
    ]
    for directory in search_paths:
        for name in names:
            path = directory / name
            if path.exists():
                return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def star_points(
    center_x: float,
    center_y: float,
    outer_radius: float,
    inner_radius: float,
    points: int = 5,
) -> list[tuple[float, float]]:
    result = []
    rotation = -math.pi / 2
    for index in range(points * 2):
        radius = outer_radius if index % 2 == 0 else inner_radius
        angle = rotation + index * math.pi / points
        result.append(
            (
                center_x + radius * math.cos(angle),
                center_y + radius * math.sin(angle),
            )
        )
    return result


def draw_ipp_marker(
    image: Image.Image,
    ipp_cell: tuple[float, float],
    args: argparse.Namespace,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    ipp_x, ipp_y = cell_to_pil_point(ipp_cell[0], ipp_cell[1], args.image_size)
    radius = args.marker_radius_px
    draw.ellipse(
        (ipp_x - radius, ipp_y - radius, ipp_x + radius, ipp_y + radius),
        fill=COLORS["ipp"],
        outline=(25, 25, 25, 255),
        width=4,
    )


def draw_find_marker(
    image: Image.Image,
    find_cell: tuple[float, float],
    args: argparse.Namespace,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    find_x, find_y = cell_to_pil_point(find_cell[0], find_cell[1], args.image_size)
    star = star_points(
        find_x,
        find_y,
        args.star_radius_px,
        args.star_radius_px * 0.45,
    )
    draw.polygon(star, fill=COLORS["find"], outline=(25, 25, 25, 255))
    draw.line(star + [star[0]], fill=(25, 25, 25, 255), width=3)


def legend_min_pixels(args: argparse.Namespace) -> int:
    return max(
        1,
        math.ceil(
            args.image_size * args.image_size * args.legend_min_coverage_pct / 100
        ),
    )


def mask_meets_legend_threshold(mask: np.ndarray, args: argparse.Namespace) -> bool:
    return bool(np.count_nonzero(mask) >= legend_min_pixels(args))


def layer_presence_from_masks(
    elevation: np.ndarray,
    masks: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, bool]:
    presence = {
        "elevation_gradients": bool(np.isfinite(elevation).any()),
    }
    for layer in (*LINEAR_LAYERS, *AREA_LAYERS):
        if layer == "powerlines" and not args.include_powerlines:
            presence[layer] = False
            continue
        presence[layer] = mask_meets_legend_threshold(masks[layer], args)
    return presence


def color_present(
    pixels: np.ndarray,
    target: tuple[int, int, int],
    threshold: int,
    min_pixels: int,
) -> bool:
    target_array = np.array(target, dtype=np.int16)
    diff = pixels.astype(np.int16) - target_array
    distance_sq = np.sum(diff * diff, axis=2)
    return bool(np.count_nonzero(distance_sq <= threshold * threshold) >= min_pixels)


def layer_presence_from_rendered_image(
    image: Image.Image, args: argparse.Namespace
) -> dict[str, bool]:
    pixels = np.asarray(image.convert("RGB"))
    min_pixels = legend_min_pixels(args)
    gray_delta = pixels.max(axis=2) - pixels.min(axis=2)
    gray_value = pixels.mean(axis=2)
    presence = {
        "elevation_gradients": bool(
            np.count_nonzero(
                (gray_delta <= 10) & (gray_value >= 80) & (gray_value <= 225)
            )
            >= 200
        )
    }

    for layer in LINEAR_LAYERS:
        presence[layer] = color_present(
            pixels,
            blend_on_white(COLORS[layer], LINE_RENDER_ALPHA),
            threshold=48,
            min_pixels=max(60, min_pixels),
        )
    for layer in AREA_LAYERS:
        presence[layer] = color_present(
            pixels,
            blend_on_white(COLORS[layer], AREA_RENDER_ALPHA),
            threshold=58,
            min_pixels=max(120, min_pixels),
        )
    return presence


def legend_layers(
    layer_presence: dict[str, bool],
    markers: tuple[str, ...],
) -> list[str]:
    layers = [
        layer for layer in FEATURE_LEGEND_ORDER if layer_presence.get(layer, False)
    ]
    marker_set = set(markers)
    layers.extend(layer for layer in MARKER_LEGEND_ORDER if layer in marker_set)
    return layers


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str | tuple[int, int, int] = "black",
) -> None:
    width, height = text_size(draw, text, font)
    draw.text(
        (center[0] - width / 2, center[1] - height / 2),
        text,
        font=font,
        fill=fill,
    )


def draw_rotated_label(
    canvas: Image.Image,
    text: str,
    center: tuple[int, int],
    font: ImageFont.ImageFont,
) -> None:
    probe = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    probe_draw = ImageDraw.Draw(probe)
    width, height = text_size(probe_draw, text, font)
    label = Image.new("RGBA", (width + 20, height + 20), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((10, 10), text, font=font, fill=(20, 20, 20, 255))
    label = label.rotate(90, expand=True)
    canvas.alpha_composite(
        label,
        (round(center[0] - label.width / 2), round(center[1] - label.height / 2)),
    )


def grid_tick_positions(
    map_size: int,
    meters_per_cell: float,
    args: argparse.Namespace,
) -> list[tuple[float, float]]:
    label_step_m = grid_label_step_m(map_size, meters_per_cell, args)
    step_px = label_step_m / meters_per_cell

    center = map_size / 2
    return [
        (center + step * step_px, step * label_step_m)
        for step in range(-AXIS_LABELS_PER_SIDE, AXIS_LABELS_PER_SIDE + 1)
    ]


def grid_label_step_m(
    map_size: int,
    meters_per_cell: float,
    args: argparse.Namespace,
) -> float:
    half_width_m = map_size * meters_per_cell / 2
    max_step_m = half_width_m / AXIS_LABELS_PER_SIDE
    step_count = max(1, math.ceil(max_step_m / args.grid_step_m - 1e-9))
    return step_count * args.grid_step_m


def format_grid_number(value_m: float) -> str:
    rounded = round(value_m)
    if abs(rounded) == 0:
        rounded = 0
    return str(rounded)


def draw_grid_numbers(
    canvas: Image.Image,
    map_origin: tuple[int, int],
    map_size: int,
    meters_per_cell: float,
    args: argparse.Namespace,
) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    left, top = map_origin
    right = left + map_size
    bottom = top + map_size
    ticks = grid_tick_positions(map_size, meters_per_cell, args)
    tick_length = max(22, round(map_size * 0.008))
    font_size = max(46, round(map_size * 0.023))
    font = load_font(font_size)
    fill = (20, 20, 20, 255)
    tick_fill = (35, 35, 35, 255)

    for position, value_m in ticks:
        label = format_grid_number(value_m)
        x = round(left + position)
        y = round(bottom - position)

        if left <= x <= right:
            draw.line((x, bottom, x, bottom + tick_length), fill=tick_fill, width=3)
            draw_centered_text(
                draw,
                (x, bottom + tick_length + round(font_size * 0.58)),
                label,
                font,
                fill,
            )

        if top <= y <= bottom:
            draw.line((left - tick_length, y, left, y), fill=tick_fill, width=3)
            label_width, label_height = text_size(draw, label, font)
            draw.text(
                (left - tick_length - 8 - label_width, y - label_height / 2),
                label,
                font=font,
                fill=fill,
            )

    draw.rectangle((left, top, right, bottom), outline=(20, 20, 20, 255), width=4)


def draw_axis_labels(
    canvas: Image.Image,
    map_origin: tuple[int, int],
    map_size: int,
    left_margin: int,
    bottom_margin: int,
) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = load_font(max(56, round(map_size * 0.032)))
    left, top = map_origin
    bottom = top + map_size
    fill = (20, 20, 20, 255)

    draw_centered_text(
        draw,
        (left + map_size // 2, bottom + round(bottom_margin * 0.58)),
        AXIS_LABEL_X,
        font,
        fill,
    )
    draw_rotated_label(
        canvas,
        AXIS_LABEL_Y,
        (round(left_margin * 0.45), top + map_size // 2),
        font,
    )


def scale_bar_label(length_m: float) -> str:
    if length_m < 1000:
        return f"{length_m:g} m"
    return f"{length_m / 1000:g} km"


def draw_legend_sample(
    draw: ImageDraw.ImageDraw,
    layer: str,
    center: tuple[int, int],
    font_size: int,
) -> None:
    cx, cy = center
    if layer == "find":
        radius = font_size * 0.48
        star = star_points(cx, cy, radius, radius * 0.45)
        draw.polygon(star, fill=COLORS["find"], outline=(25, 25, 25, 255))
        draw.line(
            star + [star[0]],
            fill=(25, 25, 25, 255),
            width=max(2, font_size // 22),
        )
        return
    if layer == "ipp":
        radius = font_size * 0.38
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=COLORS["ipp"],
            outline=(25, 25, 25, 255),
            width=max(2, font_size // 18),
        )
        return
    if layer in AREA_LAYERS:
        width = font_size * 1.18
        height = font_size * 0.52
        draw.rectangle(
            (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2),
            fill=COLORS[layer],
            outline=(25, 25, 25, 255),
            width=max(2, font_size // 24),
        )
        return

    line_width = max(6, font_size // 6)
    draw.line(
        (cx - font_size * 0.62, cy, cx + font_size * 0.62, cy),
        fill=COLORS[layer],
        width=line_width,
    )


def draw_legend(
    canvas: Image.Image,
    map_origin: tuple[int, int],
    map_size: int,
    layer_presence: dict[str, bool],
    markers: tuple[str, ...],
    meters_per_cell: float,
    args: argparse.Namespace,
) -> None:
    items = legend_layers(layer_presence, markers)
    if not items:
        return

    draw = ImageDraw.Draw(canvas, "RGBA")
    font_size = max(32, round(map_size * 0.026))
    font = load_font(font_size)
    row_height = round(font_size * 1.32)
    box_padding_x = round(font_size * 0.85)
    box_padding_y = round(font_size * 0.7)
    sample_width = round(font_size * 1.7)
    gap = round(font_size * 0.65)
    scale_bar_m = grid_label_step_m(map_size, meters_per_cell, args)
    scale_px = max(1, round(scale_bar_m / meters_per_cell))

    max_label_width = max(
        text_size(draw, LAYER_LABELS[item], font)[0] for item in items
    )
    box_width = max(
        box_padding_x * 2 + sample_width + gap + max_label_width,
        box_padding_x * 2 + scale_px,
    )
    box_height = box_padding_y * 2 + row_height * len(items)
    left = map_origin[0] + map_size + round(map_size * 0.08)
    top = map_origin[1] + round(map_size * 0.16)
    right = left + box_width
    bottom = top + box_height

    bar_x0 = round(left + (box_width - scale_px) / 2)
    bar_x1 = bar_x0 + scale_px
    bar_y = top - round(font_size * 1.05)
    cap = max(12, round(font_size * 0.26))
    label = scale_bar_label(scale_bar_m)
    label_width, label_height = text_size(draw, label, font)
    draw.line((bar_x0, bar_y, bar_x1, bar_y), fill=(20, 20, 20, 255), width=5)
    draw.line(
        (bar_x0, bar_y - cap, bar_x0, bar_y + cap), fill=(20, 20, 20, 255), width=5
    )
    draw.line(
        (bar_x1, bar_y - cap, bar_x1, bar_y + cap), fill=(20, 20, 20, 255), width=5
    )
    draw.text(
        ((bar_x0 + bar_x1 - label_width) / 2, bar_y - cap - label_height - 8),
        label,
        font=font,
        fill=(20, 20, 20, 255),
    )

    draw.rectangle(
        (left, top, right, bottom),
        fill=(255, 255, 255, 238),
        outline=(35, 35, 35, 255),
        width=3,
    )
    for index, item in enumerate(items):
        row_center_y = top + box_padding_y + row_height * index + row_height // 2
        sample_center_x = left + box_padding_x + sample_width // 2
        draw_legend_sample(draw, item, (sample_center_x, row_center_y), font_size)
        draw.text(
            (
                left + box_padding_x + sample_width + gap,
                row_center_y - font_size * 0.52,
            ),
            LAYER_LABELS[item],
            font=font,
            fill=(20, 20, 20, 255),
        )


def decorate_map_frame(
    map_image: Image.Image,
    layer_presence: dict[str, bool],
    markers: tuple[str, ...],
    meters_per_cell: float,
    args: argparse.Namespace,
) -> Image.Image:
    if args.no_grid_legend:
        return map_image

    map_image = map_image.convert("RGBA")
    map_width, map_height = map_image.size
    if map_width != map_height:
        raise ValueError(f"Expected a square map image, got {map_width}x{map_height}")

    left_margin = max(260, round(map_width * 0.085))
    top_margin = max(130, round(map_width * 0.045))
    bottom_margin = max(220, round(map_width * 0.073))
    right_margin = max(1500, round(map_width * 0.55))
    canvas = Image.new(
        "RGBA",
        (
            left_margin + map_width + right_margin,
            top_margin + map_height + bottom_margin,
        ),
        (255, 255, 255, 255),
    )
    map_origin = (left_margin, top_margin)
    canvas.alpha_composite(map_image, map_origin)
    draw_grid_numbers(canvas, map_origin, map_width, meters_per_cell, args)
    draw_axis_labels(canvas, map_origin, map_width, left_margin, bottom_margin)
    draw_legend(
        canvas,
        map_origin,
        map_width,
        layer_presence,
        markers,
        meters_per_cell,
        args,
    )
    return canvas


def render_map(
    elevation: np.ndarray,
    masks: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Image.Image:
    image = Image.new("RGBA", (args.image_size, args.image_size), (255, 255, 255, 255))
    image.alpha_composite(
        render_contours(
            elevation,
            args.image_size,
            args.contour_count,
            args.contour_stride,
        )
    )

    for layer in ("lake_interiors", "river_interiors"):
        paste_mask(image, masks[layer], COLORS[layer], alpha=AREA_RENDER_ALPHA)
    for layer in (
        "streams",
        "riverbanks",
        "roads",
        "railroads",
        "powerlines",
        "lake_shorelines",
        "hiking_trails",
    ):
        if layer == "powerlines" and not args.include_powerlines:
            continue
        paste_mask(image, masks[layer], COLORS[layer], alpha=LINE_RENDER_ALPHA)

    return image


def save_image_versions(
    base_image: Image.Image,
    layer_presence: dict[str, bool],
    ipp_cell: tuple[float, float],
    find_cell: tuple[float, float],
    meters_per_cell: float,
    base_path: Path,
    v1_path: Path,
    v2_path: Path,
    args: argparse.Namespace,
) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    v1_path.parent.mkdir(parents=True, exist_ok=True)
    v2_path.parent.mkdir(parents=True, exist_ok=True)

    decorate_map_frame(
        base_image,
        layer_presence,
        (),
        meters_per_cell,
        args,
    ).convert(
        "RGB"
    ).save(base_path)

    v1_image = base_image.copy()
    draw_ipp_marker(v1_image, ipp_cell, args)
    decorate_map_frame(
        v1_image,
        layer_presence,
        ("ipp",),
        meters_per_cell,
        args,
    ).convert("RGB").save(v1_path)

    v2_image = v1_image.copy()
    draw_find_marker(v2_image, find_cell, args)
    decorate_map_frame(
        v2_image,
        layer_presence,
        ("ipp", "find"),
        meters_per_cell,
        args,
    ).convert("RGB").save(v2_path)


def decorate_existing_incident(
    incident: Incident, args: argparse.Namespace
) -> dict[str, Any]:
    images_dir = args.output_dir / "images"
    image_path = images_dir / f"base_{incident.incident_index:03d}.png"
    v1_path = args.v1_dir / f"incident_{incident.incident_index:03d}.png"
    v2_path = args.v2_dir / f"incident_{incident.incident_index:03d}.png"
    find_east_m, find_north_m = latlon_offset_m(
        incident.find_lat,
        incident.find_lon,
        incident.ipp_lat,
        incident.ipp_lon,
    )
    meters_per_cell = meters_per_cell_for_incident(
        find_east_m,
        find_north_m,
        args.image_size,
        args,
    )
    required_paths = (image_path, v1_path, v2_path)
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise ValueError(f"Missing existing image files: {missing}")

    base_image = Image.open(image_path).convert("RGBA")
    if base_image.size != (args.image_size, args.image_size):
        raise ValueError(
            f"{image_path} is {base_image.size[0]}x{base_image.size[1]}, "
            f"expected {args.image_size}x{args.image_size}. Restore plain maps or "
            "regenerate before using --decorate-existing."
        )
    layer_presence = layer_presence_from_rendered_image(base_image, args)

    v1_image = Image.open(v1_path).convert("RGBA")
    v2_image = Image.open(v2_path).convert("RGBA")
    for path, image in ((v1_path, v1_image), (v2_path, v2_image)):
        if image.size != (args.image_size, args.image_size):
            raise ValueError(
                f"{path} is {image.size[0]}x{image.size[1]}, "
                f"expected {args.image_size}x{args.image_size}."
            )

    decorate_map_frame(
        base_image,
        layer_presence,
        (),
        meters_per_cell,
        args,
    ).convert(
        "RGB"
    ).save(image_path)
    decorate_map_frame(
        v1_image,
        layer_presence,
        ("ipp",),
        meters_per_cell,
        args,
    ).convert("RGB").save(v1_path)
    decorate_map_frame(
        v2_image,
        layer_presence,
        ("ipp", "find"),
        meters_per_cell,
        args,
    ).convert("RGB").save(v2_path)

    layers = ",".join(legend_layers(layer_presence, ()))
    print(
        f"decorated_existing: incident {incident.incident_index:03d} (layers={layers})"
    )
    return {
        "incident_index": incident.incident_index,
        "status": "decorated_existing",
        "legend_layers": layers,
        "meters_per_cell": meters_per_cell,
        "ground_width_m": meters_per_cell * args.image_size,
        "base_file": str(image_path),
        "v1_file": str(v1_path),
        "v2_file": str(v2_path),
        "matrix_file": "",
    }


def save_mat_file(
    output_path: Path,
    incident: Incident,
    bbox: tuple[float, float, float, float],
    elevation: np.ndarray,
    masks: dict[str, np.ndarray],
    slope: np.ndarray,
    inaccessible: np.ndarray,
    meters_per_cell: float,
    args: argparse.Namespace,
) -> None:
    linear_features = np.zeros_like(inaccessible, dtype=bool)
    for layer in LINEAR_LAYERS:
        linear_features |= masks[layer]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        output_path,
        {
            "sZelev": elevation.astype(np.float32),
            "BWLF": linear_features.astype(np.uint8),
            "BWInac": inaccessible.astype(np.uint8),
            "BWslope": slope.astype(np.uint8),
            "BW_streams": masks["streams"].astype(np.uint8),
            "BW_riverbanks": masks["riverbanks"].astype(np.uint8),
            "BW_roads": masks["roads"].astype(np.uint8),
            "BW_railroads": masks["railroads"].astype(np.uint8),
            "BW_powerlines": masks["powerlines"].astype(np.uint8),
            "BW_lake_shorelines": masks["lake_shorelines"].astype(np.uint8),
            "BW_hiking_trails": masks["hiking_trails"].astype(np.uint8),
            "BW_river_interiors": masks["river_interiors"].astype(np.uint8),
            "BW_lake_interiors": masks["lake_interiors"].astype(np.uint8),
            "incident_index": np.array([[incident.incident_index]], dtype=np.int32),
            "IPP_lat": np.array([[incident.ipp_lat]], dtype=np.float64),
            "IPP_lon": np.array([[incident.ipp_lon]], dtype=np.float64),
            "find_lat": np.array([[incident.find_lat]], dtype=np.float64),
            "find_lon": np.array([[incident.find_lon]], dtype=np.float64),
            "bbox_west_south_east_north": np.array([bbox], dtype=np.float64),
            "meters_per_cell": np.array([[meters_per_cell]], dtype=np.float64),
            "slope_threshold_deg": np.array(
                [[args.slope_threshold_deg]], dtype=np.float64
            ),
        },
        do_compression=True,
    )


def process_incident(incident: Incident, args: argparse.Namespace) -> dict[str, Any]:
    if args.decorate_existing:
        return decorate_existing_incident(incident, args)

    find_east_m, find_north_m = latlon_offset_m(
        incident.find_lat,
        incident.find_lon,
        incident.ipp_lat,
        incident.ipp_lon,
    )
    meters_per_cell = meters_per_cell_for_incident(
        find_east_m,
        find_north_m,
        args.image_size,
        args,
    )
    bbox = bbox_around_ipp(
        incident.ipp_lat,
        incident.ipp_lon,
        args.image_size,
        meters_per_cell,
    )
    images_dir = args.output_dir / "images"
    matrices_dir = args.output_dir / "matrices"
    image_path = images_dir / f"base_{incident.incident_index:03d}.png"
    v1_path = args.v1_dir / f"incident_{incident.incident_index:03d}.png"
    v2_path = args.v2_dir / f"incident_{incident.incident_index:03d}.png"
    mat_path = matrices_dir / f"incident_{incident.incident_index:03d}.mat"

    outputs_exist = (
        image_path.exists()
        and v1_path.exists()
        and v2_path.exists()
        and (args.no_mat or mat_path.exists())
    )
    if outputs_exist and not args.overwrite:
        print(f"skipped_exists: incident {incident.incident_index:03d}")
        return {
            "incident_index": incident.incident_index,
            "status": "skipped_exists",
            "base_file": str(image_path),
            "v1_file": str(v1_path),
            "v2_file": str(v2_path),
            "matrix_file": "" if args.no_mat else str(mat_path),
        }

    west, south, east, north = bbox
    print(
        f"incident {incident.incident_index:03d}: "
        f"bbox=({west:.6f},{south:.6f},{east:.6f},{north:.6f}), "
        f"scale={meters_per_cell:.3f} m/cell"
    )
    if args.dry_run:
        return {
            "incident_index": incident.incident_index,
            "status": "dry_run",
            "base_file": str(image_path),
            "v1_file": str(v1_path),
            "v2_file": str(v2_path),
            "matrix_file": "" if args.no_mat else str(mat_path),
        }

    elevation = fetch_usgs_dem(bbox, args.image_size, args)
    if args.skip_osm:
        masks = blank_masks(args.image_size)
        osm_feature_count = 0
    else:
        elements = fetch_osm_layers(bbox, args)
        osm_feature_count = len(elements)
        masks = rasterize_osm_layers(
            elements,
            bbox,
            args.image_size,
            max(1, args.line_width_px),
        )

    slope = slope_mask(elevation, meters_per_cell, args.slope_threshold_deg)
    matrix_masks = matrix_line_masks(masks, args.matrix_line_width_px)
    water_inaccessible = (
        matrix_masks["river_interiors"] | matrix_masks["lake_interiors"]
    )
    inaccessible = water_inaccessible | slope

    ipp_cell = (args.image_size / 2, args.image_size / 2)
    find_cell = (
        ipp_cell[0] + find_east_m / meters_per_cell,
        ipp_cell[1] + find_north_m / meters_per_cell,
    )

    image = render_map(elevation, masks, args)
    layer_presence = layer_presence_from_masks(elevation, masks, args)
    save_image_versions(
        image,
        layer_presence,
        ipp_cell,
        find_cell,
        meters_per_cell,
        image_path,
        v1_path,
        v2_path,
        args,
    )

    if not args.no_mat:
        save_mat_file(
            mat_path,
            incident,
            bbox,
            elevation,
            matrix_masks,
            slope,
            inaccessible,
            meters_per_cell,
            args,
        )

    found_inside = (
        0 <= find_cell[0] < args.image_size and 0 <= find_cell[1] < args.image_size
    )
    print(
        f"saved: {image_path}, {v1_path}, {v2_path} "
        f"(osm_features={osm_feature_count}, found_inside={found_inside})"
    )
    return {
        "incident_index": incident.incident_index,
        "status": "rendered",
        "IPP_lat": incident.ipp_lat,
        "IPP_lon": incident.ipp_lon,
        "find_lat": incident.find_lat,
        "find_lon": incident.find_lon,
        "bbox_west": west,
        "bbox_south": south,
        "bbox_east": east,
        "bbox_north": north,
        "meters_per_cell": meters_per_cell,
        "ground_width_m": meters_per_cell * args.image_size,
        "grid_step_m": args.grid_step_m,
        "scale_bar_m": args.scale_bar_m,
        "image_size": args.image_size,
        "find_cell_x": find_cell[0],
        "find_cell_y": find_cell[1],
        "found_inside_image": found_inside,
        "osm_feature_count": osm_feature_count,
        "legend_layers": ",".join(legend_layers(layer_presence, ())),
        "base_file": str(image_path),
        "v1_file": str(v1_path),
        "v2_file": str(v2_path),
        "matrix_file": "" if args.no_mat else str(mat_path),
    }


def write_metadata(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = output_dir / "metadata.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"metadata: {path}")


def expected_image_paths(
    incident_index: int, args: argparse.Namespace
) -> dict[str, Path]:
    return {
        "base": args.output_dir / "images" / f"base_{incident_index:03d}.png",
        "v1": args.v1_dir / f"incident_{incident_index:03d}.png",
        "v2": args.v2_dir / f"incident_{incident_index:03d}.png",
    }


def format_incident_numbers(indices: list[int]) -> str:
    return ", ".join(f"{index:03d}" for index in sorted(indices))


def clipped_error(value: Any, max_chars: int = 220) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def print_run_summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    print("\nRun summary:")
    if status_counts:
        counts_text = ", ".join(
            f"{status}={count}" for status, count in sorted(status_counts.items())
        )
        print(f"  Status counts: {counts_text}")

    failed_rows = [row for row in rows if row.get("status") == "failed"]
    if failed_rows:
        failed_indices = [
            int(row["incident_index"])
            for row in failed_rows
            if "incident_index" in row
        ]
        print(f"  Failed incidents: {format_incident_numbers(failed_indices)}")
        for row in failed_rows:
            incident_index = int(row["incident_index"])
            error_type = row.get("error_type", type(row.get("error")).__name__)
            error = clipped_error(row.get("error", ""))
            print(f"    incident {incident_index:03d}: {error_type}: {error}")
    else:
        print("  Failed incidents: none")

    if args.dry_run:
        print("  Missing output images: not checked (--dry-run)")
        return

    missing_rows = []
    for row in rows:
        if "incident_index" not in row:
            continue
        incident_index = int(row["incident_index"])
        paths = expected_image_paths(incident_index, args)
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            missing_rows.append((incident_index, missing, paths))

    if missing_rows:
        print("  Missing output images:")
        for incident_index, missing, paths in missing_rows:
            missing_text = ", ".join(
                f"{name} ({paths[name]})" for name in missing
            )
            print(f"    incident {incident_index:03d}: {missing_text}")
    else:
        print("  Missing output images: none")
        if failed_rows:
            print(
                "  Note: failed incidents still have image files on disk; "
                "those files may be from an earlier run."
            )


def main() -> None:
    args = parse_args()
    validate_args(args)
    incidents = selected_incidents(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Incidents: {len(incidents)}")
    print(f"Output: {args.output_dir}")
    print(f"V1 output: {args.v1_dir}")
    print(f"V2 output: {args.v2_dir}")
    print(f"Grid: {args.image_size}x{args.image_size}")
    if args.failed_from_metadata:
        print(f"Selection: failed incidents from {args.output_dir / 'metadata.csv'}")
    if args.meters_per_cell is None:
        print(
            "Scale: auto per incident "
            f"(min width={args.min_ground_width_m / 1000:g} km, "
            f"padding={args.point_padding_m:g} m)"
        )
    else:
        print(f"Scale: fixed {args.meters_per_cell:g} m/cell")
    print(
        "Axis labels: "
        f"{AXIS_LABELS_PER_SIDE} per side, "
        f"spacing snapped to {scale_bar_label(args.grid_step_m)} multiples"
    )
    print(
        f"HTTP attempts: {args.max_attempts} "
        f"(initial retry delay={args.retry_delay_s:g}s)"
    )
    if not args.no_grid_legend:
        print(f"Legend layer threshold: {args.legend_min_coverage_pct:g}%")
    if args.decorate_existing:
        print("Mode: decorate existing PNGs; network downloads skipped")
    else:
        print(f"USGS 3DEP: {args.usgs_url}")
        if args.skip_osm:
            print("OSM layers: skipped")
        else:
            print(f"OSM layers: {args.overpass_url}")

    rows = []
    for incident in incidents:
        try:
            rows.append(process_incident(incident, args))
        except Exception as error:
            print(
                f"failed: incident {incident.incident_index:03d}: "
                f"{type(error).__name__}: {error}"
            )
            image_paths = expected_image_paths(incident.incident_index, args)
            rows.append(
                {
                    "incident_index": incident.incident_index,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "base_file": str(image_paths["base"]),
                    "v1_file": str(image_paths["v1"]),
                    "v2_file": str(image_paths["v2"]),
                }
            )
            if not args.keep_going:
                write_metadata(args.output_dir, rows)
                print_run_summary(rows, args)
                raise SystemExit(1) from error
        if args.rate_limit_s and not args.dry_run and not args.decorate_existing:
            time.sleep(args.rate_limit_s)

    write_metadata(args.output_dir, rows)
    print_run_summary(rows, args)


if __name__ == "__main__":
    main()
