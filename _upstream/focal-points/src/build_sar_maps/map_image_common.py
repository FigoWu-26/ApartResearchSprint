import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


STATIC_MAPS_URL = "https://maps.googleapis.com/maps/api/staticmap"
EARTH_RADIUS_M = 6_371_008.8
WEB_MERCATOR_EQUATOR_M_PER_PX = 156_543.03392
GOOGLE_TILE_SIZE_PX = 256
REQUIRED_COLUMNS = {"incident_index", "IPP_lat", "IPP_lon", "find_lat", "find_lon"}
MAP_TYPES = ("roadmap", "satellite", "terrain", "hybrid")


@dataclass(frozen=True)
class Incident:
    incident_index: int
    ipp_lat: float
    ipp_lon: float
    find_lat: float
    find_lon: float
    found_east_m: float
    found_north_m: float
    ipp_to_found_m: float


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def latlon_offset_m(
    lat: float,
    lon: float,
    reference_lat: float,
    reference_lon: float,
) -> tuple[float, float]:
    mean_lat_rad = math.radians((lat + reference_lat) / 2)
    east_m = (
        math.radians(lon - reference_lon)
        * EARTH_RADIUS_M
        * math.cos(mean_lat_rad)
    )
    north_m = math.radians(lat - reference_lat) * EARTH_RADIUS_M
    return east_m, north_m


def read_initial_conditions(csv_path: Path) -> list[Incident]:
    with csv_path.open(newline="") as file:
        first_line = file.readline()
        if first_line.lstrip("\ufeff").startswith("incident_index"):
            file.seek(0)

        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

        incidents = []
        for row in reader:
            incident_index = int(row["incident_index"])
            ipp_lat = float(row["IPP_lat"])
            ipp_lon = float(row["IPP_lon"])
            find_lat = float(row["find_lat"])
            find_lon = float(row["find_lon"])
            found_east_m, found_north_m = latlon_offset_m(
                find_lat,
                find_lon,
                ipp_lat,
                ipp_lon,
            )
            incidents.append(
                Incident(
                    incident_index=incident_index,
                    ipp_lat=ipp_lat,
                    ipp_lon=ipp_lon,
                    find_lat=find_lat,
                    find_lon=find_lon,
                    found_east_m=found_east_m,
                    found_north_m=found_north_m,
                    ipp_to_found_m=haversine_m(ipp_lat, ipp_lon, find_lat, find_lon),
                )
            )

    if not incidents:
        raise ValueError(f"No initial conditions found in {csv_path}")
    return incidents


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile for an empty list")
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    lower_weight = upper - position
    upper_weight = position - lower
    return sorted_values[lower] * lower_weight + sorted_values[upper] * upper_weight


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def extent_value_m(incident: Incident, metric: str) -> float:
    if metric == "component":
        return max(abs(incident.found_east_m), abs(incident.found_north_m))
    if metric == "distance":
        return incident.ipp_to_found_m
    raise ValueError(f"Unsupported extent metric: {metric}")


def target_ground_width_m(
    incidents: list[Incident],
    ground_width_km: float | None,
    usable_fraction: float,
    extent_metric: str,
    extent_stat: str,
) -> tuple[float, str]:
    if ground_width_km is not None:
        return ground_width_km * 1000, "forced"

    values = [extent_value_m(incident, extent_metric) for incident in incidents]
    stats = summarize(values)
    selected = stats[extent_stat]
    width_m = 2 * selected / usable_fraction
    return width_m, f"{extent_stat}_{extent_metric}"


def meters_per_pixel(latitude_deg: float, zoom: int) -> float:
    return (
        WEB_MERCATOR_EQUATOR_M_PER_PX
        * math.cos(math.radians(latitude_deg))
        / (2**zoom)
    )


def actual_ground_width_m(latitude_deg: float, zoom: int, map_width_px: int) -> float:
    return meters_per_pixel(latitude_deg, zoom) * map_width_px


def zoom_for_ground_width(
    latitude_deg: float,
    map_width_px: int,
    target_width_m: float,
) -> int:
    numerator = (
        WEB_MERCATOR_EQUATOR_M_PER_PX
        * math.cos(math.radians(latitude_deg))
        * map_width_px
    )
    raw_zoom = math.log2(numerator / target_width_m)
    return max(0, min(21, math.floor(raw_zoom)))


def common_zoom_for_ground_width(
    incidents: list[Incident],
    map_width_px: int,
    target_width_m: float,
) -> int:
    for zoom in range(21, -1, -1):
        if all(
            actual_ground_width_m(incident.ipp_lat, zoom, map_width_px) >= target_width_m
            for incident in incidents
        ):
            return zoom
    return 0


def common_zoom_for_visibility(
    incidents: list[Incident],
    map_width_px: int,
    usable_fraction: float,
) -> int:
    for zoom in range(21, -1, -1):
        if all(
            visibility_ratio(
                incident,
                actual_ground_width_m(incident.ipp_lat, zoom, map_width_px),
            )
            <= usable_fraction
            for incident in incidents
        ):
            return zoom
    return 0


def zoom_for_visibility(
    incident: Incident,
    map_width_px: int,
    usable_fraction: float,
) -> int:
    for zoom in range(21, -1, -1):
        ground_width_m = actual_ground_width_m(incident.ipp_lat, zoom, map_width_px)
        if visibility_ratio(incident, ground_width_m) <= usable_fraction:
            return zoom
    return 0


def choose_zooms(
    incidents: list[Incident],
    zoom: int | None,
    zoom_mode: str,
    map_width_px: int,
    target_width_m: float,
    usable_fraction: float,
) -> dict[int, int]:
    if zoom is not None:
        return {incident.incident_index: zoom for incident in incidents}

    if zoom_mode == "fixed":
        common_zoom = common_zoom_for_visibility(incidents, map_width_px, usable_fraction)
        return {incident.incident_index: common_zoom for incident in incidents}

    if zoom_mode == "fit-each":
        return {
            incident.incident_index: zoom_for_visibility(
                incident,
                map_width_px,
                usable_fraction,
            )
            for incident in incidents
        }

    return {
        incident.incident_index: zoom_for_ground_width(
            incident.ipp_lat,
            map_width_px,
            target_width_m,
        )
        for incident in incidents
    }


def visibility_ratio(incident: Incident, ground_width_m: float) -> float:
    half_width_m = ground_width_m / 2
    return max(abs(incident.found_east_m), abs(incident.found_north_m)) / half_width_m


def google_coord(lat: float, lon: float) -> str:
    return f"{lat:.12f},{lon:.12f}"


def output_extension(image_format: str) -> str:
    if image_format in {"jpg", "jpg-baseline"}:
        return "jpg"
    if image_format == "png32":
        return "png"
    return image_format


def add_scale_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--size",
        type=int,
        default=640,
        help="Logical Static Maps size in pixels. Google Static Maps allows up to 640.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        choices=(1, 2),
        default=2,
        help="Static Maps pixel scale. size=640 and scale=2 returns 1280x1280 pixels.",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=None,
        help="Optional fixed zoom for every image. Overrides --zoom-mode.",
    )
    parser.add_argument(
        "--zoom-mode",
        choices=("fixed", "target-width", "fit-each"),
        default="fit-each",
        help=(
            "fit-each zooms each incident separately for a tighter view. fixed chooses "
            "one shared zoom that keeps every found location visible. target-width chooses "
            "a zoom per incident so each image is at least the target ground width."
        ),
    )
    parser.add_argument(
        "--ground-width-km",
        type=float,
        default=None,
        help="Override the dataset-derived target ground width.",
    )
    parser.add_argument(
        "--usable-fraction",
        type=float,
        default=0.9,
        help=(
            "Fraction of center-to-edge distance where the found marker should fit. "
            "0.9 leaves a 10 percent margin before the image edge."
        ),
    )
    parser.add_argument(
        "--extent-metric",
        choices=("component", "distance"),
        default="component",
        help=(
            "component uses max(abs(east), abs(north)), which fits a square image. "
            "distance reproduces the radial IPP-to-found approach."
        ),
    )
    parser.add_argument(
        "--extent-stat",
        choices=("max", "p95", "mean"),
        default="max",
        help="Statistic used to derive target ground width when --ground-width-km is not set.",
    )


def validate_scale_args(args: argparse.Namespace) -> None:
    if args.size < 1 or args.size > 640:
        raise ValueError("--size must be between 1 and 640")
    if args.zoom is not None and (args.zoom < 0 or args.zoom > 21):
        raise ValueError("--zoom must be between 0 and 21")
    if args.ground_width_km is not None and args.ground_width_km <= 0:
        raise ValueError("--ground-width-km must be positive")
    if args.usable_fraction <= 0 or args.usable_fraction > 1:
        raise ValueError("--usable-fraction must be in the range (0, 1]")


def print_dataset_stats(incidents: list[Incident], target_width_m: float, basis: str) -> None:
    distances_km = [incident.ipp_to_found_m / 1000 for incident in incidents]
    components_km = [
        max(abs(incident.found_east_m), abs(incident.found_north_m)) / 1000
        for incident in incidents
    ]
    distance_stats = summarize(distances_km)
    component_stats = summarize(components_km)
    print(f"Rows: {len(incidents)}")
    print(
        "IPP-to-found distance km: "
        f"mean={distance_stats['mean']:.2f}, "
        f"p95={distance_stats['p95']:.2f}, "
        f"max={distance_stats['max']:.2f}"
    )
    print(
        "Square-frame component km: "
        f"mean={component_stats['mean']:.2f}, "
        f"p95={component_stats['p95']:.2f}, "
        f"max={component_stats['max']:.2f}"
    )
    print(f"Target ground width: {target_width_m / 1000:.2f} km ({basis})")


def web_mercator_world_px(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    siny = math.sin(math.radians(lat))
    siny = min(max(siny, -0.9999), 0.9999)
    world_size = GOOGLE_TILE_SIZE_PX * (2**zoom)
    x = world_size * (0.5 + lon / 360)
    y = world_size * (
        0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)
    )
    return x, y


def latlon_to_image_px(
    lat: float,
    lon: float,
    center_lat: float,
    center_lon: float,
    zoom: int,
    logical_size_px: int,
    image_width_px: int,
    image_height_px: int,
) -> tuple[float, float]:
    point_x, point_y = web_mercator_world_px(lat, lon, zoom)
    center_x, center_y = web_mercator_world_px(center_lat, center_lon, zoom)
    scale_x = image_width_px / logical_size_px
    scale_y = image_height_px / logical_size_px
    x = image_width_px / 2 + (point_x - center_x) * scale_x
    y = image_height_px / 2 + (point_y - center_y) * scale_y
    return x, y
