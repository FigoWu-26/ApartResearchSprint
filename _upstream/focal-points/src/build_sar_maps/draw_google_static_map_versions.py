import argparse
import csv
import math
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError as exc:
    raise SystemExit(
        "Pillow is required for drawing markers. Install the project requirements "
        "or run with the repository virtual environment."
    ) from exc

try:
    from .map_image_common import MAP_TYPES, latlon_to_image_px
except ImportError:
    from map_image_common import MAP_TYPES, latlon_to_image_px


RED = (226, 28, 35, 255)
YELLOW = (255, 214, 0, 255)
GOLD = (245, 176, 0, 255)
BLACK = (20, 20, 20, 255)
WHITE = (255, 255, 255, 255)
SHADOW = (0, 0, 0, 85)
ANTIALIAS_SCALE = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw local annotation versions from clean Google Static Maps base images. "
            "This script does not call Google APIs."
        )
    )
    parser.add_argument(
        "--maptype",
        choices=(*MAP_TYPES, "all"),
        default="satellite",
        help=(
            "Map type folder to read/write when --metadata, --v1-dir, or --v2-dir "
            "are not supplied. Use all to draw every supported type."
        ),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=(
            "Metadata CSV created by download_google_static_maps.py. Defaults to "
            "data/SAR_maps/base/<maptype>/image_metadata.csv."
        ),
    )
    parser.add_argument(
        "--v1-dir",
        type=Path,
        default=None,
        help="Output directory for IPP-only images. Defaults to data/SAR_maps/v1/<maptype>.",
    )
    parser.add_argument(
        "--v2-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for IPP plus found-location images. "
            "Defaults to data/SAR_maps/v2/<maptype>."
        ),
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=None,
        help="Optional incident_index values to draw.",
    )
    parser.add_argument(
        "--pin-radius-px",
        type=int,
        default=None,
        help="Red IPP pushpin head radius in output pixels. Default scales with image size.",
    )
    parser.add_argument(
        "--star-radius-px",
        type=int,
        default=None,
        help="Yellow found-location star outer radius in output pixels. Default scales with image size.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip metadata rows whose clean base image is missing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing v1/v2 files.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace, maptype: str) -> tuple[Path, Path, Path]:
    if args.maptype == "all" and args.metadata is not None:
        raise ValueError("--metadata cannot be combined with --maptype all")

    if args.metadata is None:
        metadata = Path("data") / "SAR_maps" / "base" / maptype / "image_metadata.csv"
    else:
        metadata = args.metadata

    if args.v1_dir is None:
        v1_dir = Path("data") / "SAR_maps" / "v1" / maptype
    elif args.maptype == "all":
        v1_dir = args.v1_dir / maptype
    else:
        v1_dir = args.v1_dir

    if args.v2_dir is None:
        v2_dir = Path("data") / "SAR_maps" / "v2" / maptype
    elif args.maptype == "all":
        v2_dir = args.v2_dir / maptype
    else:
        v2_dir = args.v2_dir

    return metadata, v1_dir, v2_dir


def read_metadata(metadata_path: Path) -> list[dict[str, str]]:
    with metadata_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows found in {metadata_path}")
    return rows


def marker_radii(image_width_px: int, args: argparse.Namespace) -> tuple[int, int]:
    pin_radius = args.pin_radius_px
    star_radius = args.star_radius_px
    if pin_radius is None:
        pin_radius = max(12, round(image_width_px * 0.010))
    if star_radius is None:
        star_radius = max(15, round(image_width_px * 0.014))
    return pin_radius, star_radius


def resampling_filter() -> int:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def scaled_points(
    points: list[tuple[float, float]],
    scale: int,
) -> list[tuple[float, float]]:
    return [(x * scale, y * scale) for x, y in points]


def polygon_points(
    x: float,
    y: float,
    outer_radius: float,
    inner_radius: float,
    point_count: int,
    rotation_rad: float,
) -> list[tuple[float, float]]:
    points = []
    for index in range(point_count * 2):
        radius = outer_radius if index % 2 == 0 else inner_radius
        angle = rotation_rad + index * math.pi / point_count
        points.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))
    return points


def make_pin_icon(radius: int) -> tuple[Image.Image, tuple[int, int]]:
    scale = ANTIALIAS_SCALE
    pad = max(5, round(radius * 0.42))
    shaft_length = radius * 2.85
    shaft_width = max(5, radius * 0.38)
    width = round(radius * 2.55 + pad * 2)
    height = round(radius * 2.12 + shaft_length + pad * 1.65)
    tip = (width / 2, height - pad * 0.38)
    head_center = (width / 2, pad + radius)

    def bbox(cx: float, cy: float, r: float) -> tuple[float, float, float, float]:
        return (
            (cx - r) * scale,
            (cy - r) * scale,
            (cx + r) * scale,
            (cy + r) * scale,
        )

    image = Image.new("RGBA", (width * scale, height * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")

    def scaled_rect(
        left: float,
        top: float,
        right: float,
        bottom: float,
    ) -> tuple[float, float, float, float]:
        return (left * scale, top * scale, right * scale, bottom * scale)

    cx, cy = head_center
    shaft_top = cy + radius * 0.82
    shaft_bottom = tip[1] - radius * 0.18
    shaft_left = cx - shaft_width / 2
    shaft_right = cx + shaft_width / 2
    shaft_rounding = shaft_width * scale / 2

    draw.ellipse(
        bbox(cx + radius * 0.08, tip[1] + radius * 0.05, radius * 0.42),
        fill=(0, 0, 0, 35),
    )
    draw.rounded_rectangle(
        scaled_rect(
            shaft_left + radius * 0.12,
            shaft_top + radius * 0.06,
            shaft_right + radius * 0.12,
            shaft_bottom + radius * 0.08,
        ),
        radius=shaft_rounding,
        fill=SHADOW,
    )
    shaft_outline = [
        (shaft_left - radius * 0.06, shaft_top),
        (shaft_right + radius * 0.06, shaft_top),
        (cx + shaft_width * 0.38, shaft_bottom),
        (cx, tip[1]),
        (cx - shaft_width * 0.38, shaft_bottom),
    ]
    draw.polygon(scaled_points(shaft_outline, scale), fill=(88, 88, 88, 255))
    shaft_body = [
        (shaft_left, shaft_top),
        (shaft_right, shaft_top),
        (cx + shaft_width * 0.27, shaft_bottom),
        (cx, tip[1] - radius * 0.06),
        (cx - shaft_width * 0.27, shaft_bottom),
    ]
    draw.polygon(scaled_points(shaft_body, scale), fill=(255, 255, 255, 255))
    draw.polygon(
        scaled_points(
            [
                (cx + shaft_width * 0.04, shaft_top + radius * 0.08),
                (shaft_right, shaft_top),
                (cx + shaft_width * 0.27, shaft_bottom),
                (cx + shaft_width * 0.04, tip[1] - radius * 0.07),
            ],
            scale,
        ),
        fill=(210, 210, 210, 185),
    )
    highlight_x = cx - shaft_width * 0.18
    draw.line(
        (
            highlight_x * scale,
            (shaft_top + radius * 0.98) * scale,
            (highlight_x + shaft_width * 0.05) * scale,
            (shaft_bottom - radius * 0.25) * scale,
        ),
        fill=(255, 255, 255, 220),
        width=max(1, round(shaft_width * 0.18 * scale)),
    )

    draw.ellipse(
        bbox(cx + radius * 0.13, cy + radius * 0.14, radius * 1.1),
        fill=(0, 0, 0, 115),
    )
    draw.ellipse(bbox(cx, cy, radius * 1.08), fill=WHITE)
    draw.ellipse(bbox(cx, cy, radius * 1.02), fill=(118, 25, 30, 255))
    draw.ellipse(bbox(cx, cy, radius), fill=(243, 31, 42, 255))
    draw.pieslice(
        bbox(cx, cy, radius),
        start=312,
        end=112,
        fill=(181, 31, 39, 230),
    )
    draw.pieslice(
        bbox(cx, cy, radius),
        start=72,
        end=220,
        fill=(244, 47, 57, 255),
    )
    draw.ellipse(
        bbox(cx - radius * 0.16, cy - radius * 0.10, radius * 0.78),
        fill=(244, 47, 57, 240),
    )
    draw.arc(
        (
            (cx - radius * 0.72) * scale,
            (cy - radius * 0.68) * scale,
            (cx + radius * 0.14) * scale,
            (cy + radius * 0.26) * scale,
        ),
        start=198,
        end=280,
        fill=(255, 159, 134, 210),
        width=max(2, round(radius * 0.2 * scale)),
    )
    draw.rounded_rectangle(
        scaled_rect(
            cx - radius * 0.18,
            cy - radius * 0.83,
            cx + radius * 0.04,
            cy - radius * 0.51,
        ),
        radius=max(1, round(radius * 0.05 * scale)),
        fill=(255, 159, 134, 185),
    )

    image = image.resize((width, height), resampling_filter())
    return image, (round(tip[0]), round(tip[1]))


def make_star_icon(radius: int) -> tuple[Image.Image, tuple[int, int]]:
    scale = ANTIALIAS_SCALE
    pad = max(5, round(radius * 0.45))
    size = round(radius * 2 + pad * 2)
    center = (size / 2, size / 2)
    image = Image.new("RGBA", (size * scale, size * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")

    def draw_star(
        outer_radius: float,
        inner_radius: float,
        fill: tuple[int, int, int, int],
        dx: float = 0,
        dy: float = 0,
    ) -> None:
        points = polygon_points(
            center[0] + dx,
            center[1] + dy,
            outer_radius,
            inner_radius,
            point_count=5,
            rotation_rad=-math.pi / 2,
        )
        draw.polygon(scaled_points(points, scale), fill=fill)

    draw_star(radius + 3.5, (radius + 3.5) * 0.48, SHADOW, dx=2.2, dy=2.2)
    draw_star(radius + 3.2, (radius + 3.2) * 0.48, WHITE)
    draw_star(radius + 1.3, (radius + 1.3) * 0.48, BLACK)
    draw_star(radius, radius * 0.48, YELLOW)
    draw_star(radius * 0.58, radius * 0.28, GOLD)

    image = image.resize((size, size), resampling_filter())
    return image, (round(center[0]), round(center[1]))


def paste_icon(
    image: Image.Image,
    icon: Image.Image,
    anchor: tuple[int, int],
    x: float,
    y: float,
) -> None:
    left = round(x - anchor[0])
    top = round(y - anchor[1])
    image.alpha_composite(icon, (left, top))


def parse_base_path(row: dict[str, str]) -> Path:
    base_file = row.get("base_file") or row.get("output_file")
    if not base_file:
        raise ValueError("Metadata row is missing base_file/output_file")
    path = Path(base_file)
    if not path.exists() and "\\" in base_file:
        path = Path(base_file.replace("\\", "/"))
    return path


def draw_versions_for_row(
    row: dict[str, str],
    args: argparse.Namespace,
    v1_dir: Path,
    v2_dir: Path,
) -> tuple[Path, Path] | None:
    incident_index = int(row["incident_index"])
    base_path = parse_base_path(row)
    if not base_path.exists():
        if args.skip_missing:
            print(f"missing base, skipped: incident {incident_index:03d} ({base_path})")
            return None
        raise FileNotFoundError(f"Missing base image for incident {incident_index}: {base_path}")

    v1_path = v1_dir / f"incident_{incident_index:03d}.png"
    v2_path = v2_dir / f"incident_{incident_index:03d}.png"
    if not args.overwrite and v1_path.exists() and v2_path.exists():
        print(f"skipped_exists: incident {incident_index:03d}")
        return v1_path, v2_path

    base_image = Image.open(base_path).convert("RGBA")
    width_px, height_px = base_image.size
    logical_size_px = int(float(row.get("size") or 640))
    zoom = int(float(row["zoom"]))
    center_lat = float(row.get("center_lat") or row["IPP_lat"])
    center_lon = float(row.get("center_lon") or row["IPP_lon"])
    ipp_lat = float(row["IPP_lat"])
    ipp_lon = float(row["IPP_lon"])
    find_lat = float(row["find_lat"])
    find_lon = float(row["find_lon"])

    ipp_x, ipp_y = latlon_to_image_px(
        ipp_lat,
        ipp_lon,
        center_lat,
        center_lon,
        zoom,
        logical_size_px,
        width_px,
        height_px,
    )
    find_x, find_y = latlon_to_image_px(
        find_lat,
        find_lon,
        center_lat,
        center_lon,
        zoom,
        logical_size_px,
        width_px,
        height_px,
    )
    pin_radius, star_radius = marker_radii(width_px, args)
    pin_icon, pin_anchor = make_pin_icon(pin_radius)
    star_icon, star_anchor = make_star_icon(star_radius)

    v1_dir.mkdir(parents=True, exist_ok=True)
    v2_dir.mkdir(parents=True, exist_ok=True)

    v1_image = base_image.copy()
    paste_icon(v1_image, pin_icon, pin_anchor, ipp_x, ipp_y)
    if args.overwrite or not v1_path.exists():
        v1_image.convert("RGB").save(v1_path)

    v2_image = base_image.copy()
    paste_icon(v2_image, pin_icon, pin_anchor, ipp_x, ipp_y)
    paste_icon(v2_image, star_icon, star_anchor, find_x, find_y)
    if args.overwrite or not v2_path.exists():
        v2_image.convert("RGB").save(v2_path)

    print(f"drawn: incident {incident_index:03d} -> {v1_path}, {v2_path}")
    return v1_path, v2_path


def draw_maptype(args: argparse.Namespace, maptype: str) -> None:
    metadata, v1_dir, v2_dir = resolve_paths(args, maptype)
    rows = read_metadata(metadata)
    if args.indices is not None:
        wanted = set(args.indices)
        rows = [row for row in rows if int(row["incident_index"]) in wanted]
        missing = wanted - {int(row["incident_index"]) for row in rows}
        if missing:
            raise ValueError(f"Requested incident indices not found in metadata: {sorted(missing)}")

    count = 0
    print(f"\nMap type: {maptype}")
    for row in sorted(rows, key=lambda item: int(item["incident_index"])):
        result = draw_versions_for_row(row, args, v1_dir, v2_dir)
        if result is not None:
            count += 1

    print(f"Done. Wrote or confirmed {count} incident(s).")
    print(f"metadata: {metadata}")
    print(f"v1 directory: {v1_dir}")
    print(f"v2 directory: {v2_dir}")


def main() -> None:
    args = parse_args()
    maptypes = MAP_TYPES if args.maptype == "all" else (args.maptype,)
    for maptype in maptypes:
        draw_maptype(args, maptype)


if __name__ == "__main__":
    main()
