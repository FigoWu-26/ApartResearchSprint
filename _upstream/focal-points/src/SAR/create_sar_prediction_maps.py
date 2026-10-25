import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

try:
    from .sar_method_colors import METHOD_COLORS, color_for_method, label_for_method
except ImportError:
    from sar_method_colors import METHOD_COLORS, color_for_method, label_for_method


DEFAULT_GROUND_TRUTH_CSV = Path("data/SAR_prompts/find_location_ground_truth.csv")
DEFAULT_PARSED_ROOT = Path("data/SAR_prompts/batch_responses/openai/gpt-5.5")
DEFAULT_V2_DIR = Path("data/SAR_maps/v2/usgs_terrain")
DEFAULT_OUTPUT_DIR = Path("data/SAR_maps/v3/usgs_terrain")
DEFAULT_STAR_RADIUS_PX = 52


@dataclass(frozen=True)
class IncidentMapInfo:
    incident_index: int
    meters_per_cell: float
    ground_width_m: float
    image_size: int
    map_origin_x: int
    map_origin_y: int
    v2_map_path: Path | None


@dataclass(frozen=True)
class Prediction:
    incident_index: int
    method: str
    repeat: int | None
    x_m: float
    y_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create data/SAR_maps/v3 maps by overlaying method/version prediction "
            "stars on the v2 maps that already contain the IPP and ground-truth find."
        )
    )
    parser.add_argument(
        "--ground-truth-csv",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_CSV,
        help=f"Ground-truth CSV. Defaults to {DEFAULT_GROUND_TRUTH_CSV}.",
    )
    parser.add_argument(
        "--parsed-json",
        nargs="+",
        type=Path,
        default=[DEFAULT_PARSED_ROOT],
        help=(
            "Parsed SAR response JSON files or directories. Directories are searched "
            "recursively for *.parsed.json. Defaults to the gpt-5.5 response root."
        ),
    )
    parser.add_argument(
        "--v2-dir",
        type=Path,
        default=DEFAULT_V2_DIR,
        help=f"Directory with v2 maps. Defaults to {DEFAULT_V2_DIR}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for v3 maps. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=(
            "Optional method keys to draw, e.g. vanilla_v1 saliency_v3. "
            "Defaults to every method/version found in the parsed files."
        ),
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        default=None,
        help='Optional incident selection, e.g. "1-5", "7", or "1-5, 10".',
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include parsed files under test directories.",
    )
    parser.add_argument(
        "--star-radius-px",
        type=int,
        default=DEFAULT_STAR_RADIUS_PX,
        help=f"Prediction-star outer radius. Defaults to {DEFAULT_STAR_RADIUS_PX}.",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Do not add the prediction-method legend in the right margin.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Optional CSV listing every rendered prediction marker.",
    )
    return parser.parse_args()


def parse_positive_int(value: str, label: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer, got {value!r}") from exc
    if number <= 0:
        raise ValueError(f"{label} must be positive, got {value!r}")
    return number


def parse_index_selection(parts: list[str] | None) -> set[int] | None:
    if not parts:
        return None

    indices: set[int] = set()
    for raw_part in parts:
        for chunk in raw_part.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "-" in chunk:
                start_text, end_text = chunk.split("-", 1)
                start = parse_positive_int(start_text.strip(), "incident index")
                end = parse_positive_int(end_text.strip(), "incident index")
                if end < start:
                    raise ValueError(f"Incident range must be ascending, got {chunk!r}")
                indices.update(range(start, end + 1))
            else:
                indices.add(parse_positive_int(chunk, "incident index"))

    return indices


def read_ground_truth(path: Path) -> dict[int, IncidentMapInfo]:
    infos: dict[int, IncidentMapInfo] = {}
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {
            "incident_index",
            "meters_per_cell",
            "ground_width_m",
            "find_cell_x",
            "find_cell_y",
            "find_pixel_x_in_framed_png",
            "find_pixel_y_in_framed_png",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        for row in reader:
            incident_index = int(row["incident_index"])
            meters_per_cell = float(row["meters_per_cell"])
            ground_width_m = float(row["ground_width_m"])
            image_size = round(ground_width_m / meters_per_cell)
            find_cell_x = float(row["find_cell_x"])
            find_cell_y = float(row["find_cell_y"])
            find_pixel_x = int(round(float(row["find_pixel_x_in_framed_png"])))
            find_pixel_y = int(round(float(row["find_pixel_y_in_framed_png"])))
            map_origin_x = find_pixel_x - round(find_cell_x)
            map_origin_y = find_pixel_y - round(image_size - 1 - find_cell_y)
            v2_map_text = row.get("v2_map_path") or ""
            infos[incident_index] = IncidentMapInfo(
                incident_index=incident_index,
                meters_per_cell=meters_per_cell,
                ground_width_m=ground_width_m,
                image_size=image_size,
                map_origin_x=map_origin_x,
                map_origin_y=map_origin_y,
                v2_map_path=Path(v2_map_text) if v2_map_text else None,
            )

    if not infos:
        raise ValueError(f"No ground-truth rows found in {path}")
    return infos


def expand_parsed_paths(paths: Iterable[Path], include_tests: bool) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*.parsed.json")))
        else:
            expanded.append(path)

    result = []
    for path in expanded:
        if not path.exists():
            raise FileNotFoundError(f"Parsed JSON path does not exist: {path}")
        normalized_parts = {part.lower() for part in path.parts}
        if not include_tests and "tests" in normalized_parts:
            continue
        result.append(path)

    if not result:
        raise ValueError("No parsed JSON files found")
    return sorted(dict.fromkeys(result))


def iter_parsed_entries(parsed: dict[str, Any]) -> Iterable[dict[str, Any]]:
    prompts = parsed.get("prompts")
    if not isinstance(prompts, dict):
        return
    for entries in prompts.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                yield entry


def entry_method(entry: dict[str, Any]) -> str:
    prompt_name = entry.get("prompt_name")
    prompt_version = entry.get("prompt_version")
    manifest = entry.get("manifest")
    if (not prompt_name or not prompt_version) and isinstance(manifest, dict):
        prompt_name = prompt_name or manifest.get("prompt_name")
        prompt_version = prompt_version or manifest.get("prompt_version")
    if prompt_name and prompt_version:
        return f"{prompt_name}_{prompt_version}"
    return str(prompt_name or "unknown")


def entry_incident_index(entry: dict[str, Any]) -> int:
    incident_index = entry.get("incident_index")
    if incident_index is None and isinstance(entry.get("manifest"), dict):
        incident_index = entry["manifest"].get("incident_index")
    if incident_index is None:
        raise ValueError(
            f"Parsed entry is missing incident_index: {entry.get('custom_id')}"
        )
    return int(incident_index)


def entry_repeat(entry: dict[str, Any]) -> int | None:
    repeat = entry.get("repeat")
    if repeat is None and isinstance(entry.get("manifest"), dict):
        repeat = entry["manifest"].get("repeat")
    return int(repeat) if repeat is not None else None


def collect_predictions(
    parsed_paths: list[Path],
    selected_methods: set[str] | None,
    selected_indices: set[int] | None,
) -> tuple[list[Prediction], list[str]]:
    predictions: list[Prediction] = []
    skipped: list[str] = []
    seen: set[tuple[int, str, int | None, float, float]] = set()

    for path in parsed_paths:
        with path.open(encoding="utf-8") as file:
            parsed = json.load(file)
        for entry in iter_parsed_entries(parsed):
            coordinate = entry.get("coordinate")
            if not isinstance(coordinate, dict):
                continue
            if coordinate.get("x") is None or coordinate.get("y") is None:
                continue

            incident_index = entry_incident_index(entry)
            method = entry_method(entry)
            if selected_indices is not None and incident_index not in selected_indices:
                continue
            if selected_methods is not None and method not in selected_methods:
                continue

            x_m = float(coordinate["x"])
            y_m = float(coordinate["y"])
            repeat = entry_repeat(entry)
            key = (incident_index, method, repeat, x_m, y_m)
            if key in seen:
                skipped.append(f"duplicate prediction skipped: {key}")
                continue
            seen.add(key)
            predictions.append(
                Prediction(
                    incident_index=incident_index,
                    method=method,
                    repeat=repeat,
                    x_m=x_m,
                    y_m=y_m,
                )
            )

    return predictions, skipped


def rgb_from_hex(color: str) -> tuple[int, int, int]:
    value = color.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {color!r}")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


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


def draw_prediction_star(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    color: str,
    radius: int,
) -> None:
    fill = (*rgb_from_hex(color), 255)
    outline = (25, 25, 25, 255)
    star = star_points(center[0], center[1], radius, radius * 0.45)
    draw.polygon(star, fill=fill, outline=outline)
    draw.line(star + [star[0]], fill=outline, width=3)


def prediction_pixel(
    prediction: Prediction,
    info: IncidentMapInfo,
) -> tuple[float, float]:
    cell_x = info.image_size / 2 + prediction.x_m / info.meters_per_cell
    cell_y = info.image_size / 2 + prediction.y_m / info.meters_per_cell
    return (
        info.map_origin_x + round(cell_x),
        info.map_origin_y + round(info.image_size - 1 - cell_y),
    )


def in_map_bounds(pixel: tuple[float, float], info: IncidentMapInfo) -> bool:
    x, y = pixel
    return (
        info.map_origin_x <= x < info.map_origin_x + info.image_size
        and info.map_origin_y <= y < info.map_origin_y + info.image_size
    )


def draw_prediction_legend(
    image: Image.Image,
    methods: list[str],
    info: IncidentMapInfo,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    map_right = info.map_origin_x + info.image_size
    x = map_right + max(220, round(info.image_size * 0.075))
    y = info.map_origin_y + round(info.image_size * 0.70)
    font_size = max(32, round(info.image_size * 0.026))
    title_font = load_font(font_size, bold=True)
    font = load_font(font_size)
    row_height = round(font_size * 1.35)
    sample_radius = round(font_size * 0.48)
    padding_x = round(font_size * 0.85)
    padding_y = round(font_size * 0.7)
    sample_width = round(font_size * 1.7)
    gap = round(font_size * 0.65)

    labels = [label_for_method(method) for method in methods]
    probe = Image.new("RGBA", (1, 1))
    probe_draw = ImageDraw.Draw(probe)
    title = "Model find predictions"
    max_text_width = max(
        [probe_draw.textbbox((0, 0), title, font=title_font)[2]]
        + [probe_draw.textbbox((0, 0), label, font=font)[2] for label in labels]
    )
    box_width = padding_x * 2 + sample_width + gap + max_text_width
    box_height = padding_y * 2 + row_height * (len(methods) + 1)
    right = x + box_width
    bottom = y + box_height

    draw.rectangle(
        (x, y, right, bottom),
        fill=(255, 255, 255, 238),
        outline=(35, 35, 35, 255),
        width=3,
    )
    draw.text(
        (x + padding_x, y + padding_y - round(font_size * 0.10)),
        title,
        font=title_font,
        fill=(20, 20, 20, 255),
    )

    for index, method in enumerate(methods):
        row_center_y = y + padding_y + row_height * (index + 1) + row_height // 2
        sample_center_x = x + padding_x + sample_width // 2
        draw_prediction_star(
            draw,
            (sample_center_x, row_center_y),
            color_for_method(method),
            sample_radius,
        )
        draw.text(
            (
                x + padding_x + sample_width + gap,
                row_center_y - round(font_size * 0.52),
            ),
            label_for_method(method),
            font=font,
            fill=(20, 20, 20, 255),
        )


def method_sort_key(method: str) -> tuple[int, str]:
    known = list(METHOD_COLORS)
    if method in known:
        return known.index(method), method
    return len(known), method


def v2_path_for_incident(info: IncidentMapInfo, v2_dir: Path) -> Path:
    if info.v2_map_path is not None and info.v2_map_path.exists():
        return info.v2_map_path
    return v2_dir / f"incident_{info.incident_index:03d}.png"


def render_maps(
    *,
    infos: dict[int, IncidentMapInfo],
    predictions: list[Prediction],
    selected_indices: set[int] | None,
    v2_dir: Path,
    output_dir: Path,
    star_radius_px: int,
    add_legend: bool,
) -> list[dict[str, str]]:
    by_incident: dict[int, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        by_incident[prediction.incident_index].append(prediction)

    incident_indices = sorted(
        selected_indices if selected_indices is not None else by_incident
    )
    metadata_rows: list[dict[str, str]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for incident_index in incident_indices:
        info = infos.get(incident_index)
        if info is None:
            raise ValueError(f"Missing map info for incident {incident_index}")

        input_path = v2_path_for_incident(info, v2_dir)
        if not input_path.exists():
            raise FileNotFoundError(
                f"Missing v2 map for incident {incident_index}: {input_path}"
            )

        incident_predictions = by_incident.get(incident_index, [])
        methods = sorted(
            {prediction.method for prediction in incident_predictions},
            key=method_sort_key,
        )
        image = Image.open(input_path).convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")

        for prediction in sorted(
            incident_predictions,
            key=lambda item: (*method_sort_key(item.method), item.repeat or 0),
        ):
            pixel = prediction_pixel(prediction, info)
            inside = in_map_bounds(pixel, info)
            if inside:
                draw_prediction_star(
                    draw,
                    pixel,
                    color_for_method(prediction.method),
                    star_radius_px,
                )
            metadata_rows.append(
                {
                    "incident_index": str(prediction.incident_index),
                    "method": prediction.method,
                    "repeat": (
                        "" if prediction.repeat is None else str(prediction.repeat)
                    ),
                    "x_m": f"{prediction.x_m:.3f}",
                    "y_m": f"{prediction.y_m:.3f}",
                    "pixel_x": f"{pixel[0]:.3f}",
                    "pixel_y": f"{pixel[1]:.3f}",
                    "inside_map": str(inside),
                    "color": color_for_method(prediction.method),
                }
            )

        if add_legend and methods:
            draw_prediction_legend(image, methods, info)

        output_path = output_dir / f"incident_{incident_index:03d}.png"
        image.convert("RGB").save(output_path)
        print(
            f"saved: {output_path} "
            f"(prediction_stars={len(incident_predictions)}, methods={','.join(methods)})"
        )

    return metadata_rows


def write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "incident_index",
        "method",
        "repeat",
        "x_m",
        "y_m",
        "pixel_x",
        "pixel_y",
        "inside_map",
        "color",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    selected_indices = parse_index_selection(args.indices)
    selected_methods = set(args.methods) if args.methods else None
    infos = read_ground_truth(args.ground_truth_csv)
    parsed_paths = expand_parsed_paths(args.parsed_json, args.include_tests)
    predictions, skipped = collect_predictions(
        parsed_paths, selected_methods, selected_indices
    )
    if not predictions:
        raise ValueError("No parsed predictions matched the requested filters")

    rows = render_maps(
        infos=infos,
        predictions=predictions,
        selected_indices=selected_indices,
        v2_dir=args.v2_dir,
        output_dir=args.output_dir,
        star_radius_px=args.star_radius_px,
        add_legend=not args.no_legend,
    )
    metadata_path = args.metadata_csv or args.output_dir / "prediction_markers.csv"
    write_metadata(metadata_path, rows)

    print(f"parsed_files: {len(parsed_paths)}")
    print(f"predictions: {len(predictions)}")
    print(f"skipped: {len(skipped)}")
    print(f"output_dir: {args.output_dir}")
    print(f"metadata_csv: {metadata_path}")


if __name__ == "__main__":
    main()
