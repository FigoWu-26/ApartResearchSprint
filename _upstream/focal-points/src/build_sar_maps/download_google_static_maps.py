import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

try:
    from .map_image_common import (
        STATIC_MAPS_URL,
        Incident,
        MAP_TYPES,
        actual_ground_width_m,
        add_scale_args,
        choose_zooms,
        google_coord,
        output_extension,
        print_dataset_stats,
        read_initial_conditions,
        target_ground_width_m,
        validate_scale_args,
        visibility_ratio,
    )
except ImportError:
    from map_image_common import (
        STATIC_MAPS_URL,
        Incident,
        MAP_TYPES,
        actual_ground_width_m,
        add_scale_args,
        choose_zooms,
        google_coord,
        output_extension,
        print_dataset_stats,
        read_initial_conditions,
        target_ground_width_m,
        validate_scale_args,
        visibility_ratio,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download clean Google Maps Static base images centered on "
            "IPP coordinates. This script does not draw IPP or found-location marks."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data") / "SAR_maps" / "InitialConditions.csv",
        help="Initial conditions CSV. A one-line title before the header is allowed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for clean base images and metadata. Defaults to "
            "data/SAR_maps/base/<maptype>."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default="GOOGLE_MAPS_API_KEY",
        help="Environment variable containing the Google Maps API key.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Dotenv file to load before reading --api-key-env.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Google Maps API key. Prefer --api-key-env to avoid exposing it in shell history.",
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=None,
        help="Optional incident_index values to download.",
    )
    parser.add_argument(
        "--maptype",
        choices=(*MAP_TYPES, "all"),
        default="satellite",
        help="Google map type, or all to download every supported type.",
    )
    parser.add_argument(
        "--format",
        choices=("png", "png32", "jpg", "jpg-baseline", "gif"),
        default="png32",
        help="Output image format requested from Google Static Maps.",
    )
    add_scale_args(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write metadata and redacted request URLs without downloading images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing base image files.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--retry-count",
        type=int,
        default=2,
        help="Number of retries for transient download failures.",
    )
    parser.add_argument(
        "--rate-limit-s",
        type=float,
        default=0.1,
        help="Sleep time between API requests.",
    )
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace, maptype: str) -> Path:
    if args.output_dir is not None:
        if args.maptype == "all":
            return args.output_dir / maptype
        return args.output_dir
    return Path("data") / "SAR_maps" / "base" / maptype


def validate_args(args: argparse.Namespace) -> None:
    validate_scale_args(args)
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be positive")
    if args.retry_count < 0:
        raise ValueError("--retry-count must be non-negative")
    if args.rate_limit_s < 0:
        raise ValueError("--rate-limit-s must be non-negative")


def static_map_params(
    incident: Incident,
    zoom: int,
    api_key: str,
    args: argparse.Namespace,
    maptype: str,
) -> list[tuple[str, str]]:
    ipp = google_coord(incident.ipp_lat, incident.ipp_lon)
    return [
        ("center", ipp),
        ("zoom", str(zoom)),
        ("size", f"{args.size}x{args.size}"),
        ("scale", str(args.scale)),
        ("maptype", maptype),
        ("format", args.format),
        ("key", api_key),
    ]


def static_map_url(params: list[tuple[str, str]]) -> str:
    return f"{STATIC_MAPS_URL}?{urllib.parse.urlencode(params, safe=',|:')}"


def redact_key(params: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(key, "REDACTED" if key == "key" else value) for key, value in params]


def download_image(
    url: str,
    output_path: Path,
    timeout_s: float,
    retry_count: int,
    overwrite: bool,
) -> str:
    if output_path.exists() and not overwrite:
        return "skipped_exists"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error = None

    for attempt in range(retry_count + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "focal-points-google-static-maps/1.0"},
            )
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                content_type = response.headers.get("Content-Type", "")
                body = response.read()

            if not content_type.startswith("image/"):
                excerpt = body[:500].decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Expected image response, got {content_type!r}: {excerpt}"
                )

            temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
            temp_path.write_bytes(body)
            temp_path.replace(output_path)
            return "downloaded"
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")[:500]
            last_error = RuntimeError(f"HTTP {error.code}: {body}")
            if error.code < 500 or attempt >= retry_count:
                break
        except (urllib.error.URLError, TimeoutError, RuntimeError) as error:
            last_error = error
            if attempt >= retry_count:
                break

        time.sleep(min(2**attempt, 10))

    raise RuntimeError(f"Failed to download {output_path}: {last_error}")


def metadata_row(
    incident: Incident,
    zoom: int,
    image_px: int,
    target_width_m: float,
    output_path: Path,
    status: str,
    redacted_url: str,
    args: argparse.Namespace,
    maptype: str,
) -> dict[str, object]:
    ground_width_m = actual_ground_width_m(incident.ipp_lat, zoom, args.size)
    ratio = visibility_ratio(incident, ground_width_m)
    return {
        "incident_index": incident.incident_index,
        "center_lat": incident.ipp_lat,
        "center_lon": incident.ipp_lon,
        "IPP_lat": incident.ipp_lat,
        "IPP_lon": incident.ipp_lon,
        "find_lat": incident.find_lat,
        "find_lon": incident.find_lon,
        "found_east_km": incident.found_east_m / 1000,
        "found_north_km": incident.found_north_m / 1000,
        "ipp_to_found_km": incident.ipp_to_found_m / 1000,
        "zoom": zoom,
        "size": args.size,
        "scale": args.scale,
        "map_width_px": args.size,
        "image_px": image_px,
        "maptype": maptype,
        "format": args.format,
        "target_ground_width_km": target_width_m / 1000,
        "actual_ground_width_km": ground_width_m / 1000,
        "visibility_ratio": ratio,
        "found_inside_image": ratio <= 1,
        "base_file": str(output_path),
        "status": status,
        "request_url_redacted": redacted_url,
    }


def write_metadata(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_incidents(
    all_incidents: list[Incident],
    indices: list[int] | None,
) -> list[Incident]:
    if indices is None:
        return all_incidents
    wanted = set(indices)
    incidents = [incident for incident in all_incidents if incident.incident_index in wanted]
    missing = wanted - {incident.incident_index for incident in incidents}
    if missing:
        raise ValueError(f"Requested incident indices not found: {sorted(missing)}")
    return incidents


def download_maptype(
    args: argparse.Namespace,
    maptype: str,
    all_incidents: list[Incident],
    incidents: list[Incident],
    api_key: str,
) -> None:
    output_dir = resolve_output_dir(args, maptype)
    all_incidents = sorted(
        all_incidents,
        key=lambda incident: incident.incident_index,
    )

    image_px = args.size * args.scale
    map_width_px = args.size
    target_width_m, target_basis = target_ground_width_m(
        all_incidents,
        ground_width_km=args.ground_width_km,
        usable_fraction=args.usable_fraction,
        extent_metric=args.extent_metric,
        extent_stat=args.extent_stat,
    )
    zooms = choose_zooms(
        all_incidents,
        zoom=args.zoom,
        zoom_mode=args.zoom_mode,
        map_width_px=map_width_px,
        target_width_m=target_width_m,
        usable_fraction=args.usable_fraction,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nMap type: {maptype}")
    print_dataset_stats(all_incidents, target_width_m, target_basis)
    if len(incidents) != len(all_incidents):
        print(
            f"Downloading {len(incidents)} selected incident(s); scale was computed "
            f"from all {len(all_incidents)} CSV rows."
        )

    metadata_rows = []
    extension = output_extension(args.format)
    for incident in incidents:
        zoom = zooms[incident.incident_index]
        output_path = output_dir / f"base_{incident.incident_index:03d}.{extension}"
        params = static_map_params(incident, zoom, api_key, args, maptype)
        request_url = static_map_url(params)
        redacted_url = static_map_url(redact_key(params))

        if args.dry_run:
            status = "dry_run"
        else:
            status = download_image(
                request_url,
                output_path,
                timeout_s=args.timeout_s,
                retry_count=args.retry_count,
                overwrite=args.overwrite,
            )
            if args.rate_limit_s:
                time.sleep(args.rate_limit_s)

        row = metadata_row(
            incident,
            zoom,
            image_px,
            target_width_m,
            output_path,
            status,
            redacted_url,
            args,
            maptype,
        )
        metadata_rows.append(row)
        print(
            f"{status}: incident {incident.incident_index:03d}, "
            f"zoom={zoom}, width={row['actual_ground_width_km']:.2f} km, "
            f"visibility={row['visibility_ratio']:.2f}"
        )

    metadata_path = output_dir / "image_metadata.csv"
    write_metadata(metadata_path, metadata_rows)

    summary = {
        "csv": str(args.csv),
        "output_dir": str(output_dir),
        "row_count": len(incidents),
        "scale_basis_row_count": len(all_incidents),
        "size": args.size,
        "scale": args.scale,
        "map_width_px": map_width_px,
        "image_px": image_px,
        "maptype": maptype,
        "format": args.format,
        "zoom_mode": "explicit" if args.zoom is not None else args.zoom_mode,
        "zooms_used": sorted({zooms[incident.incident_index] for incident in incidents}),
        "target_ground_width_km": target_width_m / 1000,
        "target_basis": target_basis,
        "usable_fraction": args.usable_fraction,
        "not_comfortable_count": sum(
            row["visibility_ratio"] > args.usable_fraction for row in metadata_rows
        ),
        "outside_image_count": sum(
            not row["found_inside_image"] for row in metadata_rows
        ),
        "metadata_file": str(metadata_path),
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if summary["outside_image_count"]:
        print(
            "Warning: some found locations are outside the image. "
            "Use a lower --zoom or larger --ground-width-km."
        )
    elif summary["not_comfortable_count"]:
        print(
            "Warning: some found locations are visible but outside the requested "
            "usable fraction. Use a lower --zoom or larger --ground-width-km."
        )

    print(f"Metadata saved to: {metadata_path}")
    print(f"Summary saved to: {summary_path}")
    print(
        "Next: run src/build_sar_maps/draw_google_static_map_versions.py to create "
        "data/SAR_maps/v1 and data/SAR_maps/v2 without another API request."
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    load_dotenv(args.env_file)

    all_incidents = sorted(
        read_initial_conditions(args.csv),
        key=lambda incident: incident.incident_index,
    )
    incidents = select_incidents(all_incidents, args.indices)

    api_key = args.api_key or os.environ.get(args.api_key_env)
    if not api_key and not args.dry_run:
        raise SystemExit(
            f"Missing Google Maps API key. Set {args.api_key_env} or pass --api-key."
        )
    if not api_key:
        api_key = "YOUR_API_KEY"

    maptypes = MAP_TYPES if args.maptype == "all" else (args.maptype,)
    for maptype in maptypes:
        download_maptype(args, maptype, all_incidents, incidents, api_key)


if __name__ == "__main__":
    main()
