import argparse
import os
import shutil
import subprocess
import sys
import types
import urllib.error
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


DEFAULT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
DEFAULT_WEIGHTS_DIR = Path("data") / "SAR_maps" / "models" / "realesrgan"
INSTALL_HINT = (
    "Install the Python backend dependencies with:\n"
    "  .venv/Scripts/python.exe -m pip install realesrgan opencv-python"
)

PYTHON_MODELS = {
    "realesrgan_x4plus": {
        "aliases": {"realesrgan_x4plus", "realesrgan-x4plus", "x4plus"},
        "file_name": "RealESRGAN_x4plus.pth",
        "url": (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/"
            "v0.1.0/RealESRGAN_x4plus.pth"
        ),
        "native_scale": 4,
        "num_block": 23,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upscale local map base images. The default backend uses the Python "
            "Real-ESRGAN library and does not call Google APIs."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data") / "SAR_maps" / "base",
        help="Root directory containing source map images.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data") / "SAR_maps" / "base_upscale",
        help="Root directory for upscaled images.",
    )
    parser.add_argument(
        "--backend",
        choices=("python", "ncnn"),
        default="python",
        help=(
            "python uses the realesrgan package. ncnn uses an external "
            "realesrgan-ncnn-vulkan executable."
        ),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional dotenv file for backend paths.",
    )
    parser.add_argument(
        "--model",
        default="realesrgan-x4plus",
        help="Model name. For the Python backend, realesrgan-x4plus is supported.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        choices=(2, 3, 4),
        default=4,
        help="Output scale. A 1280x1280 image becomes 5120x5120 with --scale 4.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help="Tile size. Use 0 for no tiling/auto behavior.",
    )
    parser.add_argument(
        "--gpu-id",
        default=None,
        help="Optional GPU id, for example 0. The Python backend auto-uses CUDA if available.",
    )
    parser.add_argument(
        "--format",
        choices=("png", "jpg", "webp"),
        default="png",
        help="Output image format.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_IMAGE_EXTENSIONS),
        help="Image extensions to process.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional path to a local RealESRGAN_x4plus.pth file.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=DEFAULT_WEIGHTS_DIR,
        help="Directory where Python backend weights are stored/downloaded.",
    )
    parser.add_argument(
        "--tile-pad",
        type=int,
        default=10,
        help="Python backend tile padding.",
    )
    parser.add_argument(
        "--pre-pad",
        type=int,
        default=0,
        help="Python backend pre padding.",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use fp32 instead of half precision in the Python backend.",
    )
    parser.add_argument(
        "--exe",
        type=Path,
        default=None,
        help=(
            "ncnn backend only: path to realesrgan-ncnn-vulkan executable. If omitted, "
            "the script checks REALESRGAN_NCNN_VULKAN and then PATH."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help=(
            "ncnn backend only: optional model directory passed as -m. If omitted, "
            "the script checks REALESRGAN_MODEL_DIR and then a models folder next to "
            "the executable."
        ),
    )
    parser.add_argument(
        "--threads",
        default=None,
        help="ncnn backend only: load:proc:save thread setting passed as -j.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned inputs and outputs without running upscaling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing upscaled files.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue processing remaining images if one image fails.",
    )
    return parser.parse_args()


def normalize_path(path: Path) -> Path:
    text = str(path)
    if os.name != "nt" and len(text) >= 3 and text[1] == ":" and text[2] in "\\/":
        return Path("/mnt") / text[0].lower() / text[3:].replace("\\", "/")
    if os.name != "nt" and "\\" in text:
        return Path(text.replace("\\", "/"))
    return path


def load_env(env_file: Path) -> None:
    if load_dotenv is not None and env_file.exists():
        load_dotenv(env_file)


def normalize_model_name(model_name: str) -> str:
    return model_name.lower().replace("-", "_")


def python_model_config(model_name: str) -> dict[str, object]:
    normalized = normalize_model_name(model_name)
    for config in PYTHON_MODELS.values():
        aliases = {normalize_model_name(alias) for alias in config["aliases"]}
        if normalized in aliases:
            return config
    supported = ", ".join(sorted(PYTHON_MODELS))
    raise SystemExit(f"Unsupported Python backend model {model_name!r}. Supported: {supported}")


def normalize_extensions(extensions: list[str]) -> set[str]:
    normalized = set()
    for extension in extensions:
        if not extension.startswith("."):
            extension = f".{extension}"
        normalized.add(extension.lower())
    return normalized


def find_images(input_root: Path, extensions: set[str]) -> list[Path]:
    if not input_root.exists():
        raise SystemExit(f"Input root does not exist: {input_root}")
    return sorted(
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def output_path_for(source: Path, input_root: Path, output_root: Path, output_format: str) -> Path:
    relative = source.relative_to(input_root)
    return (output_root / relative).with_suffix(f".{output_format}")


def download_weights(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    print(f"Downloading weights: {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            with temp_path.open("wb") as file:
                shutil.copyfileobj(response, file)
        temp_path.replace(output_path)
    except (urllib.error.URLError, TimeoutError) as error:
        if temp_path.exists():
            temp_path.unlink()
        raise SystemExit(f"Failed to download Real-ESRGAN weights: {error}") from error


def resolve_weights(args: argparse.Namespace, config: dict[str, object]) -> Path:
    if args.weights is not None:
        weights = normalize_path(args.weights)
        if not weights.exists():
            raise SystemExit(f"Weights file does not exist: {weights}")
        return weights

    weights_dir = normalize_path(args.weights_dir)
    weights_path = weights_dir / str(config["file_name"])
    if not weights_path.exists():
        download_weights(str(config["url"]), weights_path)
    return weights_path


def patch_torchvision_functional_tensor() -> None:
    try:
        import torchvision.transforms.functional as functional
    except ImportError:
        return

    module_name = "torchvision.transforms.functional_tensor"
    if module_name in sys.modules:
        return

    module = types.ModuleType(module_name)
    module.rgb_to_grayscale = functional.rgb_to_grayscale
    sys.modules[module_name] = module


def import_python_backend():
    try:
        patch_torchvision_functional_tensor()
        import cv2
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError as error:
        raise SystemExit(f"Missing Python Real-ESRGAN dependency: {error}\n{INSTALL_HINT}") from error
    return cv2, torch, RRDBNet, RealESRGANer


def make_python_upsampler(args: argparse.Namespace):
    cv2, torch, RRDBNet, RealESRGANer = import_python_backend()
    config = python_model_config(args.model)
    weights_path = resolve_weights(args, config)

    native_scale = int(config["native_scale"])
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=int(config["num_block"]),
        num_grow_ch=32,
        scale=native_scale,
    )

    cuda_available = torch.cuda.is_available()
    gpu_id = int(args.gpu_id) if args.gpu_id is not None else (0 if cuda_available else None)
    half = cuda_available and not args.fp32
    upsampler = RealESRGANer(
        scale=native_scale,
        model_path=str(weights_path),
        model=model,
        tile=args.tile_size,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=half,
        gpu_id=gpu_id,
    )

    device = torch.cuda.get_device_name(gpu_id) if gpu_id is not None and cuda_available else "CPU"
    print(f"Python backend: realesrgan")
    print(f"Weights: {weights_path}")
    print(f"Device: {device}")
    print(f"Half precision: {half}")
    return cv2, upsampler


def upscale_python_image(cv2, upsampler, source: Path, output_path: Path, args: argparse.Namespace) -> bool:
    if output_path.exists() and not args.overwrite:
        print(f"skipped_exists: {output_path}")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"failed: could not read {source}")
        return False

    print(f"upscaling: {source} -> {output_path}")
    try:
        output, _ = upsampler.enhance(image, outscale=args.scale)
        return bool(cv2.imwrite(str(output_path), output))
    except RuntimeError as error:
        print(f"failed: {source} ({error})")
        return False


def resolve_executable(explicit_exe: Path | None) -> Path:
    candidates = []
    if explicit_exe is not None:
        candidates.append(normalize_path(explicit_exe))

    env_exe = os.environ.get("REALESRGAN_NCNN_VULKAN")
    if env_exe:
        candidates.append(normalize_path(Path(env_exe)))

    for executable_name in ("realesrgan-ncnn-vulkan", "realesrgan-ncnn-vulkan.exe"):
        found = shutil.which(executable_name)
        if found:
            candidates.append(Path(found))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise SystemExit(
        "Could not find realesrgan-ncnn-vulkan. Pass --exe, add it to PATH, "
        "or set REALESRGAN_NCNN_VULKAN in .env."
    )


def resolve_model_dir(explicit_model_dir: Path | None, exe: Path) -> Path | None:
    candidates = []
    if explicit_model_dir is not None:
        candidates.append(normalize_path(explicit_model_dir))

    env_model_dir = os.environ.get("REALESRGAN_MODEL_DIR")
    if env_model_dir:
        candidates.append(normalize_path(Path(env_model_dir)))

    candidates.append(exe.parent / "models")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def build_ncnn_command(
    exe: Path,
    model_dir: Path | None,
    source: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        str(exe),
        "-i",
        str(source),
        "-o",
        str(output_path),
        "-n",
        args.model,
        "-s",
        str(args.scale),
        "-t",
        str(args.tile_size),
        "-f",
        args.format,
    ]
    if model_dir is not None:
        command.extend(["-m", str(model_dir)])
    if args.gpu_id is not None:
        command.extend(["-g", args.gpu_id])
    if args.threads is not None:
        command.extend(["-j", args.threads])
    return command


def upscale_ncnn_image(
    exe: Path,
    model_dir: Path | None,
    source: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> bool:
    if output_path.exists() and not args.overwrite:
        print(f"skipped_exists: {output_path}")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_ncnn_command(exe, model_dir, source, output_path, args)
    print(f"upscaling: {source} -> {output_path}")
    result = subprocess.run(command, check=False)
    if result.returncode == 0:
        return True

    print(f"failed: {source} (exit code {result.returncode})")
    return False


def run_upscale_loop(planned: list[tuple[Path, Path]], upscale_one, args: argparse.Namespace) -> None:
    success_count = 0
    failure_count = 0
    for source, output_path in planned:
        ok = upscale_one(source, output_path)
        if ok:
            success_count += 1
            continue

        failure_count += 1
        if not args.keep_going:
            raise SystemExit("Stopping after first failure. Use --keep-going to continue.")

    print(
        f"Done. Successful or skipped: {success_count}. "
        f"Failed: {failure_count}. Output root: {args.output_root}"
    )


def main() -> None:
    args = parse_args()
    args.input_root = normalize_path(args.input_root)
    args.output_root = normalize_path(args.output_root)
    args.env_file = normalize_path(args.env_file)
    load_env(args.env_file)

    extensions = normalize_extensions(args.extensions)
    images = find_images(args.input_root, extensions)
    if args.limit is not None:
        images = images[: args.limit]

    planned = [
        (source, output_path_for(source, args.input_root, args.output_root, args.format))
        for source in images
    ]

    print(f"Source images: {len(planned)}")
    print(f"Input root: {args.input_root}")
    print(f"Output root: {args.output_root}")
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model}")
    print(f"Scale: {args.scale}x")

    if args.dry_run:
        for source, output_path in planned:
            print(f"dry_run: {source} -> {output_path}")
        return

    if args.backend == "python":
        cv2, upsampler = make_python_upsampler(args)
        run_upscale_loop(
            planned,
            lambda source, output_path: upscale_python_image(
                cv2,
                upsampler,
                source,
                output_path,
                args,
            ),
            args,
        )
        return

    exe = resolve_executable(args.exe)
    model_dir = resolve_model_dir(args.model_dir, exe)
    print(f"Real-ESRGAN executable: {exe}")
    if model_dir is not None:
        print(f"Real-ESRGAN model directory: {model_dir}")
    else:
        print("Real-ESRGAN model directory: default from executable")

    run_upscale_loop(
        planned,
        lambda source, output_path: upscale_ncnn_image(
            exe,
            model_dir,
            source,
            output_path,
            args,
        ),
        args,
    )


if __name__ == "__main__":
    main()
