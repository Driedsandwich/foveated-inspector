#!/usr/bin/env bash
# Cross-build Python 3.12 Linux Lambda layers, without touching the current venv.
# Nothing is installed globally. This script downloads wheels when YOU run it.
# Native imports still require a later arm64/x86_64 Lambda smoke check.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
BUILD_DIR="$APP_DIR/build"
MODEL_DIR="$APP_DIR/models"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ARCH="all"
# The measured NumPy 2.3.3 aarch64 wheel needs glibc >=2.28. Lambda Python
# 3.12 uses AL2023 (glibc 2.34); keep 2.17 available as an explicit ablation.
GLIBC="2.28"
OPENCV_VERSION="5.0.0.93"
NUMPY_VERSION="2.3.3"
# Leave 5 MiB for function code. deploy.py rechecks the actual combined limit.
MAX_LAYER_BYTES=$((245 * 1024 * 1024))
STAGE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch) ARCH="$2"; shift 2 ;;
    --glibc) GLIBC="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --help|-h)
      printf '%s\n' 'Usage: bash aws/build_layer.sh [--arch arm64|x86_64|all] [--glibc 2.17|2.28 (default)] [--model-dir models]'
      exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
case "$ARCH" in arm64|x86_64|all) ;; *) printf 'Invalid architecture\n' >&2; exit 2 ;; esac
case "$GLIBC" in 2.17|2.28) ;; *) printf 'Invalid glibc target (use 2.17 or 2.28)\n' >&2; exit 2 ;; esac
command -v uv >/dev/null || { printf 'uv is required; this script does not install it.\n' >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || { printf 'A Python interpreter is required.\n' >&2; exit 1; }
mkdir -p "$BUILD_DIR"

cleanup_stage() {
  if [[ -n "$STAGE" ]]; then
    case "$STAGE" in
      "$BUILD_DIR"/_layer_*) rm -rf -- "$STAGE" ;;
      *) printf 'Refusing cleanup outside build staging directory\n' >&2; return 1 ;;
    esac
    STAGE=""
  fi
}
trap cleanup_stage EXIT

install_wheels() {
  local package="$1" platform="$2"
  UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_INSTALLS=1 UV_PYTHON_DOWNLOADS=never \
    uv pip install --target "$STAGE/python" --python-version 3.12 \
       --python-platform "$platform" --only-binary=:all: \
       "$package==$OPENCV_VERSION" "numpy==$NUMPY_VERSION" || {
         printf 'Binary-only wheel installation failed for %s. No source build was attempted.\n' "$platform" >&2
         printf 'The default glibc target is 2.28. Check binary wheel availability; native Lambda imports remain unverified.\n' >&2
         return 1
       }
}

prepare_layer() {
  local architecture="$1" package="$2" platform="$3"
  # Only disposable, newly-created staging trees are pruned.
  find "$STAGE/python" -depth -type d \( -name tests -o -name test -o -name __pycache__ \) -exec rm -rf -- {} +
  find "$STAGE/python" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
  "$PYTHON_BIN" - "$STAGE" "$MODEL_DIR" "$architecture" "$package" "$OPENCV_VERSION" "$NUMPY_VERSION" "$platform" "$GLIBC" <<'PY'
import json
from pathlib import Path
import shutil
import struct
import sys

stage, model_dir = Path(sys.argv[1]), Path(sys.argv[2])
architecture, package, opencv_version, numpy_version, platform, glibc = sys.argv[3:]
expected_machine = 183 if architecture == "arm64" else 62
for library in (stage / "python").rglob("*.so*"):
    if not library.is_file():
        continue
    with library.open("rb") as handle:
        header = handle.read(20)
    if header[:4] != b"\x7fELF" or len(header) < 20:
        raise SystemExit("A wheel contains a non-ELF native library")
    byte_order = "<" if header[5] == 1 else ">"
    if struct.unpack(byte_order + "H", header[18:20])[0] != expected_machine:
        raise SystemExit("Native wheel architecture does not match the requested Lambda architecture")

# Trained weights are optional in the layer; runtime can fetch validated S3
# models. Copy only the metadata and its six named forests when all are present.
model_files = []
metadata_path = model_dir / "metadata.json"
if metadata_path.is_file():
    metadata = json.loads(metadata_path.read_text())
    expected_models = {scale: {kind: f"scale_{scale}_{kind}.yml.gz" for kind in ("binary", "kind")}
                       for scale in ("0.25", "0.5", "1")}
    names = [metadata.get("models", {}).get(scale, {}).get(kind)
             for scale in ("0.25", "0.5", "1") for kind in ("binary", "kind")]
    if (metadata.get("calibration") == "sampled" and metadata.get("models") == expected_models
            and all(isinstance(name, str) and name and Path(name).name == name for name in names)
            and len(set(names)) == 6):
        candidates = [metadata_path, *(model_dir / name for name in names)]
        if (all(path.is_file() for path in candidates)
                and sum(path.stat().st_size for path in candidates) <= 32 * 1024 * 1024):
            (stage / "models").mkdir()
            for source in candidates:
                shutil.copyfile(source, stage / "models" / source.name)
                model_files.append(source.name)
manifest = {"python": "3.12", "architecture": architecture, "opencv_package": package,
            "opencv_version": opencv_version, "numpy_version": numpy_version,
            "platform": platform, "glibc": glibc,
            "model_files": model_files, "native_import_verified": False}
(stage / "layer_manifest.json").write_text(json.dumps(manifest, indent=2))
PY
}

layer_size() {
  "$PYTHON_BIN" - "$STAGE" <<'PY'
from pathlib import Path
import sys
print(sum(path.stat().st_size for path in Path(sys.argv[1]).rglob("*") if path.is_file()))
PY
}

build_architecture() {
  local architecture="$1" platform output package size
  if [[ "$architecture" == "arm64" ]]; then
    platform="aarch64-manylinux2014"
    output="$BUILD_DIR/layer.zip"
  else
    platform="x86_64-manylinux2014"
    output="$BUILD_DIR/layer_x86_64.zip"
  fi
  if [[ "$GLIBC" == "2.28" ]]; then
    if [[ "$architecture" == "arm64" ]]; then platform="aarch64-manylinux_2_28";
    else platform="x86_64-manylinux_2_28"; fi
  fi
  # Preserve an earlier finished artifact unless the operator moves it first.
  [[ ! -e "$output" ]] || { printf 'Output already exists: %s\n' "$output" >&2; return 1; }
  STAGE="$(mktemp -d "$BUILD_DIR/_layer_${architecture}.XXXXXX")"
  package="opencv-contrib-python-headless"
  install_wheels "$package" "$platform"
  prepare_layer "$architecture" "$package" "$platform"
  size="$(layer_size)"
  if (( size >= MAX_LAYER_BYTES )); then
    printf '%s layer is too large (%s bytes); retrying plain headless OpenCV.\n' "$architecture" "$size"
    cleanup_stage
    STAGE="$(mktemp -d "$BUILD_DIR/_layer_${architecture}.XXXXXX")"
    package="opencv-python-headless"
    install_wheels "$package" "$platform"
    prepare_layer "$architecture" "$package" "$platform"
    size="$(layer_size)"
  fi
  (( size < MAX_LAYER_BYTES )) || { printf 'Layer still exceeds the size allowance.\n' >&2; return 1; }
  "$PYTHON_BIN" - "$STAGE" "$output" <<'PY'
from pathlib import Path
import shutil
import sys
import zipfile

stage, destination = map(Path, sys.argv[1:])
temporary = destination.with_suffix(".zip.tmp")
with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for source in sorted(stage.rglob("*")):
        if source.is_file():
            entry = zipfile.ZipInfo(source.relative_to(stage).as_posix(), (1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            with source.open("rb") as incoming, archive.open(entry, "w", force_zip64=True) as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
temporary.replace(destination)
PY
  printf 'Built %s (%s uncompressed bytes, %s); runtime import remains unverified.\n' "$output" "$size" "$package"
  cleanup_stage
}

if [[ "$ARCH" == "all" ]]; then
  build_architecture arm64
  build_architecture x86_64
else
  build_architecture "$ARCH"
fi
