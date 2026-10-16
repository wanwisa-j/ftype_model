#!/usr/bin/env bash
# Create the project env on /fs2 (inside the container: /workspace/.venv-docker).
#   docker compose run --rm dev bash scripts/docker_setup_env.sh          # create
#   docker compose run --rm dev bash scripts/docker_setup_env.sh --force  # recreate
set -euo pipefail

mkdir -p "$HOME" "$TMPDIR" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$MAMBA_ROOT_PREFIX"

if [[ -d "$ENV_PREFIX" && "${1:-}" != "--force" ]]; then
    echo "Env already exists at $ENV_PREFIX (use --force to recreate)"
else
    rm -rf "$ENV_PREFIX"
    micromamba create -y -p "$ENV_PREFIX" -f /workspace/environment.yml
fi

# Plain `tensorflow` (environment.yml) ships without CUDA/cuDNN;
# the [and-cuda] extra adds the nvidia-* wheels so the GPU is usable.
"$ENV_PREFIX/bin/pip" install "tensorflow[and-cuda]==2.21.0"

"$ENV_PREFIX/bin/python" - <<'EOF'
import sys, tensorflow as tf, rasterio, geopandas
from osgeo import gdal
print("python", sys.version.split()[0], "| tf", tf.__version__, "| gdal", gdal.__version__,
      "| rasterio", rasterio.__version__, "| geopandas", geopandas.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
EOF
