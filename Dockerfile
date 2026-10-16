# =============================================================================
# ftype_model_development — dev image (thin)
#
# The host's Docker disk (/) is nearly full, so the image only carries the
# micromamba binary. The Python env (~10 GB: Python 3.12 + GDAL + TF/CUDA)
# lives on /fs2 inside the bind-mounted repo at /workspace/.venv-docker and is
# created once with:
#
#     docker compose run --rm dev bash scripts/docker_setup_env.sh
# =============================================================================
FROM mambaorg/micromamba:2.0.5

# Run as the host user (see compose); keep every cache/tmp on /fs2, not in the
# container layer on /.
ENV HOME=/tmp/home \
    ENV_PREFIX=/workspace/.venv-docker \
    MAMBA_ROOT_PREFIX=/workspace/.cache/mamba-root \
    CONDA_PKGS_DIRS=/workspace/.cache/mamba-pkgs \
    PIP_CACHE_DIR=/workspace/.cache/pip \
    TMPDIR=/workspace/.cache/tmp \
    PATH=/workspace/.venv-docker/bin:/opt/conda/bin:/usr/local/bin:/usr/bin:/bin \
    NV=/workspace/.venv-docker/lib/python3.12/site-packages/nvidia
# Env libs first: TF otherwise pulls in the system libstdc++ (too old for
# PROJ/rasterio) and can't find the pip CUDA libs (e.g. libcusolver.so.11).
ENV LD_LIBRARY_PATH=/workspace/.venv-docker/lib:$NV/cublas/lib:$NV/cuda_cupti/lib:$NV/cuda_nvrtc/lib:$NV/cuda_runtime/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/curand/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nccl/lib:$NV/nvjitlink/lib
ENV \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=1

WORKDIR /workspace
ENTRYPOINT []
CMD ["bash"]
