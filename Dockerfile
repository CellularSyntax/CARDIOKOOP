# CARDIOKOOP — reproduction image for the manuscript tables (Array, revision 3).
#
#   docker build -t cardiokoop .
#   docker run --rm -v "$PWD/out:/workspace/out" cardiokoop          # regenerate + compare
#   docker run --rm -v "$PWD/out:/workspace/out" cardiokoop bash     # interactive shell
#
# The image contains the pinned environment (environment/requirements-pinned.txt, CPU build of
# torch 2.6.0), the package, the frozen checkpoints and the committed results.  The seed-42
# splits (*.csv, Git-LFS) are NOT baked in; scripts/reproduce.sh downloads them from the Zenodo
# dataset record 10.5281/zenodo.21163127 at run time and verifies their MD5 checksums.
FROM python:3.11-slim

ARG VERSION=dev
LABEL org.opencontainers.image.source="https://github.com/CellularSyntax/CARDIOKOOP" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.title="CARDIOKOOP" \
      org.opencontainers.image.description="Control-aware Koopman operator models for real-time hemodynamic prediction — pinned environment that reproduces the manuscript Tables 3-5 and statistics from the frozen checkpoints (CPU, float64 rollout)."

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    REPRO_OUT=/workspace/out

# curl (Zenodo download), git + git-lfs (fallback), build-essential (fastdtw ships only an
# sdist with a C++ extension; the compiler is removed again after the pip installs).
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git git-lfs \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 1) pinned environment: CPU wheel of torch 2.6.0 from the PyTorch index, then every other pin
#    from environment/requirements-pinned.txt (torch==2.6.0 there is satisfied by 2.6.0+cpu).
COPY environment/requirements-pinned.txt environment/requirements-pinned.txt
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && pip install --upgrade pip \
 && pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r environment/requirements-pinned.txt \
 && apt-get purge -y --auto-remove build-essential \
 && rm -rf /var/lib/apt/lists/* /root/.cache

# 2) the package (non-editable; dependencies already pinned above)
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps .

# 3) the rest of the repository (scripts, checkpoints, committed results, LFS pointers)
COPY . .
RUN chmod +x scripts/reproduce.sh && mkdir -p /workspace/out

VOLUME ["/workspace/out"]
CMD ["bash", "scripts/reproduce.sh"]
