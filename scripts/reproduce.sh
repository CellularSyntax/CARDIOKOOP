#!/usr/bin/env bash
# CARDIOKOOP — one-command reproduction of the manuscript Tables 3, 4, 5 and the statistics
# paragraph (seed-42 test1 split, frozen checkpoint), followed by a comparison against the
# committed results/revision3 at manuscript rounding.  This is the default CMD of the
# Docker image and the check that runs in GitHub Actions (.github/workflows/reproduce.yml).
#
#   (a) make sure the seed-42 splits are real files (the repository stores *.csv via Git-LFS;
#       tarballs and plain checkouts only contain pointer files).  Missing / pointer files are
#       downloaded from the Zenodo dataset record 10.5281/zenodo.21163127 (record 21163128) and
#       verified against the MD5 checksums published on that record.  If Zenodo is unreachable
#       and the repository is a writable git checkout, `git lfs pull` is tried instead.
#   (b) python scripts/revision3/export_manuscript_tables.py --out-dir $REPRO_OUT
#   (c) python scripts/revision3/compare_results.py  (exit 1 on any difference beyond rounding)
#
# Environment variables
#   REPRO_REPO          repository root            (default: directory above this script)
#   REPRO_OUT           output directory           (default: /workspace/out, or $REPRO_REPO/out
#                                                   if /workspace does not exist)
#   CARDIOKOOP_DATA_DIR directory for the splits   (default: $REPRO_REPO/data if writable,
#                                                   otherwise $REPRO_OUT/../cardiokoop-data or /tmp)
#   REPRO_THREADS       torch CPU threads          (default: torch default = all cores)
#   REPRO_DTYPE         float64 (default) | float32
#   REPRO_STRICT        1 -> off-by-one rounding differences also fail the check
#   REPRO_SKIP_COMPARE  1 -> only regenerate, do not compare
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPRO_REPO:-$(dirname "$HERE")}"
if [ -z "${REPRO_OUT:-}" ]; then
  if [ -d /workspace ]; then REPRO_OUT=/workspace/out; else REPRO_OUT="$REPO/out"; fi
fi
OUT="$REPRO_OUT"
mkdir -p "$OUT"

ZENODO_RECORD=21163128
ZENODO_BASE="https://zenodo.org/api/records/${ZENODO_RECORD}/files"
# name  md5 (from https://zenodo.org/api/records/21163128/files)  size [bytes]
REQUIRED_FILES=(
  "csv_data_500_12sigs_test1_x.csv  9d786f9e7b6ab2f487b463e68c3a5d28  23112605"
  "csv_data_500_12sigs_test1_u.csv  1d5bd1c25c18ef33264dbc701852e61f  1950000"
  "csv_data_500_12sigs_train1_x.csv 990d37f1b8d7ce7968f64a624b8577a2  184835313"
  "normalization_mean.npy           1dd5f81e5b07eb6c6b48ce3fcf53087b  224"
  "normalization_std.npy            7b2fd9dbbe470d76f8aea83d4b7812ce  224"
)
# (train1_u.csv, val1_*.csv and cardiovascular_parameter_space.csv are not needed by the
#  revision-3 export script and are therefore not downloaded.)

log() { printf '[reproduce] %s\n' "$*"; }

md5_of() {
  if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | cut -d' ' -f1
  else md5 -q "$1"; fi
}

is_pointer_or_missing() {   # true if the file is absent or a Git-LFS pointer
  local f="$1"
  [ -f "$f" ] || return 0
  [ "$(stat -c %s "$f" 2>/dev/null || stat -f %z "$f")" -lt 1024 ] && \
    head -c 40 "$f" | grep -q '^version https://git-lfs' && return 0
  return 1
}

# ── data directory ─────────────────────────────────────────────────────────────────────────
if [ -z "${CARDIOKOOP_DATA_DIR:-}" ]; then
  if [ -w "$REPO/data" ]; then
    CARDIOKOOP_DATA_DIR="$REPO/data"
  else
    CARDIOKOOP_DATA_DIR="${TMPDIR:-/tmp}/cardiokoop-data"
    log "$REPO/data is read-only -> using $CARDIOKOOP_DATA_DIR for the splits"
  fi
fi
export CARDIOKOOP_DATA_DIR
mkdir -p "$CARDIOKOOP_DATA_DIR"

download() {   # download <name> to <dest> with retries
  local name="$1" dest="$2"
  curl --fail --location --silent --show-error --retry 5 --retry-delay 15 --retry-all-errors \
       --connect-timeout 30 --max-time 1800 -o "$dest.part" "$ZENODO_BASE/$name/content" \
    && mv "$dest.part" "$dest"
}

lfs_fallback() {   # try `git lfs pull` for one file (only in a writable git checkout)
  local name="$1" dest="$2"
  if [ -d "$REPO/.git" ] && [ -w "$REPO/data" ] && command -v git >/dev/null && git -C "$REPO" lfs version >/dev/null 2>&1; then
    log "Zenodo unreachable -> trying: git lfs pull --include=data/$name"
    git -C "$REPO" lfs pull --include="data/$name" || return 1
    [ "$REPO/data/$name" = "$dest" ] || cp "$REPO/data/$name" "$dest"
    return 0
  fi
  return 1
}

log "repository : $REPO"
log "output dir : $OUT"
log "data dir   : $CARDIOKOOP_DATA_DIR"
t_start=$(date +%s)

for entry in "${REQUIRED_FILES[@]}"; do
  read -r name md5 size <<<"$entry"
  src="$REPO/data/$name"; dest="$CARDIOKOOP_DATA_DIR/$name"
  if [ -f "$dest" ] && ! is_pointer_or_missing "$dest"; then
    :                                    # already present in the data dir (e.g. cached)
  elif ! is_pointer_or_missing "$src"; then
    [ "$src" = "$dest" ] || cp "$src" "$dest"   # real file in the repository
  else
    log "downloading $name ($size bytes) from Zenodo record $ZENODO_RECORD"
    if ! download "$name" "$dest"; then
      lfs_fallback "$name" "$dest" || { log "ERROR: could not obtain $name (Zenodo download and git-lfs fallback failed)"; exit 3; }
    fi
  fi
  got="$(md5_of "$dest")"
  if [ "$got" != "$md5" ]; then
    log "ERROR: MD5 mismatch for $name: expected $md5, got $got"
    exit 3
  fi
  log "verified   : $name  md5=$got"
done
log "data ready in $(( $(date +%s) - t_start )) s"

# ── (b) regenerate the manuscript tables ───────────────────────────────────────────────────
ARGS=(--out-dir "$OUT" --dtype "${REPRO_DTYPE:-float64}")
[ -n "${REPRO_THREADS:-}" ] && ARGS+=(--threads "$REPRO_THREADS")
log "running: python scripts/revision3/export_manuscript_tables.py ${ARGS[*]}"
t0=$(date +%s)
( cd "$REPO" && python scripts/revision3/export_manuscript_tables.py "${ARGS[@]}" ) 2>&1 | tee "$OUT/export_manuscript_tables.log"
log "export finished in $(( $(date +%s) - t0 )) s"

# ── (c) compare with the committed results at manuscript rounding ──────────────────────────
if [ "${REPRO_SKIP_COMPARE:-0}" = "1" ]; then
  log "REPRO_SKIP_COMPARE=1 -> skipping comparison"; exit 0
fi
CMP=(--committed "$REPO/results/revision3" --fresh "$OUT" --report "$OUT/compare_report.md")
[ "${REPRO_STRICT:-0}" = "1" ] && CMP+=(--strict)
log "running: python scripts/revision3/compare_results.py ${CMP[*]}"
set +e
( cd "$REPO" && python scripts/revision3/compare_results.py "${CMP[@]}" ) 2>&1 | tee "$OUT/compare_results.log"
rc=${PIPESTATUS[0]}
set -e
log "total wall time $(( $(date +%s) - t_start )) s; compare exit code $rc"
exit "$rc"
