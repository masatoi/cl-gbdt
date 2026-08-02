#!/usr/bin/env bash
# Fetch the LightGBM / XGBoost shared libraries for development and testing.
# This only extracts binaries from the upstream official Python wheels; nothing is built.
set -euo pipefail

LIGHTGBM_VERSION="${LIGHTGBM_VERSION:-4.7.0}"
XGBOOST_VERSION="${XGBOOST_VERSION:-3.3.0}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="${repo_root}/vendor/lib"
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

mkdir -p "${dest}"

fetch() {
  local pkg="$1" version="$2" pattern="$3"
  echo "==> downloading wheel for ${pkg}==${version}"
  python3 -m pip download "${pkg}==${version}" \
    --no-deps --only-binary=:all: -d "${work}/${pkg}" >/dev/null

  local wheel
  wheel="$(find "${work}/${pkg}" -name '*.whl' -print -quit)"
  if [ -z "${wheel}" ]; then
    echo "error: no wheel downloaded for ${pkg}" >&2
    return 1
  fi

  unzip -q -o "${wheel}" -d "${work}/${pkg}/extracted"

  local lib
  lib="$(find "${work}/${pkg}/extracted" -name "${pattern}" -print -quit)"
  if [ -z "${lib}" ]; then
    echo "error: ${pattern} not found inside ${wheel}" >&2
    echo "shared objects present in the wheel:" >&2
    find "${work}/${pkg}/extracted" \( -name '*.so' -o -name '*.dylib' \) >&2
    return 1
  fi

  cp "${lib}" "${dest}/"
  echo "    $(basename "${lib}") -> ${dest}/"

  # Also copy any vendored dependencies (manylinux wheels put these in *.libs/)
  local libs_dir
  libs_dir="$(find "${work}/${pkg}/extracted" -type d -name '*.libs' -print -quit)"
  if [ -n "${libs_dir}" ]; then
    for dep in "${libs_dir}"/*.so*; do
      if [ -f "${dep}" ]; then
        cp "${dep}" "${dest}/"
        echo "    $(basename "${dep}") -> ${dest}/"
      fi
    done
  fi
}

fetch lightgbm "${LIGHTGBM_VERSION}" 'lib_lightgbm.*'
fetch xgboost  "${XGBOOST_VERSION}"  'libxgboost.*'

echo
echo "fetched libraries:"
ls -la "${dest}"
