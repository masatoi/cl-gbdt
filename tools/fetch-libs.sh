#!/usr/bin/env bash
# Fetch the LightGBM / XGBoost shared libraries for development and testing.
# This only extracts binaries from the upstream official Python wheels; nothing is built.
#
# The wheel's internal directory layout is preserved under vendor/, because the
# libraries carry an RPATH relative to it. libxgboost.so has
# RPATH=$ORIGIN/../../xgboost.libs and NEEDs a hash-suffixed libgomp that exists
# only inside the wheel; flattening the layout makes it impossible to dlopen.
set -euo pipefail

LIGHTGBM_VERSION="${LIGHTGBM_VERSION:-4.7.0}"
XGBOOST_VERSION="${XGBOOST_VERSION:-3.3.0}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="${repo_root}/vendor"
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

  local extracted="${work}/${pkg}/extracted"
  unzip -q -o "${wheel}" -d "${extracted}"

  # Locate the library as a path relative to the wheel root, so it can be copied
  # back at the same relative position.
  local lib
  lib="$(cd "${extracted}" && find . -name "${pattern}" -print -quit)"
  if [ -z "${lib}" ]; then
    echo "error: ${pattern} not found inside ${wheel}" >&2
    echo "shared objects present in the wheel:" >&2
    find "${extracted}" \( -name '*.so' -o -name '*.dylib' \) >&2
    return 1
  fi
  lib="${lib#./}"

  mkdir -p "${dest}/$(dirname "${lib}")"
  cp "${extracted}/${lib}" "${dest}/${lib}"
  echo "    ${lib}"

  # manylinux wheels place vendored dependencies in a top-level <pkg>.libs/
  # directory that the RPATH points at. Copy it at the same relative position.
  local libs_dir
  for libs_dir in "${extracted}"/*.libs; do
    [ -d "${libs_dir}" ] || continue
    cp -R "${libs_dir}" "${dest}/"
    echo "    $(basename "${libs_dir}")/ (vendored dependencies)"
  done
}

fetch lightgbm "${LIGHTGBM_VERSION}" 'lib_lightgbm.*'
fetch xgboost  "${XGBOOST_VERSION}"  'libxgboost.*'

echo
echo "fetched into ${dest}:"
find "${dest}" \( -name '*.so' -o -name '*.so.*' -o -name '*.dylib' \) | sort
