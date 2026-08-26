#!/usr/bin/env bash
# Fetch the C API headers from upstream release tags.
#
# Only the headers reachable from each c_api.h are vendored. LightGBM's include
# tree is ~16k lines of C++ internals; c_api.h reaches just three files, and
# vendoring the rest would bury plan 4's upstream-diff job in irrelevant churn.
#
# Nothing is edited by hand -- the reference implementations patched c_api.h with
# the procedure recorded only as a prose comment, which is the reproducibility hole
# this project exists to avoid.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
spec_root="${repo_root}/ffi-spec"
versions_file="${spec_root}/VERSIONS"

# Read the tag pinned for KEY in ffi-spec/VERSIONS, in the v-prefixed form this script
# wants (a git tag / URL path segment). tools/latest-upstream.sh's
# pinned_in_versions_file reads the same file with the same sed shape but strips the
# "v", since it wants a bare version to compare -- mind the difference. Fails loudly,
# naming the file, rather than falling back to a default: a silent fallback is exactly
# how the old hardcoded defaults drifted from the pin in the first place.
pinned_tag() {
  local key="$1"
  local tag
  tag="$(sed -n "s/^${key} \(v[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\)\$/\1/p" "${versions_file}")"
  if [ -z "${tag}" ]; then
    echo "error: ${versions_file} has no '${key} v<MAJOR.MINOR.PATCH>' line." >&2
    return 1
  fi
  printf '%s' "${tag}"
}

# The pin lives in ffi-spec/VERSIONS; these defaults read it back so the tags fetched
# here cannot drift from it. An environment override is how a version bump is
# performed: `XGBOOST_TAG=v3.4.2 tools/fetch-headers.sh` fetches that tag instead, and
# below, writes it into ffi-spec/VERSIONS as the new pin.
LIGHTGBM_TAG="${LIGHTGBM_TAG:-$(pinned_tag lightgbm)}"
XGBOOST_TAG="${XGBOOST_TAG:-$(pinned_tag xgboost)}"

# Headers reachable from LightGBM/c_api.h. Verified with:
#   grep -E '^#include <LightGBM/' c_api.h arrow.h export.h
LIGHTGBM_HEADERS=(c_api.h arrow.h export.h)

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

lgbm_dest="${spec_root}/lightgbm/include/LightGBM"
rm -rf "${spec_root}/lightgbm/include"
mkdir -p "${lgbm_dest}"
echo "==> fetching LightGBM ${LIGHTGBM_TAG}"
curl -sSL "https://github.com/lightgbm-org/LightGBM/archive/refs/tags/${LIGHTGBM_TAG}.tar.gz" \
  | tar -xz -C "${work}" "LightGBM-${LIGHTGBM_TAG#v}/include/LightGBM"
for header in "${LIGHTGBM_HEADERS[@]}"; do
  src="${work}/LightGBM-${LIGHTGBM_TAG#v}/include/LightGBM/${header}"
  if [ ! -f "${src}" ]; then
    echo "error: ${header} is not present in LightGBM ${LIGHTGBM_TAG}" >&2
    exit 1
  fi
  cp "${src}" "${lgbm_dest}/"
  echo "    LightGBM/${header}"
done

# XGBoost: c_api.h depends only on standard headers, so fetch it alone.
xgb_dest="${spec_root}/xgboost/include/xgboost"
rm -rf "${spec_root}/xgboost/include"
mkdir -p "${xgb_dest}"
echo "==> fetching XGBoost ${XGBOOST_TAG}"
curl -sSL --fail -o "${xgb_dest}/c_api.h" \
  "https://raw.githubusercontent.com/dmlc/xgboost/${XGBOOST_TAG}/include/xgboost/c_api.h"
echo "    xgboost/c_api.h"

cat > "${spec_root}/VERSIONS" <<EOF
lightgbm ${LIGHTGBM_TAG}
xgboost ${XGBOOST_TAG}
EOF

echo
echo "vendored headers:"
find "${spec_root}" -name '*.h' | sort
echo
cat "${spec_root}/VERSIONS"
