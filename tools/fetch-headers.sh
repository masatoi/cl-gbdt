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

LIGHTGBM_TAG="${LIGHTGBM_TAG:-v4.7.0}"
XGBOOST_TAG="${XGBOOST_TAG:-v3.3.0}"

# Headers reachable from LightGBM/c_api.h. Verified with:
#   grep -E '^#include <LightGBM/' c_api.h arrow.h export.h
LIGHTGBM_HEADERS=(c_api.h arrow.h export.h)

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
spec_root="${repo_root}/ffi-spec"
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

lgbm_dest="${spec_root}/lightgbm/include/LightGBM"
rm -rf "${spec_root}/lightgbm/include"
mkdir -p "${lgbm_dest}"
echo "==> fetching LightGBM ${LIGHTGBM_TAG}"
curl -sSL "https://github.com/microsoft/LightGBM/archive/refs/tags/${LIGHTGBM_TAG}.tar.gz" \
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
