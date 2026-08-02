#!/usr/bin/env bash
# Fetch the C API headers from upstream release tags.
# Nothing is edited by hand. The fetched versions are recorded in ffi-spec/VERSIONS.
set -euo pipefail

LIGHTGBM_TAG="${LIGHTGBM_TAG:-v4.7.0}"
XGBOOST_TAG="${XGBOOST_TAG:-v3.3.0}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
spec_root="${repo_root}/ffi-spec"

# LightGBM: c_api.h includes <LightGBM/export.h> and <LightGBM/arrow.h>,
# so fetch the whole include/LightGBM/ tree.
lgbm_dest="${spec_root}/lightgbm/include"
rm -rf "${lgbm_dest}/LightGBM"
mkdir -p "${lgbm_dest}"
echo "==> fetching include/LightGBM/ from LightGBM ${LIGHTGBM_TAG}"
curl -sSL "https://github.com/microsoft/LightGBM/archive/refs/tags/${LIGHTGBM_TAG}.tar.gz" \
  | tar -xz --strip-components=2 -C "${lgbm_dest}" \
        "LightGBM-${LIGHTGBM_TAG#v}/include/LightGBM"

# XGBoost: c_api.h depends only on standard headers, so fetch it alone.
xgb_dest="${spec_root}/xgboost/include/xgboost"
mkdir -p "${xgb_dest}"
echo "==> fetching c_api.h from XGBoost ${XGBOOST_TAG}"
curl -sSL -o "${xgb_dest}/c_api.h" \
  "https://raw.githubusercontent.com/dmlc/xgboost/${XGBOOST_TAG}/include/xgboost/c_api.h"

cat > "${spec_root}/VERSIONS" <<EOF
lightgbm ${LIGHTGBM_TAG}
xgboost ${XGBOOST_TAG}
EOF

echo
echo "fetched headers:"
find "${spec_root}" -name '*.h' | sort
echo
cat "${spec_root}/VERSIONS"
