#!/usr/bin/env bash
# Report the latest stable LightGBM and XGBoost releases, next to what ffi-spec/VERSIONS
# pins, as KEY=value lines on stdout.
#
# PyPI rather than the GitHub release feeds, because PyPI is where tools/fetch-libs.sh
# actually obtains the shared libraries. The question worth answering is "what is the
# latest release we could vendor", and a GitHub tag with no wheel is not an answer to it.
#
# .github/workflows/upstream.yml appends this output straight to $GITHUB_OUTPUT. Running it
# by hand prints the same lines, which is the point: every decision that workflow makes is
# a string comparison on output you can reproduce in a second.
#
# EXIT STATUS. Non-zero only when this script CANNOT LOOK -- a fetch failure, a version in
# a shape it cannot parse, or an ffi-spec/VERSIONS line it cannot parse. It never exits
# non-zero merely because a pin is behind: that is a finding, reported as behind=true, and
# the caller decides what it costs. The first half of that rule is tools/check-upstream.lisp's,
# whose own header records the incident that established it -- a detector that cannot look
# must not report "nothing changed".
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
versions_file="${repo_root}/ffi-spec/VERSIONS"

# The PyPI project names. Overridable ONLY so the cannot-look path can be exercised by
# pointing at a name PyPI does not have; nothing in CI sets these.
lightgbm_pkg="${CHECK_PKG_LIGHTGBM:-lightgbm}"
xgboost_pkg="${CHECK_PKG_XGBOOST:-xgboost}"

# The one version shape this project accepts. A pre-release ("3.5.0rc1"), a post-release
# ("3.4.1.post1"), or a change in PyPI's JSON must fail rather than be rounded to something
# plausible: fetch-libs.sh would go on to download whatever it was handed.
stable_re='^[0-9]+\.[0-9]+\.[0-9]+$'

latest_on_pypi() {
  local pkg="$1"
  # Three distinct cannot-look causes get three distinct messages. Folding them into one
  # `curl | python3' pipeline was tried and rejected: a 404 came out as "PyPI reports
  # <pkg> , which is not MAJOR.MINOR.PATCH", blaming the version shape for what was
  # actually a missing package, and leaking a Python traceback into the log ahead of it.
  #
  # Each fetch is guarded with `if ! var="$(...)"' rather than left to `set -e'. An
  # assignment from a failing command substitution does not reliably abort under `set -e'
  # inside a function -- measured, not assumed: that is exactly how the 404 above reached
  # the version check instead of stopping at the fetch.
  local body version
  if ! body="$(curl -fsS "https://pypi.org/pypi/${pkg}/json")"; then
    echo "error: could not fetch https://pypi.org/pypi/${pkg}/json -- curl said so above." >&2
    return 1
  fi
  if ! version="$(printf '%s' "${body}" | python3 -c \
       'import json,sys; print(json.load(sys.stdin)["info"]["version"])' 2>/dev/null)"; then
    echo "error: PyPI's JSON for ${pkg} carries no info.version -- its shape has changed." >&2
    return 1
  fi
  if [[ ! "${version}" =~ ${stable_re} ]]; then
    echo "error: PyPI reports ${pkg} ${version}, which is not MAJOR.MINOR.PATCH." >&2
    echo "       Refusing to guess. Check https://pypi.org/project/${pkg}/ by hand." >&2
    return 1
  fi
  printf '%s' "${version}"
}

pinned_in_versions_file() {
  local key="$1"
  local version
  version="$(sed -n "s/^${key} v\(.*\)\$/\1/p" "${versions_file}")"
  if [[ ! "${version}" =~ ${stable_re} ]]; then
    echo "error: ${versions_file} has no '${key} v<MAJOR.MINOR.PATCH>' line." >&2
    echo "       Found: '${version}'" >&2
    return 1
  fi
  printf '%s' "${version}"
}

lightgbm_latest="$(latest_on_pypi "${lightgbm_pkg}")"
xgboost_latest="$(latest_on_pypi "${xgboost_pkg}")"
lightgbm_pinned="$(pinned_in_versions_file lightgbm)"
xgboost_pinned="$(pinned_in_versions_file xgboost)"

# Inequality, not "less than". A pin AHEAD of PyPI's latest -- a yanked release, or a typo
# in ffi-spec/VERSIONS -- is also a state that needs a human, and comparing versions
# numerically would quietly call it fine.
lightgbm_behind=false
xgboost_behind=false
[ "${lightgbm_latest}" = "${lightgbm_pinned}" ] || lightgbm_behind=true
[ "${xgboost_latest}" = "${xgboost_pinned}" ] || xgboost_behind=true

behind=false
if [ "${lightgbm_behind}" = true ] || [ "${xgboost_behind}" = true ]; then
  behind=true
fi

# Only a backend that is actually behind becomes a leg. Running the current backend's
# pinned combination again would re-test exactly what test.yml's `test' job already covers,
# for eight minutes.
#
# Each leg pairs one backend's latest with the OTHER backend's PIN, never with the other's
# latest. The two libraries are independent shared objects with no interaction between
# them, so a full cross product costs twice as much for a combination there is no reason to
# think behaves differently from either axis alone -- the argument
# .github/workflows/test.yml:148-152 already makes for version-matrix.
legs=""
if [ "${lightgbm_behind}" = true ]; then
  legs="${legs}{\"name\":\"lightgbm-${lightgbm_latest}\""
  legs="${legs},\"lightgbm_version\":\"${lightgbm_latest}\""
  legs="${legs},\"xgboost_version\":\"${xgboost_pinned}\"},"
fi
if [ "${xgboost_behind}" = true ]; then
  legs="${legs}{\"name\":\"xgboost-${xgboost_latest}\""
  legs="${legs},\"lightgbm_version\":\"${lightgbm_pinned}\""
  legs="${legs},\"xgboost_version\":\"${xgboost_latest}\"},"
fi

cat <<EOF
lightgbm_latest=${lightgbm_latest}
xgboost_latest=${xgboost_latest}
lightgbm_pinned=${lightgbm_pinned}
xgboost_pinned=${xgboost_pinned}
lightgbm_behind=${lightgbm_behind}
xgboost_behind=${xgboost_behind}
behind=${behind}
matrix={"include":[${legs%,}]}
EOF
