#!/usr/bin/env bash
# Run c2ffi through Docker. To cffi/c2ffi this looks like an ordinary c2ffi binary.
#
# cffi/c2ffi passes both repository paths and temporary files created by
# uiop:with-temporary-file. Mount both so they resolve at the same absolute path
# inside the container.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="${TMPDIR:-/tmp}"

exec docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "${repo_root}:${repo_root}" \
  -v "${tmp_dir}:${tmp_dir}" \
  -w "$(pwd)" \
  cl-gbdt-c2ffi:llvm-18 "$@"
