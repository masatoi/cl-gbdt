#!/usr/bin/env bash
# Run c2ffi through Docker. To cffi/c2ffi this looks like an ordinary c2ffi binary.
#
# cffi/c2ffi passes both repository paths and temporary files created by
# uiop:with-temporary-file. Mount both so they resolve at the same absolute path
# inside the container.
#
# The working directory is load-bearing. cffi/c2ffi forwards :sys-include-paths
# to c2ffi verbatim, without absolutising them (see its c2ffi.lisp,
# generate-spec-using-c2ffi), and cl-gbdt.asd supplies a repository-relative one.
# Docker silently creates an empty directory for an unmounted -w, so a wrong
# working directory would surface as a confusing "header not found" instead of a
# clear error. Refuse to run in that case.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
tmp_dir="$(cd "${TMPDIR:-/tmp}" && pwd -P)"
cwd="$(pwd -P)"

case "${cwd}/" in
  "${repo_root}/"*|"${tmp_dir}/"*) ;;
  *)
    echo "error: c2ffi.sh must run from inside ${repo_root} or ${tmp_dir}." >&2
    echo "       Current directory: ${cwd}" >&2
    echo "       Only those paths are mounted into the container, so relative" >&2
    echo "       arguments such as --sys-include would not resolve." >&2
    exit 1
    ;;
esac

exec docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "${repo_root}:${repo_root}" \
  -v "${tmp_dir}:${tmp_dir}" \
  -w "${cwd}" \
  cl-gbdt-c2ffi:llvm-18 "$@"
