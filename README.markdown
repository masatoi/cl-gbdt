# cl-gbdt

[![tests](https://github.com/masatoi/cl-gbdt/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/masatoi/cl-gbdt/actions/workflows/test.yml)
[![lint](https://github.com/masatoi/cl-gbdt/actions/workflows/lint.yml/badge.svg?branch=master)](https://github.com/masatoi/cl-gbdt/actions/workflows/lint.yml)

A Common Lisp library that wraps two gradient boosting decision tree
implementations -- [LightGBM](https://github.com/microsoft/LightGBM) and
[XGBoost](https://github.com/dmlc/xgboost) -- behind a single high-level API.

Full design rationale lives in
`docs/superpowers/specs/2026-08-02-cl-gbdt-design.md` (not checked in; see that
file if you have it locally).

## Status

**Foundation only.** This branch establishes the condition hierarchy, the backend
registry and `open-backend` protocol, the zero-copy matrix marshalling, and a
binding generator that produces the raw CFFI declarations for both C APIs. It does
**not** yet call into either shared library: `make-dataset`, `train`, `predict`,
and the rest of the unified API are declared as generic functions with
docstrings but no methods. Loading `cl-gbdt` does not require either
`liblightgbm.so` or `libxgboost.so` to be installed.

## Systems

| System | Purpose |
|---|---|
| `cl-gbdt` | Core: package, condition hierarchy, matrix marshalling, backend registry and `open-backend` protocol, the unified API's generic functions |
| `cl-gbdt/lightgbm` | Generated CFFI bindings for the LightGBM C API (`src/lightgbm/c-api.lisp`) |
| `cl-gbdt/xgboost` | Generated CFFI bindings for the XGBoost C API (`src/xgboost/c-api.lisp`) |
| `cl-gbdt/regen` | The binding emitter (`src/regen/`). Development-only -- never appears in `cl-gbdt`'s, `cl-gbdt/lightgbm`'s, or `cl-gbdt/xgboost`'s dependency graph, so an ordinary user never needs it or its dependencies (`alexandria`, `com.inuoe.jzon`). `cffi/c2ffi` is *not* one of them -- it is a dependency of `tools/regen.lisp`, which quickloads it directly, not of the `cl-gbdt/regen` system itself |
| `cl-gbdt/tests` | The Rove test suite |

Each backend system currently depends only on `cffi`, by way of its generated
`c-api.lisp` (`cl-gbdt/lightgbm` depends on `cl-gbdt/src/lightgbm/c-api`, which
declares nothing but `cl` and `cffi`). Neither backend system depends on `cl-gbdt`
itself yet, since the unified API has no methods to call into them (see Status
above). Each backend is loadable independently, so the library works on a machine
where only one of the two shared libraries is present.

## Running the tests

```lisp
(ql:quickload :cl-gbdt/tests)
(asdf:test-system :cl-gbdt/tests)
```

or, from the shell (this project does not put `sbcl` on `PATH`; use `ros run`):

```bash
ros run -- --non-interactive \
  --eval '(ql:quickload :cl-gbdt/tests :silent t)' \
  --eval '(asdf:test-system :cl-gbdt/tests)'
```

No shared library is required: every test in this branch runs against the
generated bindings' source text, a mock backend, or in-process data structures
(design doc section 12, layer 1). Rove prints one line per test *suite*, not per
`deftest` form -- read the printed test names rather than trusting the "N tests
completed" count.

To run a single test from the REPL:

```lisp
(rove:run-test 'cl-gbdt/tests/backend::some-test-name)
```

## Running the functional tests

`cl-gbdt/tests/functional` is a separate system that calls the real LightGBM and
XGBoost shared libraries -- design doc section 12, layer 2. It exercises the raw FFI
directly: loading each library, reading its version, and running a small train/predict
round trip against a trivially separable dataset. Each round trip asserts more than
final prediction values -- every handle it creates is non-null, every output buffer's
length matches the row count, and the boosting iteration count reads back correctly --
and, as the property that ties the FFI plumbing together, that positive-label
predictions come back higher than negative-label ones.

This system is SBCL-only: both round trips pin arrays with `sb-sys` primitives
directly, unlike `src/data.lisp`'s `#+sbcl`-guarded idiom, and have no portable
fallback.

Run `./tools/fetch-libs.sh` first to vendor the libraries into `vendor/`. Then:

```bash
ros run -- --non-interactive \
  --eval '(ql:quickload :cl-gbdt/tests/functional :silent t)' \
  --eval '(asdf:test-system :cl-gbdt/tests/functional)'
```

A backend whose library is missing skips rather than fails, naming
`./tools/fetch-libs.sh` in the skip message -- `vendor/` is git-ignored, so a fresh
clone legitimately has neither library yet. `CL_GBDT_LIGHTGBM_LIB` and
`CL_GBDT_XGBOOST_LIB` override discovery, for pointing the suite at a system-wide
install instead of the vendored copy.

## Continuous integration

Two workflows, so the badges above report independently — a GitHub Actions badge covers a
whole workflow file, not a job:

- `.github/workflows/test.yml` runs both suites on Linux x86_64, Linux aarch64 and macOS
  aarch64. The matrix is the point: the bindings are generated on one machine and
  committed, so passing on that machine proves little. macOS is also the only place the
  `.dylib` discovery path is exercised at all.
- `.github/workflows/lint.yml` runs the static checks on one target, since nothing they
  look at varies by machine.

The logic lives in scripts rather than in the YAML, so the same checks run locally:

```bash
CL_GBDT_TEST_SYSTEM=cl-gbdt/tests ros run -- --non-interactive \
  --load tools/ci/run-tests.lisp          # layer 1
CL_GBDT_TEST_SYSTEM=cl-gbdt/tests/functional ros run -- --non-interactive \
  --load tools/ci/run-tests.lisp          # layer 2, needs ./tools/fetch-libs.sh first
ros run -- --non-interactive --load tools/ci/lint.lisp
ros run -- --non-interactive --load tools/ci/check-leaf-systems.lisp
```

Two things those scripts do that the plain commands above do not, and that CI needs:

- **They exit non-zero when a test fails.** `asdf:test-system` exits 0 regardless, so a job
  invoking it directly would be permanently green. `rove:run` returns false on failure and
  the script turns that into a status.
- **They check which foreign libraries were opened.** Layer 1 must open none; layer 2 must
  open both. A functional run where every test skipped for want of `vendor/` prints the same
  summary as one where every test passed, so the difference has to be asserted rather than
  inferred.

`tools/ci/lint.lisp` runs mallet *and* a column-width check, because mallet does not check
line length — a 132-column file passes it without comment. mallet is not in the Quicklisp
dist; the workflow clones it into `~/.roswell/local-projects/`, pinned, and a current ASDF is
installed first because Roswell ships 3.3.1, which predates `:local-nicknames` in
`uiop:define-package`.

**`tools/ci/check-leaf-systems.lisp` loads every leaf system alone, each in its own fresh
`ros run` subprocess.** This guards the principal risk this library's
`:package-inferred-system` layout carries: ASDF infers a file's dependencies *only* from its
`uiop:define-package` clauses — `:use`, `:import-from` (even one naming zero symbols), and
`:local-nicknames`. A file that calls, say, `cffi:defcfun` without naming `#:cffi` in one of
those clauses gets no declared dependency on CFFI; it loads correctly whenever something else
in the same image happened to load CFFI first, and breaks the moment load order shifts. Because
that failure is order-dependent, loading every leaf into one shared image — the obvious way to
"prove" they all load — proves nothing: the first file's load satisfies the next file's
undeclared dependency. Each leaf therefore gets a subprocess with a fresh image, where nothing
but what it declares is on hand.

The check also doubles as the enforcement mechanism for this project's naming convention: **a
leaf system's name is `cl-gbdt/` followed by its path from the repository root, extension
dropped** — `src/lightgbm/c-api.lisp` names `cl-gbdt/src/lightgbm/c-api`, and its
`uiop:define-package` form must name that same symbol. The checker derives its list of systems
to check from the filesystem (every `.lisp` file under `src/` and `tests/`, generated files
included) rather than from a hardcoded list, so a new file is picked up automatically the next
time CI runs — a contributor adding one only needs to follow the path-is-the-name rule and give
the package the `defpackage` clauses its file actually needs.

**On macOS the functional tests also need `brew install libomp`.** The macOS wheels link
against `@rpath/libomp.dylib` and, unlike the manylinux ones, do not vendor an OpenMP
runtime, so `dlopen` fails without it.

## Regenerating the bindings

`src/lightgbm/c-api.lisp` and `src/xgboost/c-api.lisp` are generated, checked in,
and architecture-independent by construction (design doc section 5.1). **You do
not need to regenerate them to use, build, or test this library** -- the normal
build reads them as ordinary `cl-source-file`s and depends on nothing but `cffi`.
Regeneration is a developer-only step, for when a header is updated or the
emitter itself changes.

Regeneration needs Docker (to run c2ffi in a pinned, reproducible LLVM 18
environment) and, for the header-fetch step, network access. Run these in order
from the repository root:

```bash
tools/fetch-headers.sh
docker build -f tools/Dockerfile.c2ffi -t cl-gbdt-c2ffi:llvm-18 tools/
ros run -- --non-interactive --load tools/regen.lisp
```

1. `tools/fetch-headers.sh` downloads the headers reachable from each backend's
   `c_api.h`, at the tags pinned in `ffi-spec/VERSIONS`, into `ffi-spec/`. No
   patching or hand-editing.
2. The `docker build` compiles c2ffi, pinned to an exact commit on its LLVM-18
   branch, into the `cl-gbdt-c2ffi:llvm-18` image. `tools/c2ffi.sh` invokes that
   image; `tools/regen.lisp` invokes `tools/c2ffi.sh`.
3. `tools/regen.lisp` runs c2ffi over the vendored headers for the local
   architecture, then runs `cl-gbdt/src/regen/all:emit-bindings` over its output, and
   validates the result (minimum function count, required symbols present, no
   architecture-dependent types) before replacing the committed file. A failed
   validation leaves the previously committed file untouched.

Expected output ends with two lines like:

```
==> LIGHTGBM
    99 functions, 12 constants -> src/lightgbm/c-api.lisp
==> XGBOOST
    78 functions, 0 constants -> src/xgboost/c-api.lisp
done
```

c2ffi generates for the local architecture only; design doc section 5.2 explains
why cross-architecture generation was tried and abandoned (it fails, and fails
silently -- see section 5.3 for how the emitter defends against that). The
architecture independence of the *output* does not depend on which architecture
generated it (section 5.1), so any one machine's regeneration is sufficient for
everyone.

`ffi-spec/ABI-BLACKLIST.md` records which C functions cl-gbdt must never call, and
why, independent of whether they happen to be emitted.

## License

MIT; see `LICENSE`.

The upstream C API headers vendored under `ffi-spec/` remain under their own terms --
LightGBM's are MIT, XGBoost's are Apache 2.0 -- and each retains its upstream copyright
notice, and both upstream licence texts are vendored verbatim under `LICENSES/` so that
recipients receive a copy rather than a link. `THIRD-PARTY-LICENSES.md` records what is
redistributed and under what. cl-gbdt does
not bundle either backend's compiled library; `tools/fetch-libs.sh` fetches those into the
git-ignored `vendor/` directory for development only.
