# cl-gbdt

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
| `cl-gbdt/regen` | The binding emitter (`src/regen/`). Development-only -- never appears in `cl-gbdt`'s, `cl-gbdt/lightgbm`'s, or `cl-gbdt/xgboost`'s dependency graph, so an ordinary user never needs it or its dependencies (`cffi/c2ffi`, `com.inuoe.jzon`) |
| `cl-gbdt/tests` | The Rove test suite |

Each backend system depends only on `cl-gbdt` and `cffi`, and each is loadable
independently, so the library works on a machine where only one of the two
shared libraries is present.

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
(rove:run-test 'cl-gbdt/tests::some-test-name)
```

## Running the functional tests

`cl-gbdt/functional-tests` is a separate system that calls the real LightGBM and
XGBoost shared libraries -- design doc section 12, layer 2. It exercises the raw FFI
directly: loading each library, reading its version, and running a small train/predict
round trip against a trivially separable dataset, asserting only that positive-label
predictions come back higher than negative-label ones.

Run `./tools/fetch-libs.sh` first to vendor the libraries into `vendor/`. Then:

```bash
ros run -- --non-interactive \
  --eval '(ql:quickload :cl-gbdt/functional-tests :silent t)' \
  --eval '(asdf:test-system :cl-gbdt/functional-tests)'
```

A backend whose library is missing skips rather than fails, naming
`./tools/fetch-libs.sh` in the skip message -- `vendor/` is git-ignored, so a fresh
clone legitimately has neither library yet. `CL_GBDT_LIGHTGBM_LIB` and
`CL_GBDT_XGBOOST_LIB` override discovery, for pointing the suite at a system-wide
install instead of the vendored copy.

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
   architecture, then runs `cl-gbdt.regen:emit-bindings` over its output, and
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

Not yet determined -- no `LICENSE` file exists in this repository and none is
referenced elsewhere in it.
