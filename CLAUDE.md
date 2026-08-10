# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Agent Guidelines

@prompts/repl-driven-development.md
@prompts/common-lisp-expert.md

### How those prompts apply here

Both prompts were imported verbatim from the `cl-mcp` project and describe *its*
environment, not this one. Where they conflict with this file, **this file wins**:

- **cl-mcp tools require a running server.** `repl-driven-development.md` assumes the MCP
  tools (`repl-eval`, `lisp-edit-form`, `clgrep-search`, `run-tests`, …) are always
  available. Here they come from the HTTP server declared in `.mcp.json` — see
  [MCP Setup](#mcp-setup). If the server is down, the tools are absent; fall back to
  `Read`/`Edit`/`Grep`/`Bash` rather than treating its shell-command prohibition as
  binding. The underlying **EXPLORE → EXPERIMENT → PERSIST → VERIFY** loop always
  applies: prototype in the REPL, then persist to `src/`.
- **The test framework is rove**, matching the prompt's assumption, so `run-tests` and
  `(rove:run-test ...)` work as described.
- **FFI is expected.** LightGBM and XGBoost are C libraries; `common-lisp-expert.md`'s
  portable-CL guidance yields to whatever CFFI needs (raw pointers, foreign arrays,
  manual lifetime management). Wrap unsafe foreign calls, don't avoid them.

The style rules in `common-lisp-expert.md` (Google CL Style Guide, 2-space indent,
≤100 columns, lisp-case naming, `;;;`/`;;`/`;` comment hierarchy) do apply.

## Project Overview

`cl-gbdt` is a Common Lisp wrapper library for gradient boosting decision tree
implementations — **LightGBM** and **XGBoost** — exposing a single shared high-level API
over both backends.

**Status: functional.** Both backends (`cl-gbdt/lightgbm`, `cl-gbdt/xgboost`) implement
all 13 generic functions of the unified API -- `make-dataset`, `train`, `predict`, and
the rest -- against the real shared libraries, exercised by 563 functional assertions across
12 test files in `cl-gbdt/tests/functional` (layer 2) on top of 432 assertions across 18
test files that need no shared library at all (layer 1). `train` returns a `training-report`
as its secondary value, and takes `:record-history` (default `t`) to turn the per-iteration
recording that fills it off -- recording roughly doubles LightGBM's `train` time, and on
XGBoost it also makes a `:valid-sets` entry the library cannot evaluate fail `train` outright
(see `README.markdown`'s Training report section). `train` also takes `:early-stopping` to end
a run once a watched metric stops improving, and `predict`, `save-model` and
`model-to-string` accept `:num-iteration :best` to resolve against the iteration it
picked -- see `README.markdown`'s Training report section for both. `make-dataset` and
`predict` also accept a `csr-matrix` (built by `make-csr-matrix`) wherever they accept a
dense matrix, gated on the `:sparse-input` capability that both vendored backends answer
true; XGBoost's sparse `predict` serves `:normal` and `:raw` only, and an absent CSR entry
means `0.0` to LightGBM but *missing* to XGBoost -- see `README.markdown`'s Sparse input
section. `make-dataset` and `predict` also take `:missing`, the value in the caller's
own data that means missing, gated on the `:missing-value` capability that only XGBoost
provides -- LightGBM signals `capability-unavailable` for any non-`NIL` value, a `NaN`
included, since its C API has no missing-value key at all, and XGBoost compares the
sentinel against the data at single precision -- see `README.markdown`'s Missing values
section. `make-dataset` also takes `:categorical-features`, the 0-based columns that hold
categories rather than quantities, gated on the `:categorical-features` capability that
both backends provide; `predict` takes no such argument, the trained trees already
carrying the category sets they split on, and XGBoost's `tree_method` must be `hist` or
`approx` since `exact` refuses categorical splits, at `train` rather than `make-dataset`
-- see `README.markdown`'s Categorical features section. `predict` also returns the shape
the backend states for the result it just wrote as a second value -- a list of integers in
`array-dimensions` order, or `NIL` where the backend states none -- gated on the
`:prediction-shape` capability that both backends provide; XGBoost reads its own
`out_shape`/`out_dim` back from the library and states it verbatim, while LightGBM has no
such call and derives what it can, stating `NIL` for `:leaf-index` -- see
`README.markdown`'s Prediction shape section. `train` also takes `:objective`, a function
that turns the current raw scores into a gradient and a Hessian so a run boosts against the
caller's own loss, gated on the `:custom-objective` capability that both backends provide;
LightGBM flattens the array group-major and XGBoost row-major, the wrapper absorbing the
difference, and on LightGBM `:objective` overrides any `objective` in `:parameters`, forcing
it to `"none"`, since `LGBM_BoosterUpdateOneIterCustom` refuses to run while the booster
holds an objective function at all -- see `README.markdown`'s Custom objective section. Core
`cl-gbdt` still loads, and is still tested, without
either `liblightgbm.so` or `libxgboost.so` present: a shared library is opened only by
an explicit `open-backend` call, from whichever backend system you load on top of the
core. See `README.markdown`'s Usage section for a worked example, its system table, and
the design doc it points at.

The system is `:class :package-inferred-system`: one file, one package, and ASDF derives
each file's dependencies from its own `uiop:define-package` clauses rather than from a
hand-written `:components` tree. `cl-gbdt.asd` has real dependencies now (`cffi`,
`alexandria`, `rove`, `com.inuoe.jzon` among them) — do not add a `:components` clause
expecting to find one to edit; there isn't one.

### Package-inferred-system rules this project relies on

These are non-obvious enough, and costly enough to get wrong silently, that they are
worth stating explicitly rather than leaving them to be rediscovered:

- **A file's package name must equal its path from the repository root**,
  `/`-separated, lower case, extension dropped — `src/lightgbm/c-api.lisp` must declare
  `#:cl-gbdt/src/lightgbm/c-api`. Nothing enforces this at compile time by itself;
  `tools/ci/check-leaf-systems.lisp` is the check that does (see below).
- **Rove discovers a package-inferred-system's tests only through dependencies whose
  names have the system's own name as a literal string prefix**
  (`system-component-p` in `rove/core/suite/file.lisp`). This is why the functional
  test system is named `cl-gbdt/tests/functional` and not, say,
  `cl-gbdt/functional-tests` — the latter is not a prefix match and rove would report
  "0 tests completed" without erroring.
- **`(:import-from #:pkg)` naming zero symbols is the deliberate way to declare a
  dependency** for a file that only ever calls `pkg`'s functions package-qualified
  (`pkg:some-function`) and imports no bare symbols. Leaving the clause out instead
  loads correctly only when something else happens to have loaded `pkg` first, and
  breaks the moment load order shifts.
- **`src/*/c-api.lisp` are generated and must never be hand-edited.** They are produced
  by `tools/regen.lisp` from vendored C headers (see README's "Regenerating the
  bindings"). This is already enforced by `tests/bindings.lisp`'s
  `committed-bindings-match-their-committed-spec` test, which re-emits from the
  committed c2ffi spec and compares the result to the committed file byte-for-byte.

### Floating-point trap masking around every foreign call

**Every method in `src/lightgbm/protocol.lisp` and `src/xgboost/protocol.lisp` that
calls into its shared library wraps its whole body in `with-foreign-float-traps-masked`**
(`cl-gbdt/src/foreign`), not just the specific call this was first found through. The
same rule binds a **`defun` in either `native.lisp` or `protocol.lisp` of a backend once
that backend's public package exports it** (the second `uiop:define-package` form in the
sibling `all.lisp`, per `tools/ci/check-float-traps.lisp`'s `:export`-clause check): such
a `defun` is a library-reaching entry point with no `defmethod` left to inherit a mask
from, so it must wrap its own whole body the same way -- LightGBM's
`booster-eval`/`booster-eval-names` and XGBoost's `evaluate-one-iteration` were the first
functions this applied to, all three in `native.lisp`; XGBoost's `slice-model` is the
first in a `protocol.lisp`, where it lives because it builds a booster handle and so must
name the concrete class defined there.
SBCL enables the `:invalid`, `:divide-by-zero` and `:overflow` floating-point traps by
default on x86-64 and none of them on aarch64; LightGBM and XGBoost are C code written
and tested against the opposite (masked) convention, where an intermediate NaN or
infinity is data to keep working with, not a signal -- confirmed for XGBoost's
`multi:softprob` softmax normalization, which produced an uncaught
`floating-point-invalid-operation` mid-foreign-call on x86-64 only, before this macro
was added. Adding a raw foreign call inside an already-wrapped method or `defun` needs
nothing extra -- the existing body wrap already covers it -- but a **new** method, or a
**newly exported** `defun`, that reaches the shared library needs the same wrap around
its own whole body, or it silently reopens the gap on x86-64 while looking identical on
aarch64.

`tools/ci/check-leaf-systems.lisp` loads every leaf system alone, each in its own fresh
`ros run` subprocess, which is the only way to catch an undeclared dependency that
happens to be satisfied by load order in a shared image. Run it with:

```bash
ros run -- --non-interactive --load tools/ci/check-leaf-systems.lisp
```

It must report every leaf system as `PASS` (`N/N leaf systems load alone`).

## MCP Setup

`.mcp.json` points at a cl-mcp HTTP server on port 3001. Start it from a REPL that has
cl-mcp available and leave it running:

```lisp
(asdf:load-system :cl-mcp)  ; or (ql:quickload :cl-mcp)
(cl-mcp:start-http-server :port 3001)
;; => serving at http://127.0.0.1:3001/mcp; the REPL stays usable
```

The server must be up *before* Claude Code starts, otherwise the `cl-mcp` tools will not
be registered for the session. Source lives at `~/cl-mcp`
(`~/.roswell/local-projects/cl-ai-project/cl-mcp`).

`.mcp.json` is machine-local configuration — keep it out of version control
(see `.gitignore`).

## Requirements

**ASDF 3.3.7 or newer.** Roswell ships 3.3.1, whose package-inferred-system dependency
scanner does not know the `:local-nicknames` clause and dies with `:LOCAL-NICKNAMES fell
through ECASE expression` on any system that reaches `src/regen/emit.lisp`. Install a
current one once:

```bash
ros install asdf
```

Both CI workflows do this. The requirement arrived with the package-inferred-system
conversion -- before it, that clause was never parsed for dependency inference.

## Testing & Linting

Load and test (layer 1 — no shared library required):

```lisp
(ql:quickload :cl-gbdt/tests)
(asdf:test-system :cl-gbdt/tests)
```

Via the MCP `run-tests` tool, the system name is `cl-gbdt/tests`. Single test from the
REPL: `(rove:run-test 'cl-gbdt/tests/backend::some-test-name)` — substitute whichever
`cl-gbdt/tests/*` package the test actually lives in; `cl-gbdt/tests` itself is a
defsystem name only, no file defines a package by that name.

The functional suite (layer 2, calls the real shared libraries; needs
`./tools/fetch-libs.sh` first) is `cl-gbdt/tests/functional`, tested the same way.

`asdf:test-system` exits 0 even when tests fail, so it is not what CI runs. The
authoritative checks, runnable locally, are:

```bash
CL_GBDT_TEST_SYSTEM=cl-gbdt/tests ros run -- --non-interactive \
  --load tools/ci/run-tests.lisp                        # layer 1
CL_GBDT_TEST_SYSTEM=cl-gbdt/tests/functional ros run -- --non-interactive \
  --load tools/ci/run-tests.lisp                        # layer 2
ros run -- --non-interactive --load tools/ci/lint.lisp   # mallet + column-width check
ros run -- --non-interactive --load tools/ci/check-leaf-systems.lisp
```

`sbcl` is not on `PATH` in this environment; every command above goes through
`ros run -- --non-interactive ...`, not a bare `sbcl` invocation.

Before committing:

```lisp
(asdf:compile-system :cl-gbdt :force t)  ; surfaces warnings
```

```bash
ros run -- --non-interactive --load tools/ci/lint.lisp
```

`mallet` alone does not check line length; `tools/ci/lint.lisp` adds the ≤100-column
check on top of it. Running mallet by itself is not equivalent.

## Code Style

- Follow the Google Common Lisp Style Guide
- 2-space indent, ≤100 columns
- Blank line between top-level forms
- Lower-case lisp-case: `my-function`, `*special*`, `+constant+`, `something-p`
- Docstrings required for public functions/classes
- Each file starts with `(in-package ...)`

## Repository Structure

```
src/          Core implementation and the binding emitter (src/regen/), one package
              per file; src/*/c-api.lisp are generated -- never hand-edit them
tests/        Rove test suites, layer 1 (no shared library) plus tests/functional/,
              layer 2 (calls the real shared libraries)
tools/ci/     The scripts CI actually runs: run-tests.lisp, lint.lisp,
              check-leaf-systems.lisp
tools/        regen.lisp (regenerates src/*/c-api.lisp) and the shell scripts it and
              CI call
ffi-spec/     Vendored C headers and the c2ffi specs generated from them
prompts/      System prompts for AI agents (imported from cl-mcp)
```
