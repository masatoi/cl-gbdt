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

**Status: functional.** Both backends implement
all 13 generic functions of the unified API -- `make-dataset`, `train`, `predict`, and
the rest -- against the real shared libraries, exercised by 946 functional assertions across
15 test files in `cl-gbdt/tests/functional` (layer 2) on top of 555 assertions across 21
test files that need no shared library at all (layer 1).

**Each backend is two systems.** `cl-gbdt/<backend>` is that backend's **Layer 1 alone**:
`src/<backend>/native.lisp` (the `%`-functions, over raw pointers) over the generated
`c-api.lisp`, plus `src/<backend>/classes.lisp` (the backend's CLOS types,
`register-backend`, and the `initialize-backend`/`shutdown-backend` pair that opens and
closes the shared library), plus `src/<backend>/api.lisp` (the finished operations: each
takes a backend or a handle, does the whole job, and hands back a handle or a result),
published by `src/<backend>/all.lisp`. It does
**not** carry the 13 unified-API methods and does not define the `cl-gbdt` package.
`cl-gbdt/<backend>/unified` adds `src/<backend>/protocol.lisp` -- those 13 methods -- and core
`cl-gbdt` with it, aggregated by `src/<backend>/unified.lisp`. Anything calling `cl-gbdt:train`
loads `/unified`; loading only Layer 1 and calling a portable generic signals
`backend-methods-not-loaded`, which names the system to load.
`tools/ci/check-layer-separation.lisp` fails the build if a Layer 1 file's dependency closure
ever reaches `cl-gbdt/src/protocol`, the training files, or the bare `cl-gbdt`.
**A Layer 1 caller trains, predicts, persists and reports**, both backends: `create-dataset`,
`create-booster`, `update-one-iteration`, `predict`, `free-dataset`, `free-booster`,
`save-model`, `load-model`, `model-to-string`, `feature-importance`, `evaluation`,
`dataset-num-rows` and `dataset-num-features` -- thirteen operations -- are published from
`api.lisp` and proven with no unified API in the image by
`tests/functional/{lightgbm,xgboost}-standalone.lisp`, each of which names its backend's
public package and no other system **of this project** -- `rove` aside, they declare
nothing. Twelve of the thirteen methods delegate their whole procedure to these; `train`
calls `create-booster` for its whole construction only -- its loop still calls
`native.lisp`'s functions directly rather than `api.lisp`'s `update-one-iteration`, which
would re-check every handle on every iteration -- and is the one that also writes the best
iteration back afterward, through the internal
`%set-booster-best-iteration` in `src/handle.lisp`, once its loop ends. The delegation left
the unified API's own behaviour alone but for one ordering, recorded in both backends'
`predict` docstrings:
`predict` now refuses a bad `:kind` *below* its `:missing` gate and its `:best` resolution
rather than above them, so a call wrong in two ways at once gets a typed `cl-gbdt` condition
where an untyped `sb-kernel:case-failure` used to escape. What a Layer 1 caller still cannot
do is the training report, early stopping and the `:objective`/`:evaluation` callbacks, which
are `train`'s own concepts and stay Layer 2.

`train` returns a `training-report`
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
difference, and on LightGBM `:objective` overrides any `objective` in `:parameters` -- all
five spellings that library honours, its `objective_type`, `app`, `application` and `loss`
aliases included -- forcing it to `"none"`, since `LGBM_BoosterUpdateOneIterCustom` refuses to
run while the booster holds an objective function at all; a non-`NIL` `:objective` that is not
a `function` signals `unsupported-argument` on both backends before any foreign call. What the
objective RETURNS is checked for shape only -- `double-float`, `single-float` and a general
array of reals all train the same model, and a non-real element signals
`unsupported-element-type` where the buffer is written -- and because the objective is the
only caller code inside `train`'s loop, `train` re-runs its own dataset and backend checks
the moment it returns, so an objective that frees the training set gets
`released-handle-error` instead of a memory fault -- see
`README.markdown`'s Custom objective section. `train` also takes `:evaluation`, a function
called once per dataset per iteration, after that iteration's update, with that dataset's
`predict :kind :normal` scores and the dataset's index -- 0 the training set, N+1 the Nth
`:valid-sets` entry, the numbering `:early-stopping`'s `:dataset` key already uses -- and that
returns a metric name and a real or `NIL` value -- a real one recorded as a `double-float`,
coerced where the entry is built so every series holds what `training-series-values` documents,
and one too large for a `double-float` recorded as the signed infinity by the same
`handler-case` wrap `src/config/missing-value.lisp`'s `%rational-json` uses, so the stored
value does not depend on whether the platform traps `:overflow`
-- gated on the `:custom-evaluation` capability
that both backends provide, LightGBM out of a probe and XGBoost out of a declaration since its
one required C function needs no optional-symbol check; the values become their own report
series, appended after the library's own for the same iteration so `evaluation`'s own pairs
stay a prefix of `training-report-series`, and are watchable by `:early-stopping` under the
name `:evaluation` returned with nothing else to arrange. Refused, on both backends and before
any foreign call, for `:record-history nil` and a non-`function` value -- a symbol included,
since `funcall` would resolve it afresh each iteration against whatever global definition was
then in force -- and refused mid-run for a name that is not a string, a value that is neither
a real nor `NIL`, a name colliding with a library metric at the same dataset index (caught at
the end of the first iteration), and a name that CHANGES at a given index between iterations:
one name per dataset index is required for the whole run, since a series is keyed by the
(index, name) pair and a varying name would make series that are shorter than the run, or --
when it varies into a library metric's name, which the first-iteration collision check cannot
reach -- longer than it. A name REWRITTEN IN PLACE is refused on the same terms and could not
have been caught by the pin alone: the name is `copy-seq`d into the entry and both call sites
take it back out of that entry for the pin and the collision check, so the caller's own string
object reaches neither them nor the history -- see `README.markdown`'s Custom evaluation
section. Core
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

### The handle layer, and `with-pointer-ownership`

`src/handle.lisp` wraps every foreign dataset and booster pointer in a CLOS object
(`dataset`, `booster`), so free-once, use-after-free detection and the unfreed-handle
finalizer are written once rather than twice per backend. `make-handle` is what takes
ownership of a raw pointer; `release-handle` frees it exactly once and cancels the
finalizer.

**`with-pointer-ownership` is a public macro covering the window before `make-handle`
runs.** A creation call such as `LGBM_DatasetCreateFromMat` returns a live foreign
resource, but the code that follows it typically has more to do -- attaching a label, a
weight, feature names -- before a Lisp object exists to own the pointer. In that window
nothing in Lisp references the resource, so nothing will ever free it and no finalizer
will ever fire for it: a body that leaves by signalling, by `throw`, by `return-from`, or
simply by returning without taking ownership orphans it outright. The macro closes exactly
that gap, calling the free function it was given unless the body handed the pointer to a
handle:

```lisp
(with-pointer-ownership (raw #'%free-dataset-unchecked take-ownership)
  (%set-info-field raw "label" label)
  (take-ownership 'lightgbm-dataset backend :dataset))
```

It is an implementor's tool, not part of the everyday API -- a caller who only trains and
predicts never reaches for it. It is on `cl-gbdt`'s public surface all the same, via
`src/all.lisp`'s re-export-every-top-level-file rule, and policy section 14 makes anything
on that surface a compatibility obligation. A new backend, or any new code that builds a
handle from a fresh foreign pointer, should use it rather than an ad-hoc
`unwind-protect`. Any error the free function itself signals is discarded, so a failing
cleanup cannot replace the condition that caused the unwind (policy section 10).

### Floating-point trap masking around every foreign call

**Every method in `src/lightgbm/protocol.lisp`, `src/xgboost/protocol.lisp`,
`src/lightgbm/classes.lisp` and `src/xgboost/classes.lisp` that
calls into its shared library wraps its whole body in `with-foreign-float-traps-masked`**
(`cl-gbdt/src/foreign`), not just the specific call this was first found through -- the two
`classes.lisp` files hold `initialize-backend` and `shutdown-backend`, which open and close
the library and are as much foreign-reaching as any protocol method. The
same rule binds a **`defun` in `native.lisp`, `classes.lisp`, `api.lisp` or `protocol.lisp`
of a backend
once that backend's public package exports it** (the second `uiop:define-package` form in the
sibling `all.lisp`, per `tools/ci/check-float-traps.lisp`'s `:export`-clause check): such
a `defun` is a library-reaching entry point with no `defmethod` left to inherit a mask
from, so it must wrap its own whole body the same way -- LightGBM's
`booster-eval`/`booster-eval-names` and XGBoost's `evaluate-one-iteration` were the first
functions this applied to, all three in `native.lisp`; every exported `defun` in either
backend's `api.lisp` is one too -- thirteen operations on LightGBM, fourteen on XGBoost,
the extra one being `slice-model`, which lives there rather than in `native.lisp` because
it builds a booster handle and so must name the concrete class `classes.lisp` defines. All
four file names are globbed by that check's `+BACKEND-FILE-PATTERNS+`; a backend file
under some other name would be scanned by nothing.
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
ros run -- --non-interactive --load tools/ci/check-layer-separation.lisp
ros run -- --non-interactive --load tools/ci/check-float-traps.lisp
ros run -- --non-interactive --load tools/ci/check-abi-blacklist.lisp
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
