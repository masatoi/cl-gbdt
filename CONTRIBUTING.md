# Contributing

How to run cl-gbdt's test suites, what continuous integration checks, and how to regenerate the generated bindings and API reference.

## Running the tests

**First, once: ASDF 3.3.7 or newer is required.** Roswell ships 3.3.1, whose
`package-inferred-system` dependency scanner does not know the `:local-nicknames`
clause; loading any system that reaches `src/regen/emit.lisp` dies with
`:LOCAL-NICKNAMES fell through ECASE expression` before a single test runs.

```bash
ros install asdf
```

Then:

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

Three workflow files below, though only two carry a badge above — a GitHub Actions badge
covers a whole workflow file, not a job, which is why the two badges report independently:

- `.github/workflows/test.yml` runs both suites on Linux x86_64, Linux aarch64 and macOS
  aarch64. The matrix is the point: the bindings are generated on one machine and
  committed, so passing on that machine proves little. macOS is also the only place the
  `.dylib` discovery path is exercised at all.
- The same workflow's `version-matrix` job (task 4) reruns layer 2 -- only layer 2, since
  layer 1 needs no library and gains nothing from repetition -- against the endpoints of the
  recorded compatible-version range: LightGBM 4.0.0 and XGBoost 2.0.0, each against the other
  backend's pinned version. **It does not run on pull requests** -- each leg costs six to
  nine minutes, and adding two of them was enough, on a busy runner pool, to leave every job
  queued and get sibling jobs cancelled; what it checks changes when upstream releases, not
  when we commit, so it runs on push to master and weekly instead. Expect it after a merge,
  not on your PR. **XGBoost 1.7.0 is not one of the legs**, though it is the
  version-matrix table's failing row (see [Backend
  differences](docs/user-guide/backend-differences.md#where-the-two-backends-genuinely-differ)).
  It was added as a deliberately-red `continue-on-error` leg and then removed, because it
  never reached the test: the 1.7.0 wheel does not install on the runner, so the leg failed in
  `Fetch the backend shared libraries` and demonstrated nothing, at roughly six minutes and
  enough contention to get sibling jobs cancelled. The evidence for narrowing the range is the
  local run recorded in `src/version.lisp`'s docstring, not a red CI job — see that job's own
  header comment in `test.yml`, which is the record of the decision. The
  pinned versions (4.7.0/3.3.0) are already covered by the job above, so this job does
  not repeat them. **One platform only, Linux x86_64** -- the three-platform matrix above
  exists to catch platform-specific bugs (byte order, calling convention, `.dylib` vs `.so`
  discovery) in bindings generated once and committed; a library *version* difference is a
  property of the upstream C source, identical across every platform cl-gbdt runs on, so
  crossing this axis with all three platforms would have tripled the job's cost for no new
  information. Crossing the two axes fully was also rejected for the same reason -- a
  LightGBM version and an XGBoost version load two independent shared libraries with no
  interaction between them, so each axis's own endpoints are varied one at a time against the
  other's pinned version, not against each other's endpoints too.
- `.github/workflows/lint.yml` runs the static checks on one target, since nothing they
  look at varies by machine.
- `.github/workflows/upstream.yml` asks PyPI once a week — and on every push to master —
  whether LightGBM or XGBoost has released, and answers in the same run whether cl-gbdt
  could take the new version. **It does not run on pull requests**, and it has no README
  badge: a red badge named `upstream` reads, to a visitor, as a broken library, where what
  it actually means is that the pin is one version behind. Its four jobs each mean one
  thing, so which job is red says which of four things happened — see
  [When the upstream workflow is red](#when-the-upstream-workflow-is-red) below.

The logic lives in scripts rather than in the YAML, so the same checks run locally. **This is
the whole set — every script under `tools/ci/`, in the order CI runs them.** Run the block
before opening a pull request, and add a line here whenever `tools/ci/` gains a script:

```bash
CL_GBDT_TEST_SYSTEM=cl-gbdt/tests ros run -- --non-interactive \
  --load tools/ci/run-tests.lisp          # layer 1
CL_GBDT_TEST_SYSTEM=cl-gbdt/tests/functional ros run -- --non-interactive \
  --load tools/ci/run-tests.lisp          # layer 2, needs ./tools/fetch-libs.sh first
ros run -- --non-interactive --load tools/ci/lint.lisp
ros run -- --non-interactive --load tools/ci/check-leaf-systems.lisp
ros run -- --non-interactive --load tools/ci/check-layer-separation.lisp
ros run -- --non-interactive --load tools/ci/check-layer-1-guards.lisp
ros run -- --non-interactive --load tools/ci/check-float-traps.lisp
ros run -- --non-interactive --load tools/ci/check-abi-blacklist.lisp
ros run -- --non-interactive --load tools/ci/check-binding-coverage.lisp
ros run -- --non-interactive --load tools/ci/check-api-reference.lisp
ros run -- --non-interactive --load tools/ci/check-functional-coverage.lisp
ros run -- --non-interactive --load tools/ci/check-doc-links.lisp
ros run -- --non-interactive --load tools/ci/check-support-matrix.lisp
```

Several of those are source scans rather than loads. `check-layer-separation.lisp` proves no
Layer 1 system reaches the unified API (see [Systems](README.markdown#systems));
`check-float-traps.lisp` proves
every backend `defmethod` and every publicly exported backend `defun` wraps its body in
`with-foreign-float-traps-masked`; and `check-abi-blacklist.lisp` proves no backend imports a
C entry point `ffi-spec/ABI-BLACKLIST.md` rules out, that every import a backend does make is
declared in its `*required-symbols*` or `*optional-symbols*`, and that every capability
either list declares is registered in `*known-capabilities*`.

**The last two are this branch's own additions, and they guard documentation rather than
code.** `check-doc-links.lisp` resolves every relative Markdown link in the tracked docs --
the file part against the linking file's own directory, and the `#fragment`, where there is
one, against the target's headings under GitHub's slug rule -- because the README split
turned a handful of cross-references into dozens and nothing in CI had ever read a Markdown
file before. `check-support-matrix.lisp` proves the CI-verified column of the README's
supported-environments table names exactly the versions CI actually runs -- the union of
`ffi-spec/VERSIONS`'s pin and `.github/workflows/test.yml`'s `version-matrix` job, checked in
both directions, so a version CI tests that the README omits fails as loudly as one the README
claims and CI does not. The "Also measured" column is deliberately outside its reach, being
hand measurement rather than anything a machine can re-derive. Both scripts fail the build
rather than warning, on the same terms as every check above them.

Two things the test scripts do that the plain commands above do not, and that CI needs:

- **They exit non-zero when a test fails.** `asdf:test-system` exits 0 regardless, so a job
  invoking it directly would be permanently green. `rove:run` returns false on failure and
  the script turns that into a status.
- **They check which foreign libraries were opened.** Layer 1 must open none; layer 2 must
  open both. rove counts a skip as pending rather than as a failure, so a functional run in
  which every library-dependent test skipped for want of `vendor/` still exits green and still
  reports each test file as completed — the summary reads the same as one where both libraries
  really were called. The difference has to be asserted rather than inferred.

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

### When the upstream workflow is red

`.github/workflows/upstream.yml` runs four jobs, and the combination says which of four
things happened:

| `pin-is-current` | `drift` | `try` | What it means |
|---|---|---|---|
| pass | pass | skipped | The pin is current. Nothing to do. |
| **fail** | pass | pass | A new release exists and **the suite passes as-is**. |
| **fail** | pass | **fail** | A new release exists and **the suite does not pass as-is**. |
| **fail** | **fail** | skipped | A new release **changed a declaration cl-gbdt imports**. |
| \- | \- | \- | `discover` failed: **the detector could not look**. Never read this as "nothing changed". |

`discover` asks PyPI what the latest stable releases are and compares them with
`ffi-spec/VERSIONS`. `pin-is-current` fails when they differ. `drift` runs
`tools/check-upstream.lisp` against the *latest* tag rather than the pinned one. `try`
fetches the latest wheel and runs layer 2 against it, on the backends that are actually
behind — and only when `drift` was clean, since a changed declaration has already answered
the question.

**`pin-is-current` stays red until `ffi-spec/VERSIONS` is bumped.** There is no
acknowledgement file and no way to mark a release as skipped; that was chosen deliberately
over an annotation-only job, on the grounds that a weekly job summary nobody opens is the
same as no detector at all.

What to do when it is red:

1. **Read `drift` and `try` first.** They say whether the bump is safe, and a red
   `pin-is-current` on its own says nothing about that.
2. **If both are green**, perform the version bump: it is its own piece of work, and
   [Regenerating the bindings](#regenerating-the-bindings) below is the procedure. Bumping
   `ffi-spec/VERSIONS` alone is not enough — the vendored headers, the generated
   `src/*/c-api.lisp`, `ffi-spec/BINDING-COVERAGE.md`, `ffi-spec/ABI-BLACKLIST.md`,
   `src/version.lisp`'s recorded range and the README's supported-environments table all
   move with it, and `tools/ci/check-support-matrix.lisp` will fail until the last of them
   does.
3. **If `drift` is red**, upstream changed the declaration of a function cl-gbdt imports --
   or `tools/check-upstream.lisp` could not look at all. Its own header lists a fetch
   failure, a tag that does not exist upstream among them, as another cause of the same
   FAIL; this is reachable whenever PyPI reports a version whose `v`-prefixed GitHub tag is
   absent or spelled differently. **Read the job's output** before concluding a declaration
   changed. `ffi-spec/ABI-BLACKLIST.md` names this tool as its own maintenance path: a
   function reported ABSENT moves from that file's "still present" table to "moot", and one
   reported CHANGED is added to "still present".
4. **If `try` is red**, read *which* tests failed before concluding anything.
   On XGBoost, `xgboost-api-open-backend-against-vendored-library-warns-nothing` fails on
   every new release by construction: it asserts `open-backend` emits no
   `untested-backend-version` warning, and `src/version.lisp`'s `*xgboost-version-range*`
   names the pinned version as its `inferred-high`. **That test failing alone means the
   recorded range has not been widened yet**, which the bump does. LightGBM's C API
   exposes no version, so its legs never reach this path.
   Any *other* failing test is a real behavioural difference, and it belongs in
   `src/version.lisp`'s recorded range and in
   [Backend differences](docs/user-guide/backend-differences.md), the way XGBoost 1.7.0's
   `rank:pairwise` failure already is.
5. **If `discover` is red**, the detector could not look — PyPI was unreachable, or reported
   a version in a shape the script refuses to guess at. Nothing about upstream has been
   established either way.

To run the detection half by hand, without waiting for a schedule:

```bash
./tools/latest-upstream.sh
```

Once the workflow is on `master`, a whole run can also be triggered on demand with
`gh workflow run upstream.yml`.

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

**`tools/regen.lisp` always rewrites both backends' `.spec` files, even when only one
header changed.** c2ffi's own output is not byte-stable across separate runs -- its
internal `ns`/`id` numbers on anonymous structs and macro-derived `const` entries can
shift, and entries can reorder, with no effect on the emitted `src/*/c-api.lisp` when
nothing in that backend's header actually changed. A regeneration aimed at one backend
should therefore `git checkout <base-commit> -- <the other backend's .spec path>` before
committing, so the diff reflects only the backend that changed, then re-run
`cl-gbdt/tests`'s `committed-bindings-match-their-committed-spec` to confirm the reverted
spec still reproduces the committed bindings for both backends. Separately,
`tools/fetch-headers.sh` `rm -rf`s each backend's whole `include/` directory before
repopulating just the headers, so both committed `.spec` files transiently show as
deleted in `git status` between the fetch and the following `regen.lisp` run -- expected,
and resolved once regeneration recreates them.

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

Every one of those 177 emitted functions is classified in
`ffi-spec/BINDING-COVERAGE.md` as `wrapped`, `planned`, or `excluded`;
`tools/ci/check-binding-coverage.lisp` fails the build on one that is none of the
three.

## Regenerating the API reference

`docs/API-REFERENCE.md` is generated, checked in, and covers every symbol `cl-gbdt`,
`cl-gbdt/lightgbm` and `cl-gbdt/xgboost` export, one section per symbol built from its
own docstring (and, for a class or condition, its slots' docstrings too). **You do not
need to regenerate it to use, build, or test this library** -- the same rule
`src/*/c-api.lisp` carries applies here: it is a developer-only step, for when a
docstring changes.

Unlike the bindings above, regeneration needs no Docker and no network access -- it
introspects a loaded image, not a C header. Run it from the repository root:

```bash
ros run -- --non-interactive --load tools/gen-api-reference.lisp
```

`tools/gen-api-reference.lisp` loads both backends' `/unified` systems (so all three
public packages are present to introspect) and the development-only `cl-gbdt/docgen`
system (`src/docgen/`), then writes `docs/API-REFERENCE.md` from what it finds. Neither
shared library is opened -- that happens only on an explicit `open-backend` call, and
nothing here makes one.

`tools/ci/check-api-reference.lisp` regenerates into a temporary file and fails the
build the same way `tests/bindings.lisp` does for the C bindings: byte-for-byte against
the committed copy, naming the first differing line and byte offset on a mismatch. It
also checks two floors a byte-for-byte comparison cannot: that every published symbol
carries a docstring (or points at a documented type, for a reader with none of its
own) and every class/condition slot a published reader exposes carries its own
`:documentation`, and that each of the three packages still publishes at least its
recorded symbol count -- catching an export dropped and honestly regenerated, which
byte-for-byte agreement alone would miss.
