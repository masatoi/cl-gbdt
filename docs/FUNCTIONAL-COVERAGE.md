# Functional Coverage

Every symbol `cl-gbdt`, `cl-gbdt/lightgbm` and `cl-gbdt/xgboost` export has a position here:
either the functional suite's own sources name it, or a row below says why they do not.
`tools/ci/check-functional-coverage.lisp` fails the build when a published symbol has neither, so
a symbol cannot reach the public surface without the project taking a position on how it is
proven.

**Covered is not written down.** The checker derives it: a symbol is covered when it appears in
some top-level form of a `tests/functional/*.lisp` file, outside that file's own package form.
Recording it here as well would give it a second home to go stale in -- the same rule
`ffi-spec/BINDING-COVERAGE.md` follows for `wrapped`.

Neither are counts or percentages. Coverage in this project is guaranteed by classification --
every symbol has a recorded position -- not by a number. The checker prints the live counts on
every run.

**What being covered does NOT mean.** That the suite's source names a symbol is not that the
symbol's contract is exercised. A condition named in a `handler-case` clause counts. So does a
function called once inside a fixture nothing asserts on. This file records where every published
symbol stands; it does not measure how well anything is tested, and a green run is not evidence
that it does.

The heading a row sits under is part of what the checker reads. It recognises exactly two
spellings: the literal `## Unproven`, and any heading beginning `## Exempt`. A row under anything
else -- a renamed heading, a new one, or one preceding the file's first `## ` heading -- fails the
build rather than landing silently in either bucket.

Each `## Exempt` heading carries its section's argument, so a row's Note is for what the heading
does not already say. A symbol whose argument fits none of the headings belongs under
`## Unproven`; adding a heading also means editing the checker's vocabulary, which is a deliberate
act rather than a matter of writing prose.

Rows name the symbol the way `docs/API-REFERENCE.md` heads its entries -- qualified by the
shortest public package that exports it -- so a reader can move between the two files.

## Unproven

Symbols that should have a functional test and do not. This is the work list. Each Note says what
a test would have to do, which is what an order of work gets argued from.

| Symbol | What a functional test would have to prove |
|---|---|
| `cl-gbdt:version-range-verified-evidence` | `version-range-tested-description` reads this slot into every VERIFIED-labelled entry it formats, but no assertion in `tests/version.lisp` checks the evidence text a formatted entry carries -- `version-range-tested-description-distinguishes-verified-from-inferred` only greps for the low/high numbers, not the parenthesised evidence string. A test proving this needs nothing from a shared library: it only has to call `version-range-tested-description` (or the accessor directly) on a fixture with a known evidence string and assert that string appears in the result. |
| `cl-gbdt:version-range-inferred-evidence` | Same gap, on the INFERRED-labelled entry: `version-range-tested-description` reads this slot too, and no assertion checks the inferred evidence text it produces. Also provable at layer 1 with no library -- the same kind of direct assertion as above, on the second entry `version-range-tested-description` returns. |
| `cl-gbdt:*lightgbm-version-range*` | Read by nothing but its own docstring and `src/version.lisp` itself -- `clgrep-search` across `src/` finds it named only in `version.lisp`'s own definition and, in prose, inside `lightgbm/classes.lisp`'s `initialize-backend` docstring, never in that method's code. Its own docstring states why: LightGBM's C API has no version entry point, `backend-version` is always NIL there, and `check-backend-version' is never called for this backend, so nothing -- no functional test, no layer 1 test, no runtime code path at all -- can compare a real library against this range. It is recorded for documentation only; there is no behavior left to prove. |
| `cl-gbdt:*xgboost-version-range*` | Read only by `xgboost/classes.lisp`'s `initialize-backend`, which passes it to `check-backend-version` right after reading the just-opened library's real version -- so proving this variable's recorded bounds match the vendored library means opening the real XGBoost backend and asserting `open-backend :xgboost` (or an equivalent functional fixture) signals no `untested-backend-version` warning. No test today, layer 1 or functional, references `*xgboost-version-range*` by name, even though every functional test that opens the XGBoost backend exercises it at runtime. |

## Exempt: proven without a shared library

The symbol's whole contract is computation the shared library takes no part in, and the layer 1
suite proves it. A functional test here would open a library, ignore it, and assert what
`tests/*.lisp` already asserts -- three minutes of CI for no additional evidence.

A row belongs here only when the layer 1 test exists. "Nothing calls the library" is half the
argument; the Note carries the other half, by naming the test that does the proving.

| Symbol | What proves it instead |
|---|---|
| `cl-gbdt:version-compare` | Orders two version strings and touches nothing else. `tests/version.lisp` covers each ordering, equal versions, differing component counts, and both malformed operands, in `version-compare-orders-by-component` and `version-compare-rejects-unparseable-input`. |
| `cl-gbdt:version-in-range-p` | Compares a version against two bounds via `version-compare' alone. `tests/version.lisp`'s `version-in-range-p-covers-inside-and-both-boundaries` proves both inclusive boundaries and a single-point range; `version-in-range-p-covers-below-and-above` proves both out-of-range directions; `version-in-range-p-cannot-confirm-unparseable-or-nil` proves an unparseable or NIL version or bound returns NIL rather than a false confirmation. |
| `cl-gbdt:check-backend-version` | Compares a version string against a range's *inferred* bounds and conditionally `warn`s; never touches a library itself, only ever called (at runtime) by XGBoost's `initialize-backend`. `tests/version.lisp`'s five `check-backend-version-*` tests, using a fixture range and `handler-bind`-collected warnings, prove: no warning inside the inferred range including both boundaries (`check-backend-version-does-not-warn-inside-the-inferred-range`); a warning below it, carrying the right backend, version and tested-description (`check-backend-version-warns-below-the-inferred-range`); a warning above it (`check-backend-version-warns-above-the-inferred-range`); a warning on an unparseable version (`check-backend-version-warns-on-unparseable-version`); and a warning on a NIL version (`check-backend-version-warns-on-nil-version`). |
| `cl-gbdt:make-version-range` | Constructs a `version-range` record from its six keyword slots. `tests/version.lisp`'s `%fixture-range` helper and, within `version-range-tested-description-distinguishes-verified-from-inferred`, a second range built with distinct VERIFIED bounds, both drive every other version test in the file -- a wrong slot order or a dropped keyword would break every one of them. |
| `cl-gbdt:version-range` | The record type `make-version-range` builds and every version-range test threads through `check-backend-version` and `version-range-tested-description`. Proven the same way as `make-version-range`: `tests/version.lisp`'s `%fixture-range` fixture and the second, differently-valued range built inside `version-range-tested-description-distinguishes-verified-from-inferred` both construct and consume an instance with exactly its six documented slots. |
| `cl-gbdt:version-range-verified-low` | Read into the VERIFIED-labelled entry `version-range-tested-description` formats. `version-range-tested-description-distinguishes-verified-from-inferred`'s first case uses a fixture with VERIFIED-LOW = VERIFIED-HIGH = `"3.3.0"` and asserts the formatted entry contains `"3.3.0 verified"`; its second case uses VERIFIED-LOW = `"4.0.0"` (distinct from VERIFIED-HIGH) and asserts the entry contains `"4.0.0-4.7.0 verified"`, which only holds if this accessor returns exactly `"4.0.0"`. |
| `cl-gbdt:version-range-verified-high` | Same mechanism as `version-range-verified-low`, proven by the same two cases in `version-range-tested-description-distinguishes-verified-from-inferred`: the first case's fixture sets VERIFIED-HIGH equal to VERIFIED-LOW and asserts the dash is omitted (`(ng (search "3.3.0-3.3.0" ...))`); the second sets VERIFIED-HIGH = `"4.7.0"`, distinct from VERIFIED-LOW, and asserts the entry contains `"4.0.0-4.7.0 verified"`, which only holds if this accessor returns exactly `"4.7.0"`. |
| `cl-gbdt:version-range-inferred-low` | Read into the INFERRED-labelled entry `version-range-tested-description` formats. `version-range-tested-description-distinguishes-verified-from-inferred`'s first case uses a fixture with INFERRED-LOW = `"1.7.0"`, INFERRED-HIGH = `"3.3.0"`, and asserts the second formatted entry contains `"1.7.0-3.3.0 inferred"`, which only holds if this accessor returns exactly `"1.7.0"`. |
| `cl-gbdt:version-range-inferred-high` | Same case as `version-range-inferred-low`: `version-range-tested-description-distinguishes-verified-from-inferred`'s assertion that the second formatted entry contains `"1.7.0-3.3.0 inferred"` only holds if this accessor returns exactly `"3.3.0"`, the fixture's INFERRED-HIGH. |
| `cl-gbdt:version-range-tested-description` | Formats a range's VERIFIED and INFERRED bounds as two evidence strings. `tests/version.lisp`'s `version-range-tested-description-distinguishes-verified-from-inferred` proves the no-dash case (VERIFIED-LOW = VERIFIED-HIGH) and the dashed case (VERIFIED-LOW /= VERIFIED-HIGH); `check-backend-version-warns-below-the-inferred-range` proves its return value is exactly what `untested-backend-version`'s `:tested` initarg receives when `check-backend-version` warns. |
| `cl-gbdt:normalize-parameters` | Turns a keyword plist into backend key/value string pairs and touches nothing else. `tests/parameters.lisp` covers name conversion to snake_case (`normalize-parameters-returns-name-value-pairs`), the odd-length-plist error (`normalize-parameters-rejects-odd-length-plist`), double-float, single-float and small-double-float printing without a type marker (`normalize-parameters-prints-double-float-without-a-type-marker`, `-single-float-`, `-small-double-float-`), ratio-to-decimal conversion (`normalize-parameters-prints-a-ratio-as-a-bare-decimal`, `-prints-a-ratio-as-a-decimal`), T/NIL as `"true"`/`"false"` (`normalize-parameters-prints-booleans-as-true-and-false`), and independence from the caller's `*print-base*` (`normalize-parameters-output-is-independent-of-the-caller-s-print-base`). |
