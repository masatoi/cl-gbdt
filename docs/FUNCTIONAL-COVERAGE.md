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
