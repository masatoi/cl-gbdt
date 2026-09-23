# cl-spec dogfooding log

## What this is

`specs/` holds an optional bundle of executable specifications written with
[cl-spec](https://github.com/masatoi/cl-spec), pinned at revision `08d3ada`
(`08d3adaf912b614537ca32cf5a9451fe0b807337`, fetched by `./tools/fetch-cl-spec.sh`). Two
systems carry it: `cl-gbdt/specs` registers six Function Specs and nine Properties with
cl-spec core alone, and `cl-gbdt/tests/specs` runs every one of them at seed 42 as a Rove
suite. Neither is a dependency of any system a user loads; core `cl-gbdt` still loads with no
cl-spec in the image. From the shell, the gate is

```bash
./tools/fetch-cl-spec.sh
CL_GBDT_TEST_SYSTEM=cl-gbdt/tests/specs ros run -- --non-interactive \
  --load tools/ci/run-tests.lisp
```

and from a cl-mcp session, `load-system cl-gbdt/specs/check-it` and then `spec-check
function=<target>` (a contract) or `spec-check symbol=<target>` (the Properties about it).
[CONTRIBUTING.md](../CONTRIBUTING.md#running-the-executable-specifications) has both routes.

This file is the dogfooding feedback log the introduction plan asked for (its section 17):
what could be specified, what could not, what was done instead, and the evidence for each.
**Every number below was re-measured on 2026-09-23** against the tree at `5f5535d`, in a fresh
cl-mcp worker, after `load-system cl-gbdt clear_fasls=true` and `load-system
cl-gbdt/specs/check-it`, with the local cl-spec checkout at exactly the pinned revision. Where
a number from planning did not reproduce, the entry says so and gives the re-measured one.
Two later changes moved the declarations, and every digest, case count and mutation result
below was re-measured the same way after each: the `*print-case*` fix (see [Findings about
cl-gbdt](#findings-about-cl-gbdt)), which widened `parameter-value` to hold symbols, and the
declared-domain fixes from PR review (G8).

## What was specified

Six Function Specs, each run with `spec-check function=... trials=200 seed="42"`, and nine
Properties, each run with `spec-check property=... profile=normal seed="42"`. Every row
passed; every contract rejected 0 generated inputs -- three have a `:pre`, which their
generators satisfy by construction (G5, G8) -- and every `:cases` contract
called every case. `rejected` is not measured for a Property -- each call reported
`rejection-counts-unmeasured` and `input-coverage-unmeasured` among its `verification_gaps`,
and each contract call reported `input-coverage-unmeasured`.

| Definition | Kind | Clauses used | Seed 42 result | Digest (`fnv1a64-v1:`) |
|---|---|---|---|---|
| `contrib-shape` | Function Spec | `:cases`, `:post`, `:args-generator` | 200/200, 0 rejected; `:derivable` 99, `:underivable` 101 | `4de05ef75c7c211a` |
| `normalize-parameters` | Function Spec | `:cases`, `:signals`, `:pre`, `:post`, `:args-generator` | 200/200, 0 rejected; `:even-length` 99, `:odd-length` 101 | `ee2e63a910959e94` |
| `objective-single-float` | Function Spec | `:cases`, `:signals`, `:post`, built-in generation | 200/200, 0 rejected; `:real` 111, `:not-real` 89 | `875553b8e348d02a` |
| `make-training-series` | Function Spec | `object-of` return, `:pre`, `:post`, `:args-generator` | 200/200, 0 rejected | `9c4f12ea038c9dbd` |
| `make-training-report` | Function Spec | `object-of` return, `:pre`, `:post`, `:args-generator` | 200/200, 0 rejected | `d6ff12b3a7478da8` |
| `training-report-from-history` | Function Spec | `object-of` return, `:post`, built-in generation | 200/200, 0 rejected | `e213621eb87d33db` |
| `normalize-parameters-keeps-order-and-renames-keys` | Property (invariant) | built-in generation | 200/200 | `59ccbf9dafea18e3` |
| `normalize-parameters-values-denote-themselves` | Property (round-trip) | built-in generation | 200/200 | `d803ac1508cea629` |
| `normalize-parameters-ignores-the-caller-s-printer` | Property (invariant) | built-in generation | 200/200 | `ec690a0043d8d5e1` |
| `objective-parameters-ends-with-the-one-canonical-objective` | Property (invariant) | built-in generation | 200/200 | `1bb00063d8475032` |
| `objective-parameters-keeps-every-other-entry-in-order` | Property (invariant) | built-in generation | 200/200 | `8f1ff2d0025de731` |
| `objective-parameters-is-idempotent` | Property (idempotence) | built-in generation | 200/200 | `8dd9a5ef5f4852e4` |
| `history-yields-one-series-per-pair-in-first-appearance-order` | Property (invariant) | built-in generation | 200/200 | `a12e3de0c035750d` |
| `history-series-values-are-the-pair-s-values-in-order` | Property (invariant) | built-in generation | 200/200 | `edf236a43a0795cc` |
| `history-series-name-is-the-dataset-s-name` | Property (invariant) | built-in generation | 200/200 | `63d4aba0cd71dd15` |

"Built-in generation" means the arguments are drawn from their declared specs; several of
those specs have custom-generator children from `specs/values.lisp` (see G1). A digest covers
the declaration and its registered dependencies, not the function body under test.

The contracts' `result-data` at seed 42 also records a `:capabilities` plist. The four
contracts with an `:args-generator` report `:shrinking :none`; `objective-single-float` and
`training-report-from-history`, whose arguments come from specs, report `:shrinking
:available` (see G2).

## Why cl-spec rather than Rove, and what stayed Rove

The rule used, from the plan's section 15: one concrete value, or a past regression, stays a
Rove example; an invariant over a whole input set becomes a Property; a function's pre/post
conditions and signals become a Function Spec; and a regression that broke a general rule
stays in Rove *and* gets that rule written here.

- **`tests/parameters.lisp`** pins `0.05d0` → `"0.05"`, `0.05` (single) → `"0.05"`,
  `1.0d-7` with an `e` marker, `1/3` as a decimal, `T`/`NIL` as `"true"`/`"false"`, an odd
  plist refused, and `*print-base*` 16 not leaking. The three `normalize-parameters`
  Properties state the rules those examples were chosen to illustrate -- names and order;
  every value reads back as itself with no Lisp-only syntax; independence from
  `*print-base*`, `*print-radix*`, `*read-default-float-format*` and `*print-case*` -- over
  generated plists mixing integers, strings, booleans, symbols, single-floats, doubles and
  ratios. The contract adds the
  even/odd split with the `data-error` refusal.
- **`tests/prediction-shape.lisp`** pins a multiclass split, a single-class model, an inexact
  division, the zero-class exact division, and degenerate counts. The `contrib-shape`
  contract states the docstring's condition -- the shape exists exactly when the counts admit
  one, and then accounts for every element -- over generated counts.
- **`tests/objective.lisp`** pins every real type `objective-single-float` coerces and the
  non-reals it refuses, and lists LightGBM's objective aliases and eight near misses by hand.
  The contract states the coercion within one single-float rounding for any real; the three
  `objective-parameters` Properties state "exactly one objective survives", "every other
  entry survives in order, near misses included" and idempotence, over generated mixtures of
  aliases (keyword and string spellings, mixed case) and near misses.
- **`tests/training-history.lisp`** pins one series per pair, first-appearance order, values
  in iteration order, a NIL kept in its slot, and the empty cases. The three `history-*`
  Properties state the same four promises over generated histories that need not look like a
  real backend's -- a pair missing from an iteration, or repeated within one -- which is the
  wider domain the implementation's hash-table fold actually guarantees.
- **`tests/training-report.lisp`** pins that each reader returns its initarg, an unnamed
  series, NIL values, the early-stopping slots and `print-object`. The two constructor
  contracts state "reports back the same contents it was built from" (by `equal`/`equalp`, not
  `eq`: the API promises contents, not that the object keeps the caller's very string or
  vector) for generated arguments, with the result validated by `object-of`.

**No Rove test was removed or edited.** Each pins either a measured value (`0.05d0`,
`1.0d-7`, the alias list measured against LightGBM 4.7.0) or a regression that once
happened, and a generated Property that happens to cover the same input is not a guarantee
that it will keep drawing it. `check-objective-result`, the error-message wording tests, and
`print-object` output have no Property: each is one concrete fact.

## CLOS observations

**`object-of` was sufficient as a return contract.** `training-series-object` and
`training-report-object` describe an instance by its readers, with no MOP and no
construction, and three contracts use them as `:returns`. A failure names the path through
the readers. In Task 4, `(cl-spec:explain-data 'training-report-object ...)` on a report whose
one series held `#(1)` reported the error path `(training-report-series 0
training-series-values 0)`, the failing conjunct `(:vector-of (:satisfies double-or-nil-p))`
and the actual value `1` -- nested three objects deep, and precise enough to act on.

**`object-of` was insufficient as an input without a named generator.** It generates
nothing by itself, so `training-series-object` carries `(:generator
training-series-generator)`, which builds a series through `make-training-series`, so that
`(list-of training-series-object)` -- `make-training-report`'s declared `:series` domain --
is generable at all (that contract draws its calls from a whole-call generator instead, per
G5). That was a one-line workaround here because both classes have a public constructor
whose arguments are plain values. It would not be for `dataset` or `booster`,
which own foreign resources (deliberately out of scope, below).

## Gaps met

Each entry uses the section 17 fields. **Blocked** means it forced a workaround in this
bundle; **would be nice** means the bundle got by without one. **Recurred** means it came up
in more than one independent contract -- the plan's rule is to propose a cl-spec change only
for a gap that recurs, so an improvement line is given for those alone, kept to one line.

### G1 -- floats, ratios and keywords validate but do not generate

- **Status:** blocked. **Recurred:** yes -- `normalize-parameters` (values), the
  `objective-single-float` contract (`element-value`), the history specs
  (`history-entry`'s value, `best-score`) and the report constructors (`best-score`).
- **Target:** every definition whose domain holds a `double-float` or a ratio.
- **Wanted:** `(range double-float -1d3 1d3)`, `double-float`, `ratio` as generable specs.
- **Result (re-measured with `cl-spec:sample`):** `(range double-float 0 1)` is refused with
  `invalid-spec-form` ("RANGE is numeric; its base type must be INTEGER or REAL");
  `double-float`, `ratio` and `keyword` signal `generator-unavailable` ("no generator is
  registered for the type ..."); `real` and `(range real -1 1)` generate, but five draws at
  seed 42 were five `single-float`s.
- **Fallback:** custom generators in `specs/values.lisp` -- `finite-double-generator` (a
  signed integer mantissa scaled by 10^-8..10^8) and `non-integer-ratio-generator` -- each
  wrapped in a named spec (`finite-double`, `non-integer-ratio`); keywords are enumerated
  with `member`.
- **Pain:** moderate to write, and a lasting cost: **a custom value generator does not
  shrink**. Nor, per cl-spec's own README, does a built-in real: in the `objective-parameters`
  mutation (section 7) the counterexample shrank from six pairs to one and its key from
  `:applications` to `:apps`, but its value, the `(range real ...)` single-float `-271.05573`,
  came through exactly as drawn. Before the `*print-case*` fix widened `parameter-value`, the
  same run kept a `finite-double` (`-8.15221d8`) as drawn instead -- so neither the custom nor
  the built-in route to a non-integer number gives a counterexample whose numbers shrink.
- **Also:** `finite-double` excludes the ±infinity a real `best-score` can hold (the custom
  evaluation path records an overflowing value as a signed infinity), so the `best-score`
  domain in both constructors and `training-report-from-history` is narrower than the
  function's; the return spec says `(nullable real)`, which admits infinity. No infinity
  generator was written.
- **Proposed improvement:** built-in, shrinking generators for `double-float`, `ratio`,
  `keyword` and `(range double-float ...)`.

### G2 -- a whole-call generator turns shrinking off

- **Status:** blocked (accepted, not worked around). **Recurred:** yes -- all four
  `:args-generator` contracts (`contrib-shape`, `normalize-parameters`,
  `make-training-series`, `make-training-report`) report `:capabilities (... :shrinking
  :none)` at seed 42.
- **Target:** any contract whose arguments must be drawn together.
- **Wanted:** a counterexample from a generated call shrunk the way a spec-drawn one is.
- **Result:** re-measured with a zero-class-accepting mutant of `contrib-shape` checked
  against the bundle's own `contrib-shape-arguments` generator: `:failed` on trial 1 with
  `(element-count 0 num-rows 25 num-features 12)`, `:shrunk-counterexample nil`,
  `:shrunk-outcome :none`, `:shrink-report (:candidates 0 :budget 100 :termination
  :no-shrinker)`. The same mutant under independently generated arguments shrank to
  `(element-count 0 num-rows 1 num-features 0)` (`:shrunk-outcome :used`).
- **Fallback:** none; the counterexample is read as drawn.
- **Pain:** low for these three-integer calls, higher for a report with nested series.
- **Proposed improvement:** let an `:args-generator` return a value per declared argument so
  each can be shrunk by its own spec.

### G3 -- independent generation misses the interesting case; the case report showed it

- **Status:** not a cl-spec defect as such. Recorded as what `:case-report` did well, plus
  one remaining risk.
- **Target:** `contrib-shape`.
- **Wanted:** both branches exercised, and the zero-class boundary (ELEMENT-COUNT 0 with
  positive rows) drawn, which the docstring singles out.
- **Result (re-measured, a scratch contract identical to the bundle's but without
  `:args-generator`):** at 200 trials, seed 42, the case report read `:derivable 3`,
  `:underivable 197` -- visible at once, where a bare `:passed` would have hidden it. At 1000
  trials the boundary was drawn once at seed 42, once at seeds 2 and 4, and not at all at
  seeds 1 and 3. A mutant that accepts a zero-class shape **passed at the CI budget of 200
  trials** at seed 42 and was caught only at trial 280 of 1000. Planning had recorded the
  boundary as "not drawn in 1000 trials"; at seed 42 that does not reproduce, though at two
  of five seeds it does. The bundle's biased generator (half constructed shapes, a quarter at
  ELEMENT-COUNT 0) catches the same mutant on trial 1 and calls the cases 99/101.
- **Fallback:** `:args-generator` -- which is what costs shrinking (G2).
- **Pain:** the right generator took one iteration once the case report was read.
- **Remaining risk:** the zero-class boundary is *inside* the `:underivable` case. A case
  report counts cases, not boundaries within one, so nothing would have flagged it had the
  case split been drawn differently. `tests/prediction-shape.lisp` still pins it as an
  example.

### G4 -- an `and` of a type and a `satisfies` validates but does not generate

- **Status:** blocked (worked around). **Recurred:** no -- `series-values` only.
- **Target:** the `values` slot of a training series.
- **Wanted:** one spec, the slot's documented type: `(and (type simple-vector) (vector-of
  (satisfies double-or-nil-p)))`.
- **Result (re-measured):** `cl-spec:sample` on `series-values` signals
  `generator-unavailable` ("no conjunct has an ordinary generator strategy").
- **Fallback:** a second spec for the same domain, `series-values-input` = `(vector-of
  (nullable finite-double) :max-length 6)`, generable because `vector-of` happens to produce
  simple-vectors (three samples at seed 42: `(simple-vector 6)`, `(simple-vector 1)`,
  `(simple-vector 3)`). `series-values` validates results; `series-values-input` draws
  arguments.
- **Pain:** two names for one concept, which must be kept in step by hand.

### G5 -- all-`&key` targets: generated calls omit keys

- **Status:** blocked (worked around). **Recurred:** yes -- both constructors,
  `make-training-series` and `make-training-report`.
- **Target:** the two report constructors, which take only keyword arguments.
- **Wanted:** "every key the constructor needs is supplied" stated once, with the keys
  generated from their declared specs.
- **Result (re-measured, scratch contracts with the bundle's `:args` and no
  `:args-generator`):** a generated keyword call may omit any key. With `:pre (and index
  metric values)`, `make-training-series` refused **175 of 200** generated calls at seed 42.
  With `:pre (and series-p num-rounds-p)` over supplied-p variables, `make-training-report`
  refused **150 of 200** (planning had recorded 146 under a `:pre` that was not kept; 150 is
  this re-run's number, for the `:pre` just stated). A demand on one key alone, `:pre
  num-rounds`, still refused 104.
- **Fallback:** the supplied-p variables and the `:pre` stay in each contract -- they are what
  the contract *admits* -- and a whole-call `:args-generator` for each constructor supplies
  those keys, so 0 generated calls are refused, at the cost of G2. The first version of the
  bundle dropped the `:pre` along with the rejections, which left the contracts admitting
  `(make-training-series)`; see G8.
- **Also:** both generators, like `normalize-parameters-arguments`, draw a narrower domain
  than their declared `:args` (`best-score` 0..999 as whole doubles, at most three series;
  `normalize-parameters` five keys and integer values 0..999). The declared `:args` still
  validate; the value-type breadth for `normalize-parameters` lives in its Properties, which
  draw from `parameter-value`.
- **Proposed improvement:** a way to mark a keyword argument as always supplied during
  generation.

### G6 -- a generator body cannot draw from a registered spec

- **Status:** would be nice. **Recurred:** no -- `draw-values` in
  `specs/training-report.lisp`.
- **Target:** `training-series-generator` and the constructor generators, which need a
  vector of doubles and NILs.
- **Wanted:** draw from `finite-double` (or `series-values-input`) inside a `defgenerator`.
- **Result:** a `defgenerator` body is plain Lisp. `cl-spec:sample` exists, but its docstring
  describes it as "intended for inspecting what a spec admits, from the REPL or from an
  agent", with its own `:seed`, not as a composition call tied to the running generator.
- **Fallback:** `draw-values`, a free-standing function that repeats
  `finite-double-generator`'s scaling scheme with narrower constants.
- **Pain:** a small duplication that can drift.

### G7 -- re-registration versus editability

- **Status:** a tension, not a blocker. **Recurred:** not applicable.
- **Target:** the bundle's layout.
- **Wanted:** both a function that re-registers every definition (cl-spec's own bundle wraps
  its definitions in a `register-specifications` function) and definitions a structural editor
  can address by name.
- **Result:** a `defspec-function` inside a `defun` is invisible to `lisp-edit-form`, which
  addresses top-level forms by `form_type` and `form_name`. Top-level was chosen, and
  `cl-gbdt/specs/all:register-specifications` re-`load`s each specification file from source
  into `cl-spec:*registry*` as bound -- `tests/specs/checks.lisp`'s
  `register-specifications-fills-a-fresh-registry` holds it to that against an empty registry.
  The first version documented `(asdf:load-system "cl-gbdt/specs" :force t)` instead. Measured
  in a fresh `ros run` image after `cl-spec:clear-registry`, that did bring back all 6 contracts
  and 9 Properties -- but by reloading cl-spec itself too (621 redefinitions), and it did so
  equally with `:force` given a list of the specification systems, and with no `:force` at all.
  Whether a load re-evaluates a dependency is up to the ASDF plan, not something ASDF promises
  for `:force t`, and in a long-lived worker redefining cl-spec's classes under live objects is
  worse than the stale registry it cures. So the documented route had to go either way.
- **Pain:** a reload replaces what it loads and leaves the rest, which is also why a long-lived
  worker can hold stale registrations -- Task 1's first
  `run-tests` failed on five leftover planning-session Function Specs until the worker was
  reset, and `tests/specs/checks.lisp`'s registry-completeness test is what caught it.

### G8 -- a whole-call generator hides a declared domain the target cannot take

- **Status:** a defect in this bundle's first version, found in PR review, and fixed. The
  cl-spec half is what made it invisible. **Recurred:** yes -- four contracts in two files.
- **Target:** `normalize-parameters`, both report constructors, `training-report-from-history`.
- **Wanted:** a contract's declared argument domain -- its `:args` specs and `:pre` -- to be
  one the target actually accepts, whatever the generator happens to draw.
- **Result:** four contracts admitted calls their targets cannot take, and every seed-42 run
  passed, because each run only saw what its generator drew. `cl-spec:check-call` against the
  first version's contracts:
  - `normalize-parameters` on `(42 7)`: the element spec `(or parameter-key parameter-value)`
    admits a number at a key position; `:error` (the target cannot name `42`).
  - `make-training-series` with no arguments, and `make-training-report` with none: every
    key is optional in an `&key` contract, and the omitted ones are NIL, which the slots do not
    admit; `:failed` on the return spec.
  - `make-training-series` with an adjustable `:values` vector: `(vector-of ...)` admits it,
    the constructor stores it as given, and the return spec's `simple-vector` refuses it;
    `:failed`.
  - `training-report-from-history` with an entry `#(0 "l2" 1d0)`: `tuple` admits a vector,
    and the target destructures a list.

  And one in the other direction, found while writing the regression test: `history-entry`
  named its metric `(member "l2" "auc" "binary_logloss")`, and `member` compares with EQL, so
  it admitted only those three string objects -- a freshly made `"l2"`, as any backend
  returns, was refused (`invalid-call-arguments`). It passed only because the generator
  handed back the very literals.
- **Fix:** `:pre (keys-are-keywords-p plist)` for `normalize-parameters`; supplied-p
  variables and `:pre` for the constructors' required keys (G5); `(and (type simple-vector)
  (vector-of ...))` for `series-values-input`; `(and (type list) (tuple ...))` for every
  tuple a helper destructures; a `metric-name` spec that validates any string and generates
  fresh ones from three names. `tests/specs/checks.lisp`'s
  `declared-domains-exclude-what-the-targets-cannot-take` pins each call above as refused or
  rejected, and the fresh `"l2"` as admitted.
- **Pain:** nothing in a passing run points at this. cl-spec checks each generated call
  *against* `:args` -- `invalid-generated-arguments` catches a generator that strays outside
  -- but not the converse, and with `:args-generator` the declared domain is simply never
  sampled. The only probe was writing counterexample calls by hand for `check-call`.
- **Proposed improvement:** a way to sample a contract's declared domain independently of its
  `:args-generator` -- for instance, a check mode that also draws from the `:args` specs and
  reports admitted calls whose target signals or fails its return spec -- and a documentation
  note that `member` over strings validates by identity.

## Findings about cl-gbdt

**`normalize-parameters` rendered a symbol value under the caller's `*print-case*`. Fixed on
this branch.** Before the fix:

```lisp
(normalize-parameters '(:boosting :gbdt))                        ; => (("boosting" . "GBDT"))
(let ((*print-case* :downcase)) (normalize-parameters '(:boosting :gbdt)))
                                                                 ; => (("boosting" . "gbdt"))
(with-standard-io-syntax (normalize-parameters '(:boosting :gbdt)))
                                                                 ; => (("boosting" . "GBDT"))
```

`parameter-value`'s docstring, which `normalize-parameters` calls for every value, said the
printer specials are bound locally "so the result cannot depend on bindings already in force
in the caller". That held for integers, floats, ratios, strings and booleans; a symbol fell
through to `princ-to-string` with only `*print-base*` and `*print-radix*` bound, so
`*print-case*` leaked. It was found while writing the printer Property: generating symbol
values and binding `*print-case*` made the Property fail at once.

The fix renders a symbol other than `T` and `NIL` as its `symbol-name` (`src/parameters.lisp`,
`parameter-value`), which is what the default printer produced, so a caller that never bound
`*print-case*` sees no change. It was made test-first:

- The Property's domain was widened first -- `parameter-value` in `specs/values.lisp` gained
  `(member :gbdt :dart :rf :binary :multiclass)`, `denotes-p` a symbol clause, and the printer
  Property a `*print-case*` argument. Against the unfixed implementation, at seed 42, it
  failed on trial 4 and shrank to `(pairs ((:learning-rate :gbdt)) base 2 radix t float-format
  double-float print-case :capitalize)`.
- A Rove regression test,
  `tests/parameters.lisp`'s `normalize-parameters-output-is-independent-of-the-caller-s-print-case`,
  failed under `:downcase` and `:capitalize`.
- After the fix both pass; the Property replays at seed 42 with the same digest
  (`292986cea7676c83`, `definition_match`, reproduction faithful) and passes 200/200.

## Mutation evidence

A passing Property is only evidence if it can fail. Each of the three below replaces one
function for the duration of one call, runs one Property at seed 42 and the `:normal`
profile, and restores the original in `unwind-protect`. The snippets were written in the
plan's Task 2-4 briefs, which are git-ignored, so they are pasted here verbatim with this
re-run's output. The worker was killed afterwards, so nothing from these runs, or from the
scratch contracts behind G2, G3 and G5, outlived the session.

### `normalize-parameters` without its printer bindings

In package `cl-gbdt/specs/parameters`:

```lisp
(let ((original (fdefinition 'normalize-parameters)))
  (unwind-protect
       (progn
         (setf (fdefinition 'normalize-parameters)
               (lambda (plist)
                 (loop :for (key value) :on plist :by #'cddr
                       :collect (cons (substitute #\_ #\- (string-downcase (string key)))
                                      (princ-to-string value)))))
         (cl-spec:property-result-status
          (cl-spec:run-property 'normalize-parameters-ignores-the-caller-s-printer
                                :profile :normal :seed 42)))
    (setf (fdefinition 'normalize-parameters) original)))
;; => :FAILED
```

The same mutant and seed, reading the result's evidence: failed on trial 1 of 200;
counterexample `(pairs ((:learning-rate 1484/73) (:num-leaves :dart) (:num-class 4459/139)
(:lambda-l1 "P2tcg6Y1z") (:objective 309/46) (:max-depth -271.05573)) base 5 radix t
float-format single-float print-case :upcase)`, shrunk (`:used`) to `(pairs ((:num-leaves
309/46)) base 2 radix nil float-format double-float print-case :downcase)` -- a ratio printed
as `"309/46"`, and in binary, instead of as a decimal.

### `objective-parameters` matching aliases by prefix

In package `cl-gbdt/specs/objective`:

```lisp
(let ((original (fdefinition 'objective-parameters)))
  (unwind-protect
       (progn
         (setf (fdefinition 'objective-parameters)
               (lambda (parameters)
                 (append (loop :for (key value) :on parameters :by #'cddr
                               :unless (some (lambda (alias)
                                               (eql 0 (search alias
                                                              (substitute
                                                               #\_ #\-
                                                               (string-downcase
                                                                (string key))))))
                                             *objective-aliases*)
                                 :append (list key value))
                         (list :objective "none"))))
         (let ((result (cl-spec:run-property
                        'objective-parameters-keeps-every-other-entry-in-order
                        :profile :normal :seed 42)))
           (list (cl-spec:property-result-status result)
                 (cl-spec:trial-observation-arguments
                  (cl-spec:property-result-shrunk-evidence result)))))
    (setf (fdefinition 'objective-parameters) original)))
;; => (:FAILED (((:APPS -271.05573))))
```

Failed on trial 1; the original counterexample was `(pairs ((:loss 1484/73) (:objective-type
:dart) (:num-class 4459/139) (:losses "P2tcg6Y1z") ("App" 309/46) (:applications
-271.05573)))`. The shrinker reached the near miss `:apps` -- which the prefix match wrongly
drops as `app` -- and left the single-float value as drawn (G1).

### `training-report-from-history` dropping NIL values

In package `cl-gbdt/specs/history`:

```lisp
(let ((original (fdefinition 'training-report-from-history)))
  (unwind-protect
       (progn
         (setf (fdefinition 'training-report-from-history)
               (lambda (history num-rounds dataset-names &rest keys)
                 (let ((report (apply original history num-rounds dataset-names keys)))
                   (cl-gbdt/src/training-report:make-training-report
                    :num-rounds num-rounds
                    :series (mapcar (lambda (series)
                                      (cl-gbdt/src/training-report:make-training-series
                                       :index (training-series-index series)
                                       :name (training-series-name series)
                                       :metric (training-series-metric series)
                                       :values (remove nil (training-series-values series))))
                                    (training-report-series report))))))
         (let ((result (cl-spec:run-property
                        'history-series-values-are-the-pair-s-values-in-order
                        :profile :normal :seed 42)))
           (list (cl-spec:property-result-status result)
                 (cl-spec:trial-observation-arguments
                  (cl-spec:property-result-shrunk-evidence result)))))
    (setf (fdefinition 'training-report-from-history) original)))
;; => (:FAILED ((((0 "l2" NIL))) (NIL "RBdpN84rX" NIL NIL)))
```

Failed on trial 1, from a six-iteration history; HISTORY shrank (`:used`) to one iteration
holding one entry with a NIL value. DATASET-NAMES was left as drawn, since the Property makes
no claim about it.

## What was deliberately not done

Out of scope for this first pass, per the plan's section 18:

- **No change to cl-spec.** Every gap above is recorded, not fixed; the improvement lines are
  proposals only.
- **No generic-function DSL, no MOP-inferred slots or initargs, no generic-function
  instrumentation.** `defspec-function` calls a symbol's function, so a generic function
  could be checked through its dispatch, but none of the 13 unified generics was specified
  here.
- **No resource-lifecycle framework, and no generators for backends, datasets or boosters.**
  Those own foreign memory; nothing in the bundle opens a shared library, and
  `tools/ci/run-tests.lisp` fails `cl-gbdt/tests/specs` if one is opened.
- **No large functional property-based testing,** and no backend-comparing Property.
- **No existing Rove test removed.**
- **Phase 4 deferred:** CLOS objects as inputs -- `early-stopping-watcher` and its
  `observe-iteration` as a stateful contract (`:capture`, `:state-post`) -- is the next step,
  not this one.
