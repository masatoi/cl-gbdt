;;;; protocol.lisp --- XGBoost backend, Layer 2: all thirteen methods of the unified API's
;;;; protocol, each delegating its C calls to `cl-gbdt/src/xgboost/native', or its whole
;;;; procedure to `cl-gbdt/src/xgboost/api', or both.
;;;;
;;;; The backend's CLOS classes and the `initialize-backend'/`shutdown-backend' pair that opens
;;;; and closes the shared library are Layer 1, not Layer 2, and live in
;;;; `cl-gbdt/src/xgboost/classes' -- see that file's header for why. `slice-model' is Layer 1
;;;; as well and lives in `cl-gbdt/src/xgboost/api' beside the other finished operations.
;;;;
;;;; A method here owns the PORTABLE CONTRACT and nothing else: the checks and translations
;;;; that exist because a unified generic promised a portable argument. The procedure a
;;;; finished operation performs is Layer 1 and lives in `cl-gbdt/src/xgboost/api' --
;;;; `make-dataset', which checks :MISSING and :CATEGORICAL-FEATURES, refuses :REFERENCE and
;;;; :PARAMETERS, renders the feature-type strings and then calls that file's `create-dataset';
;;;; `predict', which checks :MISSING's capability and resolves :BEST and then calls that
;;;; file's `predict'; `save-model' and `model-to-string', which resolve :BEST and refuse a
;;;; :NUM-ITERATION this library has no route for and then call that file's functions of the
;;;; same name; `feature-importance', which refuses :BEST explicitly and then any other
;;;; :NUM-ITERATION the same way, before calling that file's `feature-importance'; and
;;;; `free-dataset', `update-one-iteration', `free-booster', `load-model', `evaluation',
;;;; `dataset-num-rows' and `dataset-num-features', whose whole bodies were procedure and
;;;; delegate entirely. A caller who loaded `cl-gbdt/xgboost' alone reaches those functions
;;;; with no method here in the image at all.
;;;;
;;;; `train' delegates too, and is the one method that also writes a result back afterward:
;;;; it calls `cl-gbdt/src/xgboost/api''s `create-booster' for its whole construction, then
;;;; writes the best iteration the early-stopping watcher found, through the internal
;;;; `%set-booster-best-iteration' in `cl-gbdt/src/handle', once the loop ends. See the
;;;; comment at its creation call for why that write could not happen any earlier.

(uiop:define-package #:cl-gbdt/src/xgboost/protocol
  (:use #:cl)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt/src/xgboost/native
                #:%check-backend-open
                #:%check-xgboost-dataset
                #:%check-unsupported
                #:%dataset-num-rows
                #:%boosted-rounds
                #:%update-one-iteration
                #:%booster-predictions
                #:%train-one-iteration-custom
                #:%read-evaluation)
  (:import-from #:cl-gbdt/src/xgboost/classes
                #:xgboost-backend
                #:xgboost-dataset
                #:xgboost-booster)
  ;; Layer 1's finished operations whose name collides with a `cl-gbdt/src/protocol' generic
  ;; are deliberately absent from this clause: `free-dataset', `update-one-iteration',
  ;; `free-booster', `predict', `save-model', `load-model', `model-to-string',
  ;; `feature-importance', `evaluation', `dataset-num-rows' and `dataset-num-features' -- the
  ;; `:import-from #:cl-gbdt/src/protocol' below names each of those as a GENERIC FUNCTION, and
  ;; each pair is two different symbols -- importing both would be a name conflict, not a
  ;; re-import. The twelve methods that need the Layer 1 functions name them in full: those
  ;; eleven, plus `train', whose own name does not collide but whose cleanup still names the
  ;; doubled `free-booster' in full.
  (:import-from #:cl-gbdt/src/xgboost/api
                #:%creation-function-name
                #:create-booster
                #:create-dataset)
  (:import-from #:cl-gbdt/src/backend
                #:backend-name
                #:backend-supports-p)
  (:import-from #:cl-gbdt/src/protocol
                #:make-dataset
                #:dataset-num-rows
                #:dataset-num-features
                #:free-dataset
                #:train
                #:update-one-iteration
                #:predict
                #:save-model
                #:load-model
                #:model-to-string
                #:feature-importance
                #:evaluation
                #:free-booster)
  (:import-from #:cl-gbdt/src/handle
                #:handle-live-pointer
                #:handle-backend
                #:%resolve-best-num-iteration
                #:%reject-best-num-iteration
                #:%set-booster-best-iteration)
  (:import-from #:cl-gbdt/src/conditions
                #:unsupported-argument
                #:capability-unavailable)
  (:import-from #:cl-gbdt/src/config/categorical-features
                #:categorical-feature-types)
  ;; `check-objective-result' only. `cl-gbdt/src/config/objective' also exports
  ;; `objective-parameters', which rewrites a parameter plist's `objective' entry to "none" --
  ;; that is LightGBM's alone, forced there because `LGBM_BoosterUpdateOneIterCustom' refuses
  ;; to run while the booster holds an objective function. `XGBoosterTrainOneIter' accepts a
  ;; custom update with any objective set, so this backend rewrites nothing and importing the
  ;; second symbol would leave an unused name suggesting otherwise.
  (:import-from #:cl-gbdt/src/config/objective
                #:check-objective-result)
  (:import-from #:cl-gbdt/src/training/custom-metric
                #:custom-metric-entry
                #:check-metric-name-collision
                #:make-metric-name-pin
                #:pin-metric-name)
  (:import-from #:cl-gbdt/src/training/history
                #:training-report-from-history)
  (:import-from #:cl-gbdt/src/training/early-stopping
                #:train-early-stopping-watcher
                #:observe-iteration
                #:watcher-best-iteration
                #:watcher-best-score
                #:watcher-stopped-p)
  (:import-from #:cl-gbdt/src/foreign
                #:with-foreign-float-traps-masked)
  (:export #:xgboost-backend))

(in-package #:cl-gbdt/src/xgboost/protocol)

;;; ---------------------------------------------------------------------------
;;; Floating-point trap safety
;;;
;;; Every method below that reaches into libxgboost.so -- all thirteen protocol methods --
;;; wraps its entire body in `with-foreign-float-traps-masked'. `initialize-backend'
;;; (`XGBoostVersion') and `shutdown-backend' (closing the library can run its own static
;;; finalizers) are wrapped exactly the same way in `cl-gbdt/src/xgboost/classes', and every
;;; Layer 1 operation in `cl-gbdt/src/xgboost/api' -- `create-dataset', `free-dataset',
;;; `create-booster', `update-one-iteration', `free-booster', `predict' and the public
;;; `slice-model' -- the same way again, where they live; this rule is the backend's,
;;; not this file's. A method that now delegates its procedure to `api' keeps its own wrap
;;; regardless: the two nest harmlessly, and dropping it would make the wrap depend on what the
;;; callee happens to do today.
;;; See that macro's docstring in `cl-gbdt/src/foreign' for why: SBCL enables
;;; floating-point traps by default on x86-64 and not on aarch64, and XGBoost's own numeric
;;; code -- confirmed for the softmax normalization behind a `multi:softprob' prediction --
;;; was written and tested against the C convention of those traps staying masked.
;;; Method-body granularity, not per-call, so a call added later inside an already-wrapped
;;; method cannot reopen this gap by omission. Every actual C call a method below makes goes
;;; through `cl-gbdt/src/xgboost/native', but the mask is established here, around the whole
;;; method body, not inside that file -- see its own header.

;;; ---------------------------------------------------------------------------
;;; The `:missing-value' gate

(defun %check-missing-value (backend)
  "Signal `capability-unavailable' when BACKEND's `:missing-value' capability reads false.

Policy section 7 requires the operation itself to re-check a capability rather than trusting
the caller to have asked `backend-supports-p' first -- the same rule
`cl-gbdt/src/xgboost/api''s `%check-sparse-input' follows for `:sparse-input'. This backend
answers true unconditionally, which does not make the check redundant: it is what keeps the
two backends' code saying the same thing, so `make-dataset' here and `make-dataset' in
`cl-gbdt/src/lightgbm/protocol' gate the argument identically and neither has to be read to
know what the other does. Mirrors that file's function of the same name.

Only a non-NIL :MISSING ever reaches this. NIL means the backend's own default sentinel --
what every caller has always got -- so a caller who passes nothing needs no capability at
all."
  (unless (backend-supports-p backend :missing-value)
    (error 'capability-unavailable
           :backend (backend-name backend) :capability :missing-value)))

;;; ---------------------------------------------------------------------------
;;; The `:categorical-features' gate

(defun %check-categorical-features (backend)
  "Signal `capability-unavailable' when BACKEND's `:categorical-features' capability reads
false.

Policy section 7 requires the operation itself to re-check a capability rather than trusting
the caller to have asked `backend-supports-p' first -- the same rule `%check-missing-value'
above and `cl-gbdt/src/xgboost/api''s `%check-sparse-input' follow for their own. This backend
answers true unconditionally, which does not make the check redundant: it is what keeps the
two backends' code saying the same thing, so `make-dataset' here and `make-dataset' in
`cl-gbdt/src/lightgbm/protocol' gate the argument identically and neither has to be read to
know what the other does.

Only a non-NIL :CATEGORICAL-FEATURES ever reaches this. NIL means what every caller has always
got -- no feature-type vector attached at all, every column read as a quantity -- so a caller
who passes nothing needs no capability."
  (unless (backend-supports-p backend :categorical-features)
    (error 'capability-unavailable
           :backend (backend-name backend) :capability :categorical-features)))

;;; ---------------------------------------------------------------------------
;;; The `:custom-objective' gate

(defun %check-custom-objective (backend objective)
  "Signal `capability-unavailable' when OBJECTIVE is non-NIL and BACKEND does not provide
`:custom-objective', and `unsupported-argument' when OBJECTIVE is non-NIL and is not a
function.

Only a non-NIL OBJECTIVE ever reaches either error: NIL means what every caller has always
got, the library computing its own gradient, so a caller who passes nothing needs no
capability and cannot fail the type check either.
Policy section 7 requires the operation itself to re-check rather than trusting the caller to
have asked `backend-supports-p' first -- the same rule `%check-missing-value' and
`%check-categorical-features' above, and `cl-gbdt/src/xgboost/api''s `%check-sparse-input',
follow for their own. Mirrors `cl-gbdt/src/lightgbm/protocol''s function of the same name.

Like LightGBM's, this backend's answer is PROBED rather than declared: `XGBoosterTrainOneIter'
-- the entry point `train''s custom loop makes its update through -- is named in
`*optional-symbols*' rather than `*required-symbols*', so an XGBoost too old to export it
opens normally and reads false here. See that variable's own docstring for why the entry
belongs there and not in `*provided-capabilities*', and for why one C name covers this where
LightGBM's entry needs three.

Refusing rather than falling back is what keeps a caller from silently getting a run boosted
against `reg:squarederror' when they asked for their own loss, which is the silent fallback
policy section 7 forbids. That matters more here than on LightGBM: this backend does not
rewrite PARAMETERS, so an ignored OBJECTIVE would leave a perfectly ordinary configured
objective training a perfectly ordinary model, with nothing about the result to show the
caller's function was never called.

The type check is here, beside the capability check, rather than left to the `funcall' in
`train''s loop. By then the booster handle exists and one iteration's scores have already
been read out of the library -- at the cost of a full prediction pass, on this backend -- so
`:objective 42' would surface as SBCL's own untyped `type-error' from mid-loop, naming
neither the argument nor the backend, where every other malformed argument here signals
`unsupported-argument' before any foreign call. `functionp' rather than a `function' type
declaration: a symbol naming a function is NOT accepted, since `funcall' would resolve it
against whatever global definition happened to be in force at each iteration rather than
against what the caller passed. Mirrors `cl-gbdt/src/lightgbm/protocol''s guard exactly, down
to the ARGUMENT string, so the two backends refuse the same value with the same report."
  (when objective
    (unless (backend-supports-p backend :custom-objective)
      (error 'capability-unavailable
             :backend (backend-name backend) :capability :custom-objective))
    (unless (functionp objective)
      (error 'unsupported-argument
             :backend (backend-name backend)
             :argument "train's :objective"
             :reason (format nil "the custom objective must be a function of one argument, ~
                                  or NIL for the library's own gradient -- got ~S"
                             objective)))))

;;; ---------------------------------------------------------------------------
;;; The `:custom-evaluation' gate

(defun %check-custom-evaluation (backend evaluation record-history)
  "Signal `capability-unavailable' when EVALUATION is non-NIL and BACKEND does not provide
`:custom-evaluation', and `unsupported-argument' when EVALUATION is non-NIL and either
RECORD-HISTORY is NIL or EVALUATION is not a function.

Only a non-NIL EVALUATION ever reaches any of the three: NIL means what every caller has
always got, the library's own metrics and nothing else, so a caller who passes nothing needs
no capability and cannot fail either check. Policy section 7 requires the operation itself to
re-check rather than trusting the caller to have asked `backend-supports-p' first -- the same
rule `%check-missing-value', `%check-categorical-features' and `%check-custom-objective'
above, and `cl-gbdt/src/xgboost/api''s `%check-sparse-input', follow for their own. Mirrors
`cl-gbdt/src/lightgbm/protocol''s function of the same name, down to the ARGUMENT string, so
the two backends refuse the same value with the same report.

UNLIKE `%check-custom-objective' beside it, and unlike LightGBM's function of this name, the
capability this reads is DECLARED rather than probed: `XGBoosterPredictFromDMatrix' -- all
`%booster-predictions' needs -- is in `*required-symbols*', so `:custom-evaluation' is named
in `*provided-capabilities*' and no open XGBoost reads false here. See that variable's own
docstring for why, and for why the sibling backend answering the same capability out of its
`*optional-symbols*' is not a disagreement between the two. The check is not thereby redundant:
it is what keeps the two backends' code saying the same thing, the same reasoning
`%check-missing-value' and `%check-categorical-features' above carry for their own
unconditionally-true capabilities, and it is reachable -- a caller who overwrites
`backend-capabilities' gets this refusal, which is how
`withdrawing-the-capability-makes-train-refuse' in tests/functional/custom-evaluation.lisp
reaches it on both backends.

RECORD-HISTORY NIL is refused rather than silently ignored, for exactly the reason
`train-early-stopping-watcher' refuses the same pair: a custom metric's whole result is the
per-iteration series RECORD-HISTORY NIL exists not to build, and the values would be computed
at full cost -- a prediction pass per dataset per iteration on this backend -- and then
dropped.

The type check is here, beside the capability check, rather than left to the `funcall' in
`train''s loop, for the same reason `%check-custom-objective' gives: by then a booster handle
exists and one dataset's predictions have already been read out of the library, so
`:evaluation 42' would surface as SBCL's own untyped `type-error' from mid-loop, naming
neither the argument nor the backend. `functionp' rather than a `function' type declaration:
a symbol naming a function is NOT accepted, since `funcall' would resolve it against whatever
global definition happened to be in force at each iteration rather than against what the
caller passed."
  (when evaluation
    (unless (backend-supports-p backend :custom-evaluation)
      (error 'capability-unavailable
             :backend (backend-name backend) :capability :custom-evaluation))
    (unless record-history
      (error 'unsupported-argument
             :backend (backend-name backend)
             :argument "train's :evaluation"
             :reason (format nil "a custom metric is recorded per iteration, which ~
                                  :record-history NIL skips; pass :record-history T, or ~
                                  drop :evaluation")))
    (unless (functionp evaluation)
      (error 'unsupported-argument
             :backend (backend-name backend)
             :argument "train's :evaluation"
             :reason (format nil "the custom metric must be a function of two arguments, ~
                                  or NIL for the library's own metrics only -- got ~S"
                             evaluation)))))

;;; ---------------------------------------------------------------------------
;;; Datasets

(defmethod make-dataset ((backend xgboost-backend) matrix
                          &key label weight group feature-names parameters reference missing
                            categorical-features)
  "Build an XGBoost dataset (a DMatrix) from MATRIX -- a dense matrix via
`XGDMatrixCreateFromDense', a `csr-matrix' via `XGDMatrixCreateFromCSR' -- attaching LABEL
and WEIGHT with `XGDMatrixSetInfoFromInterface', GROUP with `XGDMatrixSetUIntInfo', and
FEATURE-NAMES with `XGDMatrixSetStrFeatureInfo' when supplied. See the `make-dataset'
generic function's docstring for what each argument means.

Signals `capability-unavailable' when MATRIX is a `csr-matrix' and this backend's
`:sparse-input' capability reads false -- see `cl-gbdt/src/xgboost/api''s `%dataset-pointer',
which checks it. LABEL, WEIGHT, GROUP and FEATURE-NAMES behave identically either way: they
are attached to the finished DMatrix by `create-dataset', which never sees which entry point
built it. REFERENCE and PARAMETERS are refused for a `csr-matrix' exactly as they are for a
dense matrix, and for the same reasons, spelled out below.

MISSING, the value that means *missing*, becomes the `\"missing\"' key of whichever creation
config JSON MATRIX's form reaches. It needs this backend's `:missing-value' capability, which
`%check-missing-value' re-checks below rather than trusting the caller to have asked, and it
signals `unsupported-argument' for anything that is neither a `real' nor NIL -- see
`missing-value-json', which renders it. NIL, the default, sends the IEEE NaN this backend
sent unconditionally before the argument existed, so a caller who passes nothing gets exactly
what they got before. The comparison the library then makes is at SINGLE precision, whatever
MATRIX's own element type: two `double-float's that share a `single-float' both count as
missing against a sentinel that narrows to it.

CATEGORICAL-FEATURES, a list of 0-based column indices, is attached with the same
`XGDMatrixSetStrFeatureInfo' FEATURE-NAMES uses, under the `\"feature_type\"' field instead of
`\"feature_name\"' -- one string per column, `\"c\"' for a named column and `\"q\"' for every
other, as `categorical-feature-types' renders them. It needs this backend's
`:categorical-features' capability, which `%check-categorical-features' re-checks below rather
than trusting the caller to have asked, and it signals `unsupported-argument' naming
`:categorical-features' for an index that is not an integer, is negative, is beyond MATRIX's
last column, or was named twice. NIL, the default, attaches no `\"feature_type\"' at all --
exactly what every call sent before the argument existed, not a vector of `\"q\"'.

The list is rendered from the CALLER's MATRIX, before `create-dataset' builds anything, so a
bad index signals with no DMatrix yet allocated and the range check is made against the same
count `cl-gbdt/src/lightgbm/protocol''s `make-dataset' checks against. The attachment then has
to wait until after creation, `XGDMatrixSetStrFeatureInfo' needing a handle -- which is also
why a `csr-matrix' needs nothing of its own here: the two creation branches have converged by
the time it runs, and the renderer reads a `csr-matrix''s declared column count where it reads
a dense matrix's second dimension.

Measured, and the reason a dataset that builds here can still fail later: `tree_method exact'
refuses categorical features at `train', not at `make-dataset'. The DMatrix is built and the
types attached without complaint, and `XGBoosterUpdateOneIter' then returns -1 with
`Updater `grow_colmaker` or `exact` tree method doesn't support categorical features'. That is
:PARAMETERS' business, not this method's -- `hist' and `approx' both work -- and nothing here
pre-validates an updater it is not given.

REFERENCE and PARAMETERS both signal `unsupported-argument' rather than being silently
dropped: REFERENCE is a LightGBM-only concept -- aligning a new dataset's bin mapper to an
existing one's, which XGBoost has nothing resembling. PARAMETERS is more subtle: the
vendored header (`ffi-spec/xgboost/include/xgboost/c_api.h') documents exactly three keys
for `XGDMatrixCreateFromDense''s config JSON -- `\"missing\"', which now has its own
:MISSING argument above and so is not what a caller reaches for PARAMETERS to set,
`\"nthread\"' and `\"data_split_mode\"' -- none of which correspond to what a caller moving
a working call from LightGBM actually means by dataset-level PARAMETERS there: binning knobs
such as
`max_bin' and `min_data_in_bin'. Forwarding `normalize-parameters''s output into that
config JSON regardless would not raise anything either: confirmed empirically against the
vendored library, `XGDMatrixCreateFromDense' returns success and silently ignores an
unrecognized config key rather than rejecting it, which would just move today's silent
drop one layer deeper, into C, instead of fixing it. The same holds for a `csr-matrix': that
header documents `XGDMatrixCreateFromCSR''s config by cross-reference to
`XGDMatrixCreateFromDense', so it is the same three keys either way, and the refusal below
names whichever of the two the caller's own MATRIX would have reached -- see
`cl-gbdt/src/xgboost/api''s `%creation-function-name', which words that name where the calls
it names are made. Either PARAMETERS or REFERENCE accepted and discarded here would let a
caller move a working `make-dataset' call from LightGBM to XGBoost and get a dataset that
looks fine but was not built the way the caller asked, which is exactly the failure mode this
project keeps finding.

Signals `foreign-call-error' when dataset creation reports success but writes a null
handle -- a library-contract violation, but one every later call through this handle would
otherwise dereference blindly.

Signals `backend-not-open' before any of that when BACKEND is not open -- see
`%check-backend-open'.

The procedure itself is Layer 1 and lives in `cl-gbdt/src/xgboost/api''s `create-dataset':
building the pointer, attaching LABEL, WEIGHT, GROUP, FEATURE-NAMES and the rendered feature
types in that order, and the ownership dance that frees the raw DMatrix when one of those
signals. What is left here is the portable contract -- the three capability checks above, the
two refusals, and rendering CATEGORICAL-FEATURES into the feature-type strings this backend
states them as. Everything the paragraphs above promise about a null handle and about the raw
handle's ownership is that function's doing; see its own docstring."
  (with-foreign-float-traps-masked
    ;; Checked here as well as inside `create-dataset', which cannot omit it either: this
    ;; method's contract is that a closed BACKEND is refused BEFORE the argument checks
    ;; below, so a caller who closed the backend and also passed a bad :CATEGORICAL-FEATURES
    ;; index still gets `backend-not-open' rather than `unsupported-argument'.
    (%check-backend-open backend)
    (when missing
      (%check-missing-value backend))
    (when categorical-features
      (%check-categorical-features backend))
    (%check-unsupported
     backend "make-dataset's :reference" reference
     "XGBoost has no bin-mapper alignment; :reference is a LightGBM-only concept")
    (%check-unsupported
     backend "make-dataset's :parameters" parameters
     (format nil "~A's config JSON only recognizes missing/nthread/data_split_mode, none ~
                   of which are LightGBM's dataset-level binning parameters, and the ~
                   library silently ignores any other key rather than rejecting it"
             (%creation-function-name matrix)))
    ;; Rendered before creation, attached after: the renderer takes the caller's own MATRIX,
    ;; so a bad index signals here with nothing yet allocated, while the attachment needs a
    ;; DMatrix handle to attach to. See this method's docstring. `create-dataset' below is
    ;; handed the finished strings: they are this backend's own vocabulary by then, and Layer
    ;; 1 neither knows nor translates them.
    (let ((feature-types (categorical-feature-types categorical-features matrix
                                                    (backend-name backend))))
      (create-dataset backend matrix :label label :weight weight :group group
                                     :feature-names feature-names
                                     :missing missing :feature-types feature-types))))

(defmethod dataset-num-rows ((dataset xgboost-dataset))
  "Return DATASET's row count, read via `XGDMatrixNumRow'.

This method's whole body was procedure -- there was no portable argument here to check or
translate -- so all of it is `cl-gbdt/src/xgboost/api''s `dataset-num-rows', which is where
the class guard the specializer above used to provide now lives too."
  (with-foreign-float-traps-masked
    ;; Named in full, not imported: `cl-gbdt/src/protocol''s `dataset-num-rows', the generic
    ;; this method is defined on, is a DIFFERENT symbol of the same name, and this file
    ;; imports that one.
    (cl-gbdt/src/xgboost/api:dataset-num-rows dataset)))

(defmethod dataset-num-features ((dataset xgboost-dataset))
  "Return DATASET's feature count, read via `XGDMatrixNumCol'. Delegates wholly, as
`dataset-num-rows' above does and for the same reason."
  (with-foreign-float-traps-masked
    (cl-gbdt/src/xgboost/api:dataset-num-features dataset)))

(defmethod free-dataset ((dataset xgboost-dataset))
  "Free DATASET via `XGDMatrixFree'. Does nothing if it was already freed.

Unlike every other operation in this file -- `free-booster' below excepted, which takes this
same path for this same reason -- this does not go through `handle-live-pointer'
and so does not signal `backend-not-open' when DATASET's backend has already been closed
-- see `cl-gbdt/src/lightgbm/protocol''s `free-dataset' for why: this runs from
`with-dataset''s `unwind-protect' cleanup form, and signalling there would replace whatever
condition is already unwinding the stack instead of letting it propagate. It `warn's instead,
the foreign memory being genuinely unreclaimable by then.

This method's whole body was procedure -- there was no portable argument here to check or
translate -- so all of it is `cl-gbdt/src/xgboost/api''s `free-dataset', which is where the
closed-backend branch and the reasoning behind it now live."
  (with-foreign-float-traps-masked
    ;; Named in full, not imported: `cl-gbdt/src/protocol''s `free-dataset', the generic
    ;; this method is defined on, is a DIFFERENT symbol of the same name, and this file
    ;; imports that one. The `with-foreign-float-traps-masked' wrap stays even though the
    ;; callee establishes its own -- the masks nest harmlessly, and every `defmethod' in
    ;; this file carries one by the rule the header states, checked by
    ;; `tools/ci/check-float-traps.lisp'.
    (cl-gbdt/src/xgboost/api:free-dataset dataset)))

;;; ---------------------------------------------------------------------------
;;; Training

(defun %valid-set-name (backend entry)
  "Return the name half of ENTRY, one element of `train''s :VALID-SETS: NIL when ENTRY is
a bare dataset, or ENTRY's car when ENTRY is a (NAME . DATASET) cons and NAME is a string.

Signals `unsupported-argument' naming :VALID-SETS and ENTRY itself, via `%check-unsupported',
when ENTRY is a cons whose car is not a string -- before any foreign call, and before the
dataset half of ENTRY is checked against `xgboost-dataset' by `%check-xgboost-dataset',
which runs afterward on `%valid-set-dataset''s result. That later check is what turns a
cons whose cdr is not this backend's own kind of dataset into `wrong-backend-reference'
instead -- a different mistake from this one, and so a different condition."
  (if (consp entry)
      (let ((name (car entry)))
        (%check-unsupported
         backend "train's :valid-sets" (not (stringp name))
         (format nil "each element must be a dataset or a (string . dataset) cons; ~S's ~
                      car is not a string" entry))
        name)
      nil))

(defun %valid-set-dataset (entry)
  "Return the dataset half of ENTRY, one element of `train''s :VALID-SETS: ENTRY itself
when it is a bare dataset, or its cdr when it is a (NAME . DATASET) cons.

Does not check that the result is an `xgboost-dataset' -- `%check-xgboost-dataset' does
that afterward, on every element `%valid-set-name' has already let through."
  (if (consp entry) (cdr entry) entry))

(defun %recheck-train-datasets (backend dataset valid-sets)
  "Re-run `train''s own opening checks over BACKEND, DATASET and VALID-SETS, and return two
values: DATASET's freshly read live pointer, and a fresh list of the VALID-SETS entries'.

`train''s loop calls this after every `funcall' of a caller-supplied OBJECTIVE and after every
`funcall' of a caller-supplied EVALUATION, which are the only points in the loop where code this
library did not write runs. That code may free the training set -- `free-dataset' from inside
the objective is the case this was found through -- and the pointer `train' read once before the
loop is then a pointer into freed memory that `XGBoosterTrainOneIter' takes as its DMatrix
argument. Measured before this function existed: `Signal 7 received', a bus error killing the
process rather than signalling. The checks are exactly the ones `train' already ran, so the
caller gets `released-handle-error', `backend-not-open' or `wrong-backend-reference' -- the
typed conditions every other freed handle in this library produces -- and never a fault. The
RETURN VALUES are the point of the exercise: re-checking and then going on to use the pointers
read before the loop would fix nothing, so `train' assigns both to the variables it reads from,
and rebuilds its DATASET-POINTERS list out of them.

The VALID-SETS entries are re-checked and returned because the same iteration hands their
pointers to `XGBoosterEvalOneIter' through `%read-evaluation' whenever RECORD-HISTORY is true
-- a freed DMatrix there is the identical use-after-free the training set is, which is why
`%check-booster-datasets-live' guards both on the public `update-one-iteration' path.
`XGBoosterPredictFromDMatrix', which `%custom-evaluation-entries' below reaches through
`%booster-predictions' with each dataset's own pointer in turn, dereferences that pointer the
same way and is covered by exactly the same re-check -- which is why that function calls this
one between two consecutive datasets' reads rather than only once per iteration, and uses the
list this returns for the next read rather than the one it was handed.

BACKEND itself is re-checked with `%check-backend-open' because `close-backend' unmaps the
shared library and the objective can call it. `handle-live-pointer' already refuses a handle
whose OWN backend has been closed, which covers the ordinary case where DATASET was built by
BACKEND; the check here is what covers the case `%check-xgboost-dataset' documents as
legitimate and therefore does not catch -- a dataset built by a second `xgboost-backend'
instance over the same library, whose own backend is still open while BACKEND is not. It
costs one slot read per iteration.

Mirrors `cl-gbdt/src/lightgbm/protocol''s function of the same name, which returns the
training pointer alone: that library's own custom update takes no DMatrix argument, its
`%read-evaluation' takes a dataset COUNT rather than pointers, and its `%booster-predictions'
addresses a dataset by LightGBM's own `data_idx' rather than by pointer -- so there is nothing
there for the validation half to be returned for. Every one of the three has a pointer
argument here, which is why this returns both halves and every caller reassigns from them."
  (%check-backend-open backend)
  (let ((train-data-pointer
          (%check-xgboost-dataset backend dataset "train's dataset argument"
                                   'xgboost-dataset)))
    (values train-data-pointer
            (mapcar (lambda (valid-set)
                      (%check-xgboost-dataset backend valid-set "a train :valid-sets entry"
                                               'xgboost-dataset))
                    valid-sets))))

(defun %custom-evaluation-entries (backend evaluation booster-pointer dataset valid-sets
                                    dataset-pointers row-counts library-entries
                                    check-collisions-p name-pin)
  "Call EVALUATION once for each of BOOSTER-POINTER's datasets and return two values: the
(DATASET-INDEX METRIC-NAME VALUE) entries the calls produced, in dataset-index order, and a
freshly re-read DATASET-POINTERS list.

DATASET-POINTERS is the training DMatrix's pointer followed by each VALID-SETS entry's, the
same list `train' hands `%read-evaluation', and ROW-COUNTS is those same datasets' row counts
in the same order. ROW-COUNTS is what says how many datasets there are as well as how wide
each one's prediction array is; `train' reads it once before its loop rather than per
iteration, since a built DMatrix's row count cannot change -- `XGDMatrixCreateFromDense' and
`XGDMatrixCreateFromCSR' each build a finished matrix and this API has no entry point that
appends rows to one -- and an integer read before the loop cannot go stale into a fault the
way a pointer can.

Each dataset's predictions come from `%booster-predictions' at `:normal', handed that
dataset's OWN pointer and OWN row count -- the 40-row training set's and the 17-row validation
set's are different arrays, and deriving one from the other would silently mis-shape what the
caller's function is handed. `:normal' rather than the `:raw' `train''s OBJECTIVE branch reads
through the same function: what a metric is handed is that dataset's prediction, the number
`predict' returns for the same rows. See `%booster-predictions', which records both
measurements.

EVALUATION is called with that array and the dataset's index, and must return two values, a
metric name and a value; `custom-metric-entry' checks both and builds the entry. The SECOND
value returned here is the point of the re-check that follows every call: EVALUATION is caller
code and may free a dataset or close BACKEND, so `%recheck-train-datasets' runs the moment it
returns -- before the NEXT dataset's `XGBoosterPredictFromDMatrix', and before `train' uses any
of these pointers again -- and both this loop and `train' read the list it returns rather than
the one they started with. That matters more here than on LightGBM, whose reader addresses a
dataset by `data_idx' and so holds no pointer to go stale. See `%recheck-train-datasets' for
what each of its three re-checks covers.

CHECK-COLLISIONS-P runs `check-metric-name-collision' against LIBRARY-ENTRIES, this
iteration's own `%read-evaluation' result. `train' passes true on the first iteration only:
what the LIBRARY reports cannot be known before a booster has produced one real evaluation,
and after that first one the library's own name set does not change.

THAT SECOND HALF IS THE HINGE, and it is worth meeting here rather than deducing later,
because it is what makes ONE comparison cover a whole run. On this backend it is a property
of where the names come from and not of anything this code does: `%read-evaluation' parses
them out of `XGBoosterEvalOneIter''s own formatted line, through `%parse-eval-result' and
`%split-eval-label', and what that line names is the booster's configured `eval_metric' --
set once through `%set-parameters' before the loop and given no per-iteration input.
NOTHING ASSERTS IT -- it is a fact about the vendored library, not about this library, so a
green suite says nothing about it. An editor moving this check off round 1, or reaching an
XGBoost whose reported names could vary within a run, is changing what makes round 1
sufficient.

The CALLER's names are a separate question and are not covered by that -- an EVALUATION
returning a safe name on iteration 1 and a colliding one afterwards would pass this check --
which is what NAME-PIN below answers.

NAME-PIN is the run's `make-metric-name-pin' table, threaded straight through from `train'
so it outlives the iteration. `pin-metric-name' records each index's name on the first
iteration and refuses a different one on every later iteration, which is what holds a caller
to one name per dataset index for the whole run and so keeps every series exactly as long as
the run. It also subsumes the round-2 collision case rather than needing a second check of
its own: with each index's name fixed at the first iteration, the only name that can ever
reach the library's is the one CHECK-COLLISIONS-P already compared. That argument INHERITS
the dependency above rather than standing free of it -- \"the one already compared\" is the
only one there is precisely because the library's own names do not change either.

Both checks run AFTER `custom-metric-entry' rather than before it, and the order is
load-bearing twice over. First, each compares with `string=', which signals a bare
`type-error' for a name that is not a string designator at all, while `custom-metric-entry' is
what turns that same name into this library's own `unsupported-argument'. Second, and this is
why both are handed `(second entry)' rather than the NAME the caller returned: the name in the
entry is `custom-metric-entry''s own `copy-seq' of it, and the copy does not exist until that
call has been made. A caller returning one string object per iteration and rewriting its
characters in place would defeat both checks AND the history if any of the three held the
caller's object -- see `custom-metric-entry', which measured what that reached.

Mirrors `cl-gbdt/src/lightgbm/protocol''s function of the same name, down to the argument
order and the order of the four steps inside the loop, differing only in taking and returning
a pointer list where that one takes neither: it addresses each dataset by index and returns the
training pointer alone."
  (let ((entries '())
        (pointers dataset-pointers))
    (loop :for index :from 0
          :for rows :in row-counts
          :do (let ((predictions (%booster-predictions booster-pointer (nth index pointers)
                                                        rows :normal)))
                (multiple-value-bind (name value) (funcall evaluation predictions index)
                  (multiple-value-bind (train-data-pointer valid-set-pointers)
                      (%recheck-train-datasets backend dataset valid-sets)
                    (setf pointers (cons train-data-pointer valid-set-pointers)))
                  (let* ((entry (custom-metric-entry (backend-name backend) name value index))
                         (pinned-name (second entry)))
                    (when check-collisions-p
                      (check-metric-name-collision (backend-name backend) pinned-name index
                                                    library-entries))
                    (pin-metric-name (backend-name backend) name-pin pinned-name index)
                    (push entry entries)))))
    (values (nreverse entries) pointers)))

(defmethod train ((backend xgboost-backend) dataset
                   &key valid-sets (num-rounds 100) parameters (record-history t)
                        early-stopping objective evaluation)
  "Train an XGBoost booster on DATASET for up to NUM-ROUNDS boosting iterations, and
return it and a `training-report' of the run.

Builds the booster with `XGBoosterCreate' over DATASET and every VALID-SETS entry's
DMatrix handle together -- see `%create-booster' for why XGBoost takes the whole set up
front rather than adding validation data afterward. Applies PARAMETERS one at a time via
`XGBoosterSetParam', then drives `XGBoosterUpdateOneIter' NUM-ROUNDS times -- or fewer,
when EARLY-STOPPING ends the run first. See the `train' generic function's docstring for
what each argument means, and for what the secondary value holds; NUM-ROUNDS defaults to
100 when not supplied.

Each VALID-SETS element is either a dataset, whose series carry no name, or a
(NAME . DATASET) cons, where NAME is a string that reaches `training-series-name' for
every series recorded at that dataset's index -- see `%valid-set-name' and
`%valid-set-dataset', which split VALID-SETS into two parallel lists, of datasets and of
names, once at the top of this method; everything below reads the datasets list under
the name VALID-SETS, exactly as before this method accepted names at all. Two entries
may legitimately share one NAME: their index, not their name, is what a caller uses to
tell them apart in the report, so this is accepted rather than rejected as a duplicate.
The training set is never a VALID-SETS entry and is always index 0 with a NIL name.

When RECORD-HISTORY is true -- the default -- this reads the whole evaluation after each
iteration through `%read-evaluation': the same function the `evaluation' method calls, over
the same DMatrix pointers in the same order, which is what keeps the history and what
`evaluation' answers afterward from being able to disagree.
`training-report-from-history' folds the run's worth of them into the report once the loop
is done; that fold is backend-neutral and shared with `cl-gbdt/src/lightgbm/protocol''s
`train', so what the two backends record cannot drift apart either. It orders series by the
(DATASET-INDEX, METRIC-NAME) pair's first appearance, which for this backend is the order
`XGBoosterEvalOneIter' formatted its own result in, so the report's series arrive in exactly
the order `evaluation' reports its entries in without anything being sorted. A field the
parse could not read as a `double-float' is recorded as NIL, keeping its place in the series
rather than shortening it -- see `training-series-values'.

RECORD-HISTORY NIL skips that read entirely -- one `XGBoosterEvalOneIter' call per
iteration, plus the parse of the line it formats, which is what makes recording cost real
wall-clock time (see the `train' generic's docstring for the measured figures). The loop is
then exactly the `XGBoosterUpdateOneIter' loop this method ran before it recorded anything,
and the report it still returns as its secondary value has an empty series list over the
same NUM-ROUNDS -- `training-report-from-history' over an empty history, the same shape a
run with `disable_default_eval_metric=1' produces.

Skipping the read also widens what this method accepts, which matters here more than it
does on LightGBM: `XGBoosterEvalOneIter' evaluates every DMatrix it is handed, and refuses
one it cannot evaluate -- an unlabelled DMatrix passed in VALID-SETS is the case this was
found through, which `XGBoosterUpdateOneIter' trains on without complaint while the
evaluation call signals `foreign-call-error' (\"label and prediction size not match\"). With
RECORD-HISTORY true that failure now propagates out of `train' itself, through the
`unwind-protect' below, where before this backend recorded anything it surfaced
only at a later `evaluation' call. RECORD-HISTORY NIL never reaches the evaluation path and
so trains such a configuration exactly as before.

A read that fails propagates, freeing the booster through the `unwind-protect' below rather
than returning a report whose series are shorter than the run: a short series is
indistinguishable from one a buggy loop recorded, and \"one value per iteration\" is the
invariant a caller reading the report relies on.

EARLY-STOPPING watches one of those recorded series and ends the loop once it has stopped
improving -- see the `train' generic function's docstring for the spec's four required
keys, and `train-early-stopping-watcher' for why it cannot be combined with
RECORD-HISTORY NIL.
The watcher sees each iteration's entries exactly as the history records them, off the one
`%read-evaluation' call this loop already makes, so what stopped the run and what the
report shows can never be two different readings. `training-report-num-rounds' needs
nothing extra to report the shortened run: it has counted actual iterations since Phase 3a.

OBJECTIVE replaces `XGBoosterUpdateOneIter' with `XGBoosterTrainOneIter' for every iteration
of the loop, driven by the gradient and Hessian the caller's own function returns -- see the
`train' generic function's docstring for what that function is called with and what it must
return. Signals `capability-unavailable' naming `:custom-objective' for a non-NIL OBJECTIVE
when the capability reads false, before any foreign call: see `%check-custom-objective'
above, which reads the capability rather than this backend's name, and `*optional-symbols*'
for why the answer here is probed rather than declared. OBJECTIVE NIL, the default, reaches
no check and runs exactly the `XGBoosterUpdateOneIter' loop this method has always run.

PARAMETERS is passed through untouched, unlike `cl-gbdt/src/lightgbm/protocol''s `train',
which forces `objective' to \"none\" because `LGBM_BoosterUpdateOneIterCustom' refuses to run
while the booster holds an objective function. `XGBoosterTrainOneIter' has no such
restriction -- measured, a custom update is accepted with any objective set -- so there is
nothing here to override, and this method calls `objective-parameters' nowhere. What that
costs the caller is that the configured objective's PREDICTION TRANSFORM stays in effect:
with `binary:logistic' still set, `predict :kind :normal' on the resulting booster returns
probabilities of a margin the caller's own loss produced, while `:raw' returns that margin.
The generic function's docstring states this as the caller's decision; nothing here signals
or warns about it. `num_class' is likewise just another parameter, and 3 of it alone gives
three output groups -- no `multi:*' objective is needed for a multiclass custom-objective run.

Each iteration reads the booster's current raw scores with `%booster-predictions' at `:raw' --
an `XGBoosterPredictFromDMatrix' margin prediction over the training DMatrix, this library
having no counterpart to LightGBM's `LGBM_BoosterGetPredict' that hands back scores it
already holds. It costs a prediction pass per iteration, which that backend's loop does not
pay. The result reaches OBJECTIVE as a (ROWS GROUPS) `double-float' array, where ROWS comes
from `%dataset-num-rows' on the training set's own pointer and GROUPS is divided out of the
prediction's reported element count by `%predict-ncol' rather than read from a parameter.
What comes back is checked by `check-objective-result' -- `cl-gbdt/src/config/objective''s,
the same backend-neutral pure code LightGBM's `train' calls, so both backends refuse the same
shapes with the same `dimension-mismatch' -- and only then flattened into the C buffers,
ROW-MAJOR on this backend (row I of group K at `(+ (* I GROUPS) K)', which is what an
`__array_interface__' of shape `[ROWS, GROUPS]' means) and converted to `single-float'. Both
the flattening and the score layout are measured; see `%train-one-iteration-custom' and
`%booster-predictions'. The flattening is this method's business and not the caller's:
OBJECTIVE is handed, and returns, a (ROWS GROUPS) array whichever order the library underneath
wants it in -- LightGBM wants the other one.

OBJECTIVE is funcalled inside this method's own `with-foreign-float-traps-masked' body wrap,
so the caller's Lisp arithmetic runs under the masked convention on x86-64 as well as on
aarch64 -- `(/ 1.0d0 0.0d0)' yields infinity there rather than signalling
`division-by-zero'. Nothing about that is specific to a custom objective; it is simply where
in `train' the caller's code now runs. A condition the caller's function does signal
propagates out of `train' through the `unwind-protect' below, freeing the booster handle
rather than orphaning it, exactly as a mid-loop foreign failure does.

An objective that frees a handle this loop depends on, or closes BACKEND, is caught rather
than crashed on: `%recheck-train-datasets' re-runs this method's own opening checks the
moment the `funcall' returns, and TRAIN-DATA-POINTER, VALID-SET-POINTERS and
DATASET-POINTERS are all reassigned from what it returns, so nothing after the caller's code
uses a pointer read before it. See that function for what each of the three re-checks is
for. This is the only place the loop needs it -- the OBJECTIVE NIL branch beside it runs no
caller code at all.

Neither RECORD-HISTORY nor EARLY-STOPPING is disabled by OBJECTIVE, and neither is made
meaningful by it: a metric configured through PARAMETERS relates to the library's own
objective, not to the caller's, and this method neither signals nor warns about that -- see
the `train' generic function's docstring, which states it as the caller's decision.

The argument is accepted by this lambda list rather than being absent from it: `train' is one
generic function, so a method that did not take the keyword at all would answer a caller who
named it with SBCL's `unknown-keyword-argument' rather than with the typed condition every
other unavailable capability on this backend answers with.

EVALUATION adds the caller's own metric to what each iteration records, one call per dataset
per iteration -- see the `train' generic function's docstring for what that function is called
with and what it must return. Signals `capability-unavailable' naming `:custom-evaluation'
when the capability reads false, and `unsupported-argument' naming \"train's :evaluation\" for
RECORD-HISTORY NIL or for a non-function, all three before any foreign call: see
`%check-custom-evaluation' above, and `*provided-capabilities*' for why the answer here is
DECLARED where LightGBM's is probed. EVALUATION NIL, the default, reaches no check and records
exactly what this method has always recorded.

The calls happen after this iteration's own `%read-evaluation' and BEFORE the history push
and the watcher, in `%custom-evaluation-entries' -- so the entries the history keeps and the
entries the watcher sees are one list, and `:early-stopping' can watch a custom metric with
nothing here to arrange it. The custom entries are APPENDED after every library entry, which
is what makes `training-report-from-history''s first-seen ordering put the library's series
first, as a prefix, exactly where `evaluation' reports them; a custom metric never reaches
`evaluation' at all, that method reading only `XGBoosterEvalOneIter'.

Each dataset's predictions are read with `%booster-predictions' at `:normal', over that
dataset's OWN DMatrix pointer and OWN row count -- a fresh `XGBoosterPredictFromDMatrix' pass
per dataset per iteration, this library having nothing that hands back predictions it already
holds. They are `predict :kind :normal''s numbers and not the `:raw' margin OBJECTIVE is
handed, which on this backend are genuinely different numbers in the same run: `train'
rewrites no parameter, so a configured objective's transform stays in effect for both. Both
facts are measured; see `%booster-predictions'. ROW-COUNTS is read once before the loop, from
the same pointers this method already validated, and only when EVALUATION is non-NIL, so a run
that asks for no custom metric makes no extra foreign call at all.

An EVALUATION that frees a handle this loop depends on, or closes BACKEND, is caught the same
way an OBJECTIVE is: `%custom-evaluation-entries' calls `%recheck-train-datasets' the moment
each `funcall' returns -- between two consecutive datasets' reads, not once per iteration --
and TRAIN-DATA-POINTER, VALID-SET-POINTERS and DATASET-POINTERS are all reassigned from the
list it returns, exactly as they are on the OBJECTIVE path.

DATASET and every VALID-SETS entry's dataset half are each run through
`%check-xgboost-dataset' before any foreign call. `train' dispatches on BACKEND, not on
DATASET, so unlike `dataset-num-rows' or `free-dataset' there is no CLOS specializer here
to rule out the wrong kind of handle first -- without this, `handle-live-pointer' would
happily hand `XGBoosterCreate' a booster's own pointer to use as one of its DMatrix
handles. Signals `wrong-backend-reference' when DATASET or a VALID-SETS entry's dataset
half is not an `xgboost-dataset', and `released-handle-error' or `backend-not-open' when
one is but has already been freed or had its own backend closed. A VALID-SETS entry that
is a cons with a non-string car never reaches this check at all: `%valid-set-name' signals
`unsupported-argument' for it first, which is the different mistake a malformed name is,
kept distinct from a wrong dataset handle.

The returned booster retains DATASET as its training set and a fresh copy of VALID-SETS
as its validation sets, keeping all of them alive for the booster's lifetime and letting
`update-one-iteration' notice if any is freed out from under it -- see
`%check-booster-datasets-live'. The copy matters: VALID-SETS is the caller's own list,
and `make-handle' would otherwise store that exact list object rather than a snapshot of
it. A caller who destructively removes an entry from VALID-SETS after `train' returns --
`delete', `(setf (cdr ...))', reusing the list elsewhere with `nconc' -- would silently
remove it from the booster's view too, since both would be the same cons cells; the
DMatrix `XGBoosterCreate' already attached would then go unchecked by
`%check-booster-datasets-live' even though XGBoost still holds its pointer -- the same
hazard `cl-gbdt/src/lightgbm/protocol''s `train' guards against, for the identical reason.
Free the result with `free-booster' or wrap it in `with-booster'.

BOOSTER is bound to a full handle already: `create-booster' manages the raw-pointer window
between `XGBoosterCreate' and its own `make-handle' call internally -- see its docstring
-- and this method never touches a pointer that let does not already own. What the
`unwind-protect' below manages instead is what happens to that handle from here: any exit
from the loop or the report construction that does not reach the final `setf' frees BOOSTER
rather than orphaning it.

Signals `backend-not-open' before any of that when BACKEND is not open -- see
`%check-backend-open'."
  (with-foreign-float-traps-masked
    (%check-backend-open backend)
    (%check-custom-objective backend objective)
    (%check-custom-evaluation backend evaluation record-history)
    (let* ((valid-set-entries (copy-list valid-sets))
           (train-data-pointer
             (%check-xgboost-dataset backend dataset "train's dataset argument"
                                      'xgboost-dataset))
           (valid-set-names
             (mapcar (lambda (entry) (%valid-set-name backend entry)) valid-set-entries))
           (valid-sets (mapcar #'%valid-set-dataset valid-set-entries))
           (valid-set-pointers
             (mapcar (lambda (valid-set)
                       (%check-xgboost-dataset backend valid-set "a train :valid-sets entry"
                                                'xgboost-dataset))
                     valid-sets))
           (dataset-pointers (cons train-data-pointer valid-set-pointers))
           (dataset-names (cons nil valid-set-names))
           ;; Read once, and only for a run that actually has a custom metric to hand
           ;; predictions to: a built DMatrix's row count cannot change, and an integer read
           ;; here cannot go stale into a fault the way a pointer can. NIL otherwise, so a
           ;; run without EVALUATION makes not one extra foreign call.
           (row-counts (when evaluation
                         (mapcar #'%dataset-num-rows dataset-pointers)))
           ;; One pin for the whole run, so `pin-metric-name' can compare this iteration's
           ;; name against the first iteration's. NIL without EVALUATION, which allocates
           ;; nothing for a run that has no custom metric to hold to a name.
           (name-pin (when evaluation (make-metric-name-pin)))
           ;; Built before `XGBoosterCreate', so a malformed spec -- or one asking for early
           ;; stopping with RECORD-HISTORY NIL -- signals with no raw booster handle in
           ;; existence yet to unwind. NIL when EARLY-STOPPING is NIL, which is what the
           ;; loop below tests to decide whether it can end early at all.
           (watcher (train-early-stopping-watcher (backend-name backend) early-stopping
                                                   record-history dataset-names))
           (history '())
           ;; Counted rather than taken from NUM-ROUNDS: the loop below runs zero iterations
           ;; for a negative count, so a caller passing :NUM-ROUNDS -1 gets an untrained
           ;; booster -- as it did before this branch -- and the report must say 0 ran, not
           ;; -1. `training-report-num-rounds' promises how many iterations actually ran, and
           ;; it is also what makes an early-stopped run report its true, shortened length
           ;; with nothing further to do here.
           (completed-rounds 0))
      ;; Delegated to `cl-gbdt/src/xgboost/api''s `create-booster'. Every method in this file
      ;; now hands its whole procedure to that file; `train' was the last that did not, and
      ;; what held it back was `booster-best-iteration' -- a `:reader'-only slot whose only
      ;; writer was `make-handle''s initarg, at construction, while this method's value comes
      ;; from the watcher after the loop. `cl-gbdt/src/handle''s `%set-booster-best-iteration'
      ;; is what removed the barrier; see its docstring for why it is internal. The barrier was
      ;; a property of the shared `handle' class rather than of either library, which is why
      ;; both backends' `train' carried the same note and both lose it together.
      ;;
      ;; `%set-parameters' is gone from this method with the rest: `create-booster' makes that
      ;; call, in the same position relative to `XGBoosterCreate' -- after it, inside the
      ;; ownership window -- as this method used to.
      ;;
      ;; The datasets are checked twice as a result: once in the `let*' above, with this
      ;; method's own argument descriptions, and once inside `create-booster' with its. The
      ;; first pair is what a caller ever sees; the second can only pass.
      (let ((booster (create-booster backend dataset
                                     :parameters parameters
                                     :valid-sets valid-sets))
            (completed nil))
        (unwind-protect
             ;; Read once, unlike TRAIN-DATA-POINTER and DATASET-POINTERS, which the loop
             ;; refreshes after every caller callback because the caller can free a dataset
             ;; mid-loop. BOOSTER cannot be freed the same way: nothing outside this method
             ;; holds it until the method returns it.
             (let ((booster-pointer (handle-live-pointer booster)))
               ;; ROUND is 1-based, which is the numbering `observe-iteration' answers
               ;; `watcher-best-iteration' in and the report publishes.
               (loop :for round :from 1 :to num-rounds
                     :do (if objective
                             ;; `%boosted-rounds', not ROUND: this is XGBoost's own 0-based
                             ;; `iter' argument, and reading it back from the booster is
                             ;; exactly what `%update-one-iteration' does for the built-in
                             ;; branch beside this one -- see that function for why the
                             ;; count is not tracked locally. ROUND is 1-based and belongs
                             ;; to the report and the early-stopping watcher, not to C.
                             (let ((scores (%booster-predictions
                                            booster-pointer train-data-pointer
                                            (%dataset-num-rows train-data-pointer)
                                            :raw :training t)))
                               (multiple-value-bind (grad hess) (funcall objective scores)
                                 ;; Before anything else this iteration does, and before the
                                 ;; next one reads TRAIN-DATA-POINTER again: the caller's
                                 ;; own code has just run and may have freed a handle this
                                 ;; loop holds a raw pointer to. DATASET-POINTERS is rebuilt
                                 ;; rather than left alone -- `%read-evaluation' below reads
                                 ;; it, and it would otherwise still hold the stale ones.
                                 (multiple-value-setq (train-data-pointer valid-set-pointers)
                                   (%recheck-train-datasets backend dataset valid-sets))
                                 (setf dataset-pointers
                                       (cons train-data-pointer valid-set-pointers))
                                 (check-objective-result grad hess
                                                         (array-dimension scores 0)
                                                         (array-dimension scores 1))
                                 (%train-one-iteration-custom
                                  booster-pointer train-data-pointer
                                  (%boosted-rounds booster-pointer) grad hess)))
                             (%update-one-iteration booster-pointer train-data-pointer))
                         (incf completed-rounds)
                         ;; Primary value only: `%read-evaluation''s RAW is `evaluation''s
                         ;; provenance, and a report carries no per-iteration raw text.
                         (let ((entries (when record-history
                                          (%read-evaluation booster-pointer
                                                            dataset-pointers))))
                           ;; Appended after every library entry, and before the push and
                           ;; the watcher, so the history and the watcher see one list. All
                           ;; three pointer variables are reassigned from what the call
                           ;; returns: the caller's own code has just run, and the next
                           ;; iteration's update and `%read-evaluation' both dereference
                           ;; them.
                           (when evaluation
                             (multiple-value-bind (custom pointers)
                                 (%custom-evaluation-entries
                                  backend evaluation booster-pointer dataset valid-sets
                                  dataset-pointers row-counts entries (= round 1) name-pin)
                               (setf entries (append entries custom)
                                     dataset-pointers pointers
                                     train-data-pointer (first pointers)
                                     valid-set-pointers (rest pointers))))
                           (when record-history
                             (push entries history))
                           (when (and watcher (observe-iteration watcher entries round))
                             (return))))
               (let* ((best-iteration (and watcher (watcher-best-iteration watcher)))
                      (report (training-report-from-history
                               (reverse history) completed-rounds dataset-names
                               :best-iteration best-iteration
                               :best-score (and watcher (watcher-best-score watcher))
                               :early-stopped-p (and watcher (watcher-stopped-p watcher)
                                                 (< completed-rounds num-rounds)))))
                 (when best-iteration
                   (%set-booster-best-iteration booster best-iteration))
                 (setf completed t)
                 (values booster report)))
          ;; Every exit that did not reach the `setf' above frees the booster. Where the old
          ;; body's `with-pointer-ownership' freed a RAW pointer, this frees the handle:
          ;; `free-booster' also marks it released and cancels its finalizer, so a signalling
          ;; run leaves nothing behind rather than an unreferenced handle whose finalizer only
          ;; warns. Named in full, not imported: `cl-gbdt/src/protocol''s `free-booster', the
          ;; generic, is a DIFFERENT symbol of the same name and is what this file imports.
          ;;
          ;; Wrapped in `handler-case', mirroring `with-pointer-ownership''s own free, so a
          ;; failing cleanup cannot replace the condition already unwinding (policy section
          ;; 10). That is not hypothetical here: on the branch that actually runs -- an open
          ;; backend and a live handle, since `create-booster' just built it -- this reaches
          ;; `%free-booster', which signals `foreign-call-error' on a non-zero
          ;; `XGBoosterFree' status. The closed-backend branch only `warn's and cannot
          ;; signal, and `wrong-backend-reference' cannot fire on a handle this method just
          ;; built, but the wrap covers all three rather than relying on which one applies.
          (unless completed
            (handler-case (cl-gbdt/src/xgboost/api:free-booster booster)
              (error () nil))))))))

(defmethod update-one-iteration ((booster xgboost-booster))
  "Advance BOOSTER by one boosting iteration via `XGBoosterUpdateOneIter'.

Unlike LightGBM's `LGBM_BoosterUpdateOneIter', which reads the booster's internal
training-set pointer implicitly, XGBoost's version takes the DMatrix handle explicitly,
so this reads it back from `booster-training-set' rather than being able to omit it. A
`load-model' booster's training set is NIL by design -- see the `booster' class'
documentation -- and handing `XGBoosterUpdateOneIter' a null DMatrixHandle would not
come back as a status code the way a bad parameter does: it is a null-pointer dereference
inside XGBoost's own implementation. That case is rejected here, before the foreign call,
for the same reason `%check-booster-datasets-live' exists for the pointers it does check.

XGBoost also reports no `produced_empty_tree'-style signal from this call, unlike
LightGBM -- there is nothing for this backend to report a false return for, so unlike
`cl-gbdt/src/lightgbm/protocol''s method of the same name, this always returns true after
a successful call; the generic function's \"returns false when no further split was
possible\" applies only insofar as a backend can report it, which this one cannot.

Signals `released-handle-error' when BOOSTER's training set, or any of its validation
sets, has already been freed -- see `%check-booster-datasets-live'. Signals
`missing-training-set' when BOOSTER has no training set at all -- a `load-model'
booster, which never went through `train' -- since handing `XGBoosterUpdateOneIter' a
null DMatrixHandle in that case is a null-pointer dereference, not something it can
reject with a status code.

This method's whole body was procedure -- there was no portable argument here to check or
translate -- so all of it is `cl-gbdt/src/xgboost/api''s `update-one-iteration', which is
where the two checks and the explicit training-set pointer now live."
  (with-foreign-float-traps-masked
    ;; Named in full, not imported: `cl-gbdt/src/protocol''s `update-one-iteration', the
    ;; generic this method is defined on, is a DIFFERENT symbol of the same name, and this
    ;; file imports that one. Not recursion. The `with-foreign-float-traps-masked' wrap stays
    ;; even though the callee establishes its own, for the reason `free-dataset' above gives.
    (cl-gbdt/src/xgboost/api:update-one-iteration booster)))

(defmethod free-booster ((booster xgboost-booster))
  "Free BOOSTER via `XGBoosterFree'. Does nothing if it was already freed.

See `free-dataset''s docstring for why this does not signal `backend-not-open' when
BOOSTER's backend has already been closed -- the same `with-booster' cleanup-form
reasoning applies here.

This method's whole body was procedure too, and is `cl-gbdt/src/xgboost/api''s
`free-booster' entirely, where the closed-backend branch now lives beside the identical one
`free-dataset' takes."
  (with-foreign-float-traps-masked
    ;; Named in full, not imported, and not recursion -- see `update-one-iteration' above,
    ;; which faces the same doubled name for the same reason.
    (cl-gbdt/src/xgboost/api:free-booster booster)))

;;; ---------------------------------------------------------------------------
;;; Inference

(defmethod predict ((booster xgboost-booster) matrix
                    &key (kind :normal) num-iteration missing)
  "Predict on MATRIX with BOOSTER -- a dense matrix via `XGBoosterPredictFromDMatrix', a
`csr-matrix' via `XGBoosterPredictFromCSR'.

KIND and NUM-ITERATION are as the `predict' generic function documents. NUM-ITERATION's
:BEST is resolved HERE, by `%resolve-best-num-iteration', and the integer it produces is what
reaches the procedure: `booster-best-iteration' is written by `train' and by nothing else, so
:BEST is a Layer 2 concept and `cl-gbdt/src/xgboost/api''s `predict' takes an integer or NIL,
refusing the keyword itself with `unsupported-argument' -- see its docstring, which says so
from the other side and measures what the keyword did before that refusal existed.
Predictions start from iteration 0 -- the protocol exposes no start-iteration override.

Signals `capability-unavailable' when MATRIX is a `csr-matrix' and this backend's
`:sparse-input' capability reads false -- see `cl-gbdt/src/xgboost/api''s
`%check-sparse-input', which checks it before any foreign call.

MISSING, the value in MATRIX that means *missing*, needs this backend's `:missing-value'
capability, which `%check-missing-value' re-checks below before any foreign call rather than
inheriting the check `make-dataset' made on the dataset BOOSTER was trained from -- policy
section 7 asks each operation to check for itself. That CAPABILITY GATE is the whole of what
this method does with the argument; the value itself is passed down unexamined and untouched.
It signals `unsupported-argument' for anything that is neither a `real' nor NIL, see
`missing-value-json', and NIL -- the default -- sends the IEEE NaN this backend sent
unconditionally before the argument existed, so a caller who passes nothing predicts exactly
what they predicted before. Which config JSON the sentinel then reaches depends on MATRIX's
own form and is the procedure's business, not this method's -- see
`cl-gbdt/src/xgboost/api''s `predict', which words that split from the other side.

Nothing here relates MISSING to the sentinel BOOSTER's training dataset was built with:
XGBoost does not record a DMatrix's sentinel on the booster, so the two are independent and
their disagreement is undetectable. See the `predict' generic function's docstring, where
that is stated as the caller's responsibility.

Returns the result array and, as a second value, the SHAPE this backend states for it -- a
list of integers in `array-dimensions' order. It is never NIL here, unlike
`cl-gbdt/src/lightgbm/protocol''s `predict': this library REPORTS a shape, through the
`out_shape'/`out_dim' pair both entry points write, and the procedure hands that report back
verbatim rather than deriving anything. `cl-gbdt/src/xgboost/api''s `predict' is where the
read-back lives and carries the measurements, the `out_dim' values per KIND among them. This
backend declares `:prediction-shape' in `*provided-capabilities*' to say the mechanism is
here; nothing re-checks that declaration, there being no argument to refuse.

The procedure itself is Layer 1 and lives in `cl-gbdt/src/xgboost/api''s `predict': the
choice of entry point, the transient DMatrix the dense path builds and frees, the sparse
path's restriction to `:normal' and `:raw', the shape read-back, the copy-out of
`out_result', together with the `:sparse-input' gate and the deliberate absence of any NaN or
infinity scan over the result. Everything the paragraphs above promise about those is that
function's doing; see its own docstring. What is left here is the portable contract: the
:MISSING capability gate, and resolving :BEST.

Refusing a KIND this backend has no prediction type for moved BELOW BOTH of those, and is
the one thing about this method a caller can observe changing: `%predict-type''s `ecase' now
runs inside the procedure rather than in the same `let' that read the pointer, so a call
wrong in two ways at once is answered by whichever check still runs first. Measured through
`cl-gbdt:predict' against the vendored library, on a booster trained without :EARLY-STOPPING
and so with no best iteration: a bad KIND together with `:num-iteration :best' signalled
`sb-kernel:case-failure' before the split and signals `unsupported-argument' now, putting a
typed `cl-gbdt' condition where an untyped one used to escape. A bad KIND alone is
`sb-kernel:case-failure' either way, and so is a bad KIND together with a non-NIL :MISSING --
the gate above never refuses while `:missing-value' reads true, which is the one row where
this backend differs from `cl-gbdt/src/lightgbm/protocol''s `predict', whose gate always
refuses and which therefore changed on that pair too. The old order could not be restored
without calling `%predict-type' here purely for effect, duplicating a check the procedure
already makes."
  (with-foreign-float-traps-masked
    ;; Read and discarded, and not redundant with the same call inside the procedure: this
    ;; method's contract is that a freed BOOSTER, or one whose backend has since been closed,
    ;; is refused BEFORE the two checks below -- so a caller who freed the booster and also
    ;; passed :MISSING, or :BEST to a booster that has no best iteration, still gets
    ;; `released-handle-error' rather than `capability-unavailable' or `unsupported-argument',
    ;; exactly as they did when this method held the whole procedure and read the pointer
    ;; first. `make-dataset' above keeps its own `%check-backend-open' for the same reason.
    (handle-live-pointer booster)
    ;; Resolved ABOVE the :MISSING gate, not in the delegation's argument list, because that is
    ;; where the old body resolved it: `%resolve-best-num-iteration' sat in the same `let' that
    ;; read the pointer, and the `when' came after. Unobservable on this backend as it stands,
    ;; `%check-missing-value' never refusing while `:missing-value' reads true -- but it is
    ;; observable on `cl-gbdt/src/lightgbm/protocol''s `predict', whose gate always refuses,
    ;; and the two methods are worth keeping in the same shape rather than letting one drift
    ;; on the strength of a capability answer that is a runtime value, not a constant.
    (let ((resolved (%resolve-best-num-iteration booster num-iteration
                                                 "predict's :num-iteration")))
      (when missing
        (%check-missing-value (handle-backend booster)))
      ;; Named in full, not imported, and not recursion -- see `update-one-iteration' above,
      ;; which faces the same doubled name for the same reason.
      (cl-gbdt/src/xgboost/api:predict booster matrix
                                       :kind kind
                                       :num-iteration resolved
                                       :missing missing))))

;;; ---------------------------------------------------------------------------
;;; Persistence

(defmethod save-model ((booster xgboost-booster) path &key num-iteration)
  "Save BOOSTER's model to PATH via `XGBoosterSaveModel'.

Signals `unsupported-argument' when NUM-ITERATION is supplied: unlike LightGBM's
`LGBM_BoosterSaveModel', `XGBoosterSaveModel' takes no iteration limit -- it always
saves every boosted round -- and silently ignoring the argument would be exactly the
failure mode `unsupported-argument' exists to prevent, per `%check-unsupported'. :BEST is
resolved by `%resolve-best-num-iteration' first, into an integer, which then meets this
same check exactly as an explicit integer would -- not special-cased around it. A caller
who wants a file that stops at the best iteration slices to it first with
`cl-gbdt/xgboost:slice-model' and saves the slice instead.

Returns PATH.

The procedure is Layer 1 and lives in `cl-gbdt/src/xgboost/api''s `save-model', which takes no
:NUM-ITERATION at all. What is left here is the refusal above, which exists because the
unified API promised a portable argument LightGBM honours and this library has no route for."
  (with-foreign-float-traps-masked
    (let ((resolved (%resolve-best-num-iteration booster num-iteration
                                                  "save-model's :num-iteration")))
      (%check-unsupported
       (handle-backend booster) "save-model's :num-iteration" resolved
       "XGBoosterSaveModel has no iteration limit; every boosted round is saved"))
    ;; Named in full, not imported, and not recursion -- see `update-one-iteration' above,
    ;; which faces the same doubled name for the same reason.
    (cl-gbdt/src/xgboost/api:save-model booster path)))

(defmethod load-model ((backend xgboost-backend) path)
  "Load an XGBoost model from PATH and return a new booster.

Unlike LightGBM's `LGBM_BoosterCreateFromModelfile', which allocates the booster and
loads the model in a single call, XGBoost splits the two: `XGBoosterCreate' first builds
a booster with no DMatrix handles at all -- see `%create-booster' -- and only then does
`XGBoosterLoadModel' populate it from PATH.

The returned booster has no training set -- see the `booster' class' documentation --
since PATH names a model, not a dataset.

The raw booster handle exists in C from the moment `XGBoosterCreate' returns, but
`make-handle' does not take ownership of it until `XGBoosterLoadModel' has also
succeeded. `with-pointer-ownership' spans exactly that gap: the pointer is owned by
nobody inside its body, and any exit that has not called TAKE-OWNERSHIP -- a failing
`XGBoosterLoadModel' the likeliest -- frees the raw booster here instead of orphaning it.

Signals `backend-not-open' before any of that when BACKEND is not open -- see
`%check-backend-open'.

This method's whole body was procedure -- there was no portable argument here to check or
translate -- so all of it is `cl-gbdt/src/xgboost/api''s `load-model', which is where the
two-call construction, the ownership window and the backend guard now live."
  (with-foreign-float-traps-masked
    (cl-gbdt/src/xgboost/api:load-model backend path)))

(defmethod model-to-string ((booster xgboost-booster) &key num-iteration)
  "Return BOOSTER's model as a JSON string via `XGBoosterSaveModelToBuffer'.

Signals `unsupported-argument' when NUM-ITERATION is supplied: `XGBoosterSaveModelToBuffer''s
config JSON has no iteration-limiting key, only `\"format\"' -- see `save-model' for the
same guard on the sibling entry point, and for why silently ignoring it is not an option.
:BEST is resolved by `%resolve-best-num-iteration' first, into an integer, which then
meets this same check exactly as an explicit integer would.

`out_dptr' is XGBoost's own memory, copied out via `foreign-string-to-lisp' with an
explicit `:count' from `out_len' rather than trusted to be null-terminated at the right
place.

The procedure is Layer 1 and lives in `cl-gbdt/src/xgboost/api''s `model-to-string', which
takes no :NUM-ITERATION at all. What is left here is the refusal above, which exists because
the unified API promised a portable argument LightGBM honours and this library has no route
for."
  (with-foreign-float-traps-masked
    (let ((resolved (%resolve-best-num-iteration booster num-iteration
                                                  "model-to-string's :num-iteration")))
      (%check-unsupported (handle-backend booster) "model-to-string's :num-iteration"
                           resolved "XGBoosterSaveModelToBuffer has no iteration limit"))
    (cl-gbdt/src/xgboost/api:model-to-string booster)))

;;; ---------------------------------------------------------------------------
;;; Feature importance

(defmethod feature-importance ((booster xgboost-booster) &key (kind :split) num-iteration)
  "Return BOOSTER's per-feature importances via `XGBoosterFeatureScore'.

Signals `unsupported-argument' when NUM-ITERATION is supplied: `XGBoosterFeatureScore''s
config JSON has no iteration-limiting key, only `importance_type', `feature_map' and
`feature_names' -- honoring it would require slicing the booster first, which this
backend does not do, so this refuses rather than silently scoring every round instead of
the requested subset.

The result has one entry per feature, indexed by column, matching
`cl-gbdt/src/lightgbm/api''s `feature-importance' -- zero for a feature never used
in a split. `XGBoosterFeatureScore' itself reports the opposite: `out_n_features' and
`out_scores' cover only features that appear in at least one split, so a feature never
split on is absent from its report, not present with a zero -- confirmed directly
against the vendored library and documented upstream. Left as `XGBoosterFeatureScore'
returns it, the result's length would be the number of *used* features, not the
dataset's column count, and its indices would not correspond to column positions --
sparse where LightGBM's equivalent is always dense. This builds a dense vector of
`%booster-num-features' entries instead, initialized to zero, and scatters each
reported score into the column `%feature-score-index' recovers from its feature name.

Signals `unsupported-argument' instead of returning a result at all when
`XGBoosterFeatureScore' reports more than one score per feature -- see
`%check-feature-score-dim'. In practice this is a linear (`gblinear') booster's `:split'
importance on a multi-class model: its scores are a per-class matrix, not one number per
feature, and there is no single value this backend can derive from that matrix without
inventing a reduction XGBoost itself does not define.

NUM-ITERATION does not accept :BEST, unlike `predict', `save-model' and
`model-to-string' -- `%reject-best-num-iteration' signals `unsupported-argument' for it
explicitly, ahead of the blanket rejection just below that would otherwise catch it only
incidentally, as any other non-NIL value.

The procedure is `cl-gbdt/src/xgboost/api''s `feature-importance', which takes no
:NUM-ITERATION at all. Both refusals stay here, in this order: :BEST explicitly, ahead of the
blanket refusal that would otherwise catch it only incidentally as any other non-NIL value."
  (with-foreign-float-traps-masked
    (%reject-best-num-iteration booster num-iteration "feature-importance's :num-iteration")
    (%check-unsupported (handle-backend booster) "feature-importance's :num-iteration"
                         num-iteration "XGBoosterFeatureScore has no iteration limit")
    (cl-gbdt/src/xgboost/api:feature-importance booster :kind kind)))

;;; ---------------------------------------------------------------------------
;;; Evaluation

(defmethod evaluation ((booster xgboost-booster))
  "Return BOOSTER's evaluation metrics via `%read-evaluation', the pointer-level reader this
backend shares between this method and `train''s per-iteration recording loop, so the two
can never disagree -- see the `evaluation' generic function's docstring for the portable
contract this satisfies.

`XGBoosterEvalOneIter' evaluates whatever DMatrices it is handed and consults nothing the
booster was built with, so the datasets this evaluates are BOOSTER's own retained
handles: its training set first, then each `train' :VALID-SETS entry in the order the
caller supplied them. That is what makes DATASET-INDEX mean the same thing here as it
does on LightGBM, which can only evaluate what training attached -- measured before this
method was written: for one booster, one set of handles and one iteration,
`XGBoosterEvalOneIter' called directly and this path through `%read-evaluation' produce
byte-identical result strings, and both agree with the logloss and error rate computed
independently from `predict' on the same data. A `load-model' booster retains no dataset
at all, which is the case an empty result comes from.

Each dataset is named to `XGBoosterEvalOneIter' by its own decimal index -- \"0\" for the
training set, \"1\" for the first validation set -- because the call requires one name per
DMatrix and builds each result label by joining that name to the metric's name with a
hyphen. `%split-eval-label' takes the label back apart against those same names, which is
the only way to recover the metric name: nothing in the result string alone marks where
one half ends and the other begins. Those names are an argument to a C call, never a
dataset name this API reports -- the caller sees the index, exactly as on LightGBM.

The values are `%parse-eval-result''s reading of `XGBoosterEvalOneIter''s formatted
output, which is what the secondary value's `:value-source :parsed-text' says, and its
`:raw' carries that output unmodified so nothing the library actually wrote is lost to
the parse. A field whose value the parser could not read as a `double-float' -- XGBoost
spells a non-finite one \"inf\" or \"nan\" -- keeps its entry with VALUE NIL rather than
disappearing from the result.

This method's whole body was procedure, so all of it -- the per-dataset pointer resolution,
the decimal naming, the parse and the provenance plist alike -- is
`cl-gbdt/src/xgboost/api''s `evaluation'. That function reads BOOSTER and every dataset it
evaluates through `handle-live-pointer' before calling `%read-evaluation', so a freed
booster or a freed retained dataset signals `released-handle-error' there; unlike
`cl-gbdt/src/lightgbm/api''s `evaluation', this backend needs no separate
`%check-booster-datasets-live', since every dataset it evaluates is one the delegate
resolves and checks explicitly, by its own handle, before any foreign call. The one thing
that changed with the move is that the booster's kind is now checked before its pointer is
read, so a value that is not a booster gets `wrong-backend-reference' rather than whatever
`handle-live-pointer' made of it. Unlike LightGBM's twin, the booster was already checked
before its retained datasets before the move, so a booster that is itself released and
also retains a released dataset has always signalled `released-handle-error' naming the
BOOSTER here, not the dataset."
  (with-foreign-float-traps-masked
    (cl-gbdt/src/xgboost/api:evaluation booster)))
