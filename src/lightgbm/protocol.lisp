;;;; protocol.lisp --- LightGBM backend, Layer 2: all thirteen methods of the unified
;;;; API's protocol, each delegating its C calls to `cl-gbdt/src/lightgbm/native', or its
;;;; whole procedure to `cl-gbdt/src/lightgbm/api', or both.
;;;;
;;;; The backend's CLOS classes and the `initialize-backend'/`shutdown-backend' pair that
;;;; opens and closes the shared library are Layer 1, not Layer 2, and live in
;;;; `cl-gbdt/src/lightgbm/classes' -- see that file's header for why.
;;;;
;;;; A method here owns the PORTABLE CONTRACT and nothing else: the checks and translations
;;;; that exist because a unified generic promised a portable argument. The procedure a
;;;; finished operation performs is Layer 1 and lives in `cl-gbdt/src/lightgbm/api' -- so far
;;;; `make-dataset', which checks :MISSING and :CATEGORICAL-FEATURES and then calls that
;;;; file's `create-dataset'; `predict', which checks :MISSING and resolves :BEST and then
;;;; calls that file's `predict'; and `free-dataset', `update-one-iteration' and
;;;; `free-booster', whose whole bodies were procedure and delegate entirely. A caller who
;;;; loaded `cl-gbdt/lightgbm' alone reaches those functions with no method here in the image
;;;; at all.
;;;;
;;;; `train' is the one exception, and deliberately so: it builds its booster itself rather
;;;; than calling `cl-gbdt/src/lightgbm/api''s `create-booster'. See the comment at its
;;;; creation call, which measures why.

(uiop:define-package #:cl-gbdt/src/lightgbm/protocol
  (:use #:cl)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt/src/lightgbm/native
                #:%check-backend-open
                #:%check-lightgbm-dataset
                #:%parameter-string
                #:%dataset-num-rows
                #:%create-booster
                #:%add-valid-data
                #:%update-one-iteration
                #:%booster-predictions
                #:%update-one-iteration-custom
                #:%free-booster-unchecked
                #:%read-evaluation)
  (:import-from #:cl-gbdt/src/lightgbm/classes
                #:lightgbm-backend
                #:lightgbm-dataset
                #:lightgbm-booster)
  ;; Layer 1's finished operations. `free-dataset', `update-one-iteration', `free-booster' and
  ;; `predict' are deliberately absent from this clause: the `:import-from
  ;; #:cl-gbdt/src/protocol' below names a GENERIC FUNCTION of each of those names, and each
  ;; pair is two different symbols -- importing both would be a name conflict, not a re-import.
  ;; The four methods that need the Layer 1 functions name them in full. `create-booster' is
  ;; absent for an unrelated reason: no method here calls it, `train' building its own booster
  ;; for the reason its creation call records, and an import naming a symbol nothing uses is
  ;; one more claim to keep true.
  (:import-from #:cl-gbdt/src/lightgbm/api
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
                #:with-pointer-ownership
                #:handle-live-pointer
                #:handle-backend
                #:%resolve-best-num-iteration)
  (:import-from #:cl-gbdt/src/conditions
                #:unsupported-argument
                #:capability-unavailable)
  (:import-from #:cl-gbdt/src/config/categorical-features
                #:categorical-feature-string)
  (:import-from #:cl-gbdt/src/config/objective
                #:check-objective-result
                #:objective-parameters)
  (:import-from #:cl-gbdt/src/parameters
                #:normalize-parameters)
  (:import-from #:cl-gbdt/src/training/history
                #:training-report-from-history)
  (:import-from #:cl-gbdt/src/training/early-stopping
                #:train-early-stopping-watcher
                #:observe-iteration
                #:watcher-best-iteration
                #:watcher-best-score
                #:watcher-stopped-p)
  (:import-from #:cl-gbdt/src/training/custom-metric
                #:custom-metric-entry
                #:check-metric-name-collision
                #:make-metric-name-pin
                #:pin-metric-name)
  (:import-from #:cl-gbdt/src/foreign
                #:with-foreign-float-traps-masked)
  (:export #:lightgbm-backend))

(in-package #:cl-gbdt/src/lightgbm/protocol)

;;; ---------------------------------------------------------------------------
;;; Floating-point trap safety
;;;
;;; Every method below that reaches into lib_lightgbm.so -- all thirteen protocol
;;; methods -- wraps its entire body in `with-foreign-float-traps-masked'.
;;; `initialize-backend' and `shutdown-backend', which load and unload the library
;;; itself, are wrapped exactly the same way in `cl-gbdt/src/lightgbm/classes', and every
;;; Layer 1 operation in `cl-gbdt/src/lightgbm/api' the same way again, where they live;
;;; this rule is the backend's, not this file's. A method that now delegates its procedure
;;; to `api' keeps its own wrap regardless: the two nest harmlessly, and dropping it would
;;; make the wrap depend on what the callee happens to do today.
;;; See that macro's docstring in `cl-gbdt/src/foreign' for why, and
;;; `cl-gbdt/src/xgboost/protocol''s identical commentary for the concrete case
;;; (XGBoost's `multi:softprob' softmax) that surfaced this: LightGBM has not tripped
;;; it yet, but its C API is exactly as unprotected against SBCL's x86-64 trap
;;; defaults, so it gets the same treatment rather than waiting for its own CI
;;; failure. Method-body granularity, not per-call, so a call added later inside an
;;; already-wrapped method cannot reopen this gap by omission. Every actual C call a
;;; method below makes goes through `cl-gbdt/src/lightgbm/native', but the mask is
;;; established here, around the whole method body, not inside that file -- see its
;;; own header.

;;; ---------------------------------------------------------------------------
;;; The `:missing-value' gate

(defun %check-missing-value (backend)
  "Signal `capability-unavailable' when BACKEND's `:missing-value' capability reads false.

Which, on this backend, is always: `LightGBM/include/LightGBM/c_api.h' contains no `missing'
anywhere, so there is no argument, key or field through which a caller could name the value
that means missing, and nothing for this backend to declare the capability from. LightGBM's
`use_missing' and `zero_as_missing' are not that: they are policy flags in the parameter
string -- whether missing values are handled at all, and whether a zero counts as one -- and
they are already reachable through `make-dataset''s :PARAMETERS, which is policy section 6's
documented escape hatch for exactly a backend's own vocabulary.

So this signals regardless of the VALUE, a NaN included -- even though a NaN in the matrix is
in fact what LightGBM's own ingestion path already treats as missing, so `:missing' naming
one would be a request the library happens to honour. A capability whose answer depended on
which value was passed could not be stated by `backend-supports-p' at all: it answers about
the backend, not about an argument it has not seen. Accepting the one value that would work
and refusing the rest would make `:missing' mean something different here than on a backend
that takes any of them.

Policy section 7 requires the operation itself to re-check the capability rather than trusting
the caller to have asked `backend-supports-p' first -- the same rule
`cl-gbdt/src/lightgbm/api''s `%check-sparse-input' follows for `:sparse-input'. Mirrors
`cl-gbdt/src/xgboost/protocol''s function of the same name, which is the same shape against
an answer that is true.

Only a non-NIL :MISSING ever reaches this. NIL means the backend's own default -- what every
caller has always got -- so a caller who passes nothing needs no capability at all, and every
existing call keeps working unchanged."
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
above and `cl-gbdt/src/lightgbm/api''s `%check-sparse-input' follow for their own. This
backend answers true unconditionally, `*provided-capabilities*' naming it because the column
list is a key in the parameter string and so has no C symbol to probe. That does not make the
check redundant: it is what keeps the two backends' code saying the same thing, so
`make-dataset' here and `make-dataset' in `cl-gbdt/src/xgboost/protocol' gate the argument
identically and neither has to be read to know what the other does.

Written against `backend-supports-p' rather than against this backend's name, like both of
those, so a caller who overwrites the capability plist is refused here rather than being
handed a dataset built from a capability the backend no longer claims.

Only a non-NIL :CATEGORICAL-FEATURES ever reaches this. NIL means what every caller has always
got -- every column read as a quantity, and nothing added to the parameter string -- so a
caller who passes nothing needs no capability and every existing call keeps working
unchanged."
  (unless (backend-supports-p backend :categorical-features)
    (error 'capability-unavailable
           :backend (backend-name backend) :capability :categorical-features)))

(defparameter *categorical-feature-parameter-names*
  '("categorical_feature" "cat_feature" "categorical_column" "cat_column"
    "categorical_features")
  "Every spelling LightGBM honours for the parameter-string key `make-dataset' now writes
itself, as `normalize-parameters' renders a key: lower case, underscores for dashes.

MEASURED against the vendored LightGBM 4.7.0, not read off a header -- the library's
parameter aliases appear in none of the vendored ones. Each of the five above changes what
is learned from a column of category ordinals; three near-misses tried alongside them --
`kategorical_feature', `categorical_feat' and `cat_features' -- leave the trained numbers
identical to a run that named no key at all. `cat_features' is the one that matters: it is
the PLURAL of `cat_feature', which IS honoured, so this list can only be enumerated and
never prefix-matched or fuzzily matched. Refusing `cat_features' would reject an ordinary
backend parameter that must keep reaching the library.
`tests/functional/categorical-features.lisp''s `the-parameters-key-is-refused' restates the
list and re-measures the distinction against the real library.")

(defun %check-categorical-parameter-keys (backend parameters)
  "Signal `unsupported-argument' against BACKEND when PARAMETERS names a key from
`*categorical-feature-parameter-names*'.

Called ONLY when `make-dataset' was given a non-NIL :CATEGORICAL-FEATURES, which is the only
state in which the clash below can arise -- see that method, where the call sits inside the
same `when' as `%check-categorical-features'. A caller who names no categorical column and
writes `categorical_feature' in :PARAMETERS by hand is using policy section 6's escape hatch
for a backend's own vocabulary, gets it honoured exactly as they always did, and never
reaches this function. Nothing about that path changed when :CATEGORICAL-FEATURES was added.

What this refuses is the two together. `make-dataset' writes `categorical_feature' into the
same parameter string PARAMETERS becomes, from :CATEGORICAL-FEATURES, so a caller who supplies
both is stating the same thing twice in one string. Which copy the library would then read is
measured, and it is the worse answer of the two: LightGBM takes the FIRST occurrence of a
duplicated key -- `categorical_feature=0 categorical_feature=1' trains exactly what
`categorical_feature=0' alone trains -- while this method appends its own entry LAST. So it is
the wrapper's copy, rendered from the argument the caller explicitly named, that would
silently lose. An argument accepted and then quietly discarded is exactly the failure mode
`cl-gbdt/src/xgboost/native''s `%check-unsupported' exists to prevent on the other backend.

Refused rather than merged with :CATEGORICAL-FEATURES, or honoured in its place: the two say
the same thing in two vocabularies, and a wrapper that reconciled them would have to say what
each of the five aliases means alongside each of the others as well.

Keys are compared against `normalize-parameters''s own output, so `:categorical-feature' and
`\"categorical_feature\"' are one key here, as they already are to the library. PARAMETERS is
normalized once here and again by `%parameter-string'; the only consequence is that an
odd-length plist's `data-error' comes from this call rather than that one, both of them
before any foreign call.

Gated on the ARGUMENT, not on the `:categorical-features' capability: this backend provides
that capability unconditionally, so gating on it would refuse the key in every state the
argument could be refused in anyway, and would take the escape hatch away from callers who
never asked for the feature."
  (let ((named (find-if (lambda (pair)
                          (member (car pair) *categorical-feature-parameter-names*
                                  :test #'string=))
                        (normalize-parameters parameters))))
    (when named
      (error 'unsupported-argument
             :backend (backend-name backend)
             :argument "make-dataset's :parameters"
             :reason (format nil "~A is one of the ~R spellings LightGBM honours for the ~
                                  categorical column list (~{~A~^, ~}), and make-dataset ~
                                  writes that key itself from its :categorical-features ~
                                  argument -- name the columns there instead"
                             (car named)
                             (length *categorical-feature-parameter-names*)
                             *categorical-feature-parameter-names*)))))

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
`%check-categorical-features' above, and `cl-gbdt/src/lightgbm/api''s `%check-sparse-input',
follow for their own. Mirrors `cl-gbdt/src/xgboost/protocol''s function of the same name.

Unlike `%check-categorical-features', whose answer this backend declares unconditionally in
`*provided-capabilities*', this one's is PROBED: the four C functions `train''s custom loop
needs are named in `*optional-symbols*' rather than `*required-symbols*', so a LightGBM
missing any of them opens normally and reads false here. See that variable's own docstring
for why the entry belongs there and not in `*provided-capabilities*'. `%check-missing-value'
is a third case again, and neither of those two: `:missing-value' appears in NEITHER list on
this backend, so its false answer is the ABSENCE of a declaration rather than a declaration --
see `backend-supports-p', which reads a capability missing from the plist as unavailable.

The type check is here, beside the capability check, rather than left to the `funcall' in
`train''s loop. By then the booster handle exists and one iteration's scores have already
been read out of the library, so `:objective 42' would surface as SBCL's own untyped
`type-error' from mid-loop -- naming neither the argument nor the backend -- where every
other malformed argument on this backend signals `unsupported-argument' before any foreign
call. `functionp' rather than a `function' type declaration: a symbol naming a function is
NOT accepted, since `funcall' would resolve it against whatever global definition happened to
be in force at each iteration rather than against what the caller passed."
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
above, and `cl-gbdt/src/lightgbm/api''s `%check-sparse-input', follow for their own. Mirrors
`cl-gbdt/src/xgboost/protocol''s function of the same name, which reads the same capability
out of that backend's `*provided-capabilities*' rather than out of a probe.

Like `:custom-objective''s, this backend's answer is PROBED rather than declared: the three
C functions `%booster-predictions' makes are named in `*optional-symbols*' rather than
`*required-symbols*', so a LightGBM missing any of them opens normally and reads false here.
See that variable's own docstring for why the entry belongs there, and for why all three are
also named by `:custom-objective' -- whose loop reads scores through the same
`%booster-predictions' -- without the two entries conflicting.

RECORD-HISTORY NIL is refused rather than silently ignored, for exactly the reason
`train-early-stopping-watcher' refuses the same pair: a custom metric's whole result is the
per-iteration series RECORD-HISTORY NIL exists not to build, and the values would be computed
at full cost and then dropped. That check lives here rather than in
`cl-gbdt/src/training/custom-metric' because it is a rule about `train''s ARGUMENT LIST, the
same way the early-stopping one is, and because it must fire before any foreign call.

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

(defmethod make-dataset ((backend lightgbm-backend) matrix
                          &key label weight group feature-names parameters reference missing
                            categorical-features)
  "Build a LightGBM dataset from MATRIX -- a dense matrix via `LGBM_DatasetCreateFromMat',
a `csr-matrix' via `LGBM_DatasetCreateFromCSR' -- attaching LABEL, WEIGHT and GROUP with
`LGBM_DatasetSetField' and FEATURE-NAMES with `LGBM_DatasetSetFeatureNames' when supplied.
See the `make-dataset' generic function's docstring for what each argument means,
including REFERENCE, and for what a `csr-matrix' changes about none of them.

Signals `capability-unavailable' when MATRIX is a `csr-matrix' and this backend's
`:sparse-input' capability reads false -- see `cl-gbdt/src/lightgbm/api''s
`%dataset-pointer', which checks it. Every other argument behaves identically either way:
PARAMETERS and REFERENCE reach the sparse entry point as the same two C parameters they
reach the dense one as, and LABEL, WEIGHT, GROUP and FEATURE-NAMES are attached to the
finished dataset by `create-dataset', which never sees which entry point built it.

Signals `capability-unavailable' naming `:missing-value' for a non-NIL MISSING, whatever the
value is and whatever form MATRIX takes -- this backend has no C-API route for a missing-value
sentinel at all. See `%check-missing-value' above, which carries the reasoning, including why
a NaN LightGBM would in fact honour is refused with the rest. MISSING NIL, the default, is
this backend's own default and reaches no check: every call that does not name a sentinel
behaves exactly as it did before the argument existed.

CATEGORICAL-FEATURES names which columns of MATRIX hold categories rather than quantities,
and becomes a `categorical_feature' entry in the parameter string -- this backend's own name
for the list, rendered by `categorical-feature-string' from the caller's own MATRIX. It is
appended after PARAMETERS' own entries, and reaches whichever creation entry point MATRIX's
form selects without a branch of its own: the string is built once, and neither
`LGBM_DatasetCreateFromMat' nor `LGBM_DatasetCreateFromCSR' can tell which of the two it was
built for.

CATEGORICAL-FEATURES NIL, the default, adds nothing to that string -- not an empty entry --
so a call that names no categorical column builds exactly the dataset it built before this
argument existed. A non-NIL value signals `capability-unavailable' naming
`:categorical-features' when the capability reads false (see `%check-categorical-features'
above, which reads the capability rather than this backend's name) and `unsupported-argument'
naming `:categorical-features' for an index that is not an integer, is negative, is beyond
MATRIX's last column, or was named twice.

Signals `unsupported-argument' naming \"make-dataset's :parameters\" when CATEGORICAL-FEATURES
is supplied AND PARAMETERS carries any of the five spellings LightGBM honours for that key --
`categorical_feature', `cat_feature', `categorical_column', `cat_column' or
`categorical_features'. Only the two together: the entry this method appends would land after
the caller's and LightGBM keeps the FIRST occurrence of a duplicated key, so the argument the
caller explicitly named would be the one silently discarded.

PARAMETERS alone is untouched by any of this. A caller who names no categorical column and
writes `categorical_feature' there by hand is on policy section 6's escape hatch for a
backend's own vocabulary, and gets it honoured exactly as they did before this argument
existed. `cat_features' is never refused either way, the library not honouring it as an alias.
See `%check-categorical-parameter-keys' above for the measurements behind all of that.

Signals `foreign-call-error' when dataset creation reports success but writes a
null handle -- a library-contract violation, but one every later call through
this handle would otherwise dereference blindly. Signals `wrong-backend-reference'
when REFERENCE is supplied but is not a `lightgbm-dataset', `released-handle-error'
when it has already been freed, and `backend-not-open' when its backend has since
been closed -- see `%reference-pointer'.

Signals `backend-not-open' before any of that when BACKEND is not open -- see
`%check-backend-open'.

The procedure itself is Layer 1 and lives in `cl-gbdt/src/lightgbm/api''s `create-dataset':
building the pointer, attaching LABEL, WEIGHT, GROUP and FEATURE-NAMES in that order, and the
ownership dance that frees the raw dataset when one of those signals. What is left here is
the portable contract -- the three checks above, and rendering CATEGORICAL-FEATURES into the
one parameter-string key this backend states it as. Everything the paragraphs above promise
about a null handle, about REFERENCE and about the raw handle's ownership is that function's
doing; see its own docstring."
  (with-foreign-float-traps-masked
    ;; Checked here as well as inside `create-dataset', which cannot omit it either: this
    ;; method's contract is that a closed BACKEND is refused BEFORE the argument checks
    ;; below, so a caller who closed the backend and also passed a bad :CATEGORICAL-FEATURES
    ;; index still gets `backend-not-open' rather than `unsupported-argument'.
    (%check-backend-open backend)
    (when missing
      (%check-missing-value backend))
    ;; Both checks hang off the same non-NIL CATEGORICAL-FEATURES, and the second one for a
    ;; reason of its own: the clash it refuses can only arise once this method is writing an
    ;; entry of its own. A caller who names no categorical column is using :PARAMETERS as
    ;; policy section 6's escape hatch and keeps working exactly as before.
    (when categorical-features
      (%check-categorical-features backend)
      (%check-categorical-parameter-keys backend parameters))
    ;; Rendered before anything is allocated, because on this backend the column list is a
    ;; key in the very parameter string that CREATES the dataset -- so a bad index signals
    ;; with nothing yet to free, and the string is built once for whichever creation entry
    ;; point MATRIX's own form selects. The renderer takes the caller's MATRIX, dense or
    ;; `csr-matrix', which is what makes both backends range-check the same column count.
    ;; `create-dataset' below is handed the finished plist: the key is this backend's own
    ;; vocabulary by then, and Layer 1 neither knows nor translates it.
    (let* ((categorical-string (categorical-feature-string categorical-features matrix
                                                           (backend-name backend)))
           (parameters (if categorical-string
                           (append parameters (list :categorical-feature categorical-string))
                           parameters)))
      (create-dataset backend matrix :label label :weight weight :group group
                                     :feature-names feature-names
                                     :parameters parameters :reference reference))))

(defmethod dataset-num-rows ((dataset lightgbm-dataset))
  "Return DATASET's row count, read via `LGBM_DatasetGetNumData'.

This method's whole body was procedure -- there was no portable argument here to check or
translate -- so all of it is `cl-gbdt/src/lightgbm/api''s `dataset-num-rows', which is where
the class guard the specializer above used to provide now lives too."
  (with-foreign-float-traps-masked
    ;; Named in full, not imported: `cl-gbdt/src/protocol''s `dataset-num-rows', the generic
    ;; this method is defined on, is a DIFFERENT symbol of the same name, and this file
    ;; imports that one.
    (cl-gbdt/src/lightgbm/api:dataset-num-rows dataset)))

(defmethod dataset-num-features ((dataset lightgbm-dataset))
  "Return DATASET's feature count, read via `LGBM_DatasetGetNumFeature'. Delegates wholly, as
`dataset-num-rows' above does and for the same reason."
  (with-foreign-float-traps-masked
    (cl-gbdt/src/lightgbm/api:dataset-num-features dataset)))

(defmethod free-dataset ((dataset lightgbm-dataset))
  "Free DATASET via `LGBM_DatasetFree'. Does nothing if it was already freed.

Unlike every other operation in this file -- `free-booster' below excepted, which
takes this same path for this same reason -- this does not go through
`handle-live-pointer' and so does not signal `backend-not-open' when DATASET's
backend has already been closed. `free-dataset' runs from `with-dataset''s
`unwind-protect' cleanup form, and a non-local exit is exactly when that cleanup
runs; signalling there would replace whatever condition is already unwinding the
stack instead of letting it propagate. It `warn's instead, the foreign memory being
genuinely unreclaimable by then.

This method's whole body was procedure -- there was no portable argument here to check or
translate -- so all of it is `cl-gbdt/src/lightgbm/api''s `free-dataset', which is where the
closed-backend branch and the measurements behind it now live."
  (with-foreign-float-traps-masked
    ;; Named in full, not imported: `cl-gbdt/src/protocol''s `free-dataset', the generic
    ;; this method is defined on, is a DIFFERENT symbol of the same name, and this file
    ;; imports that one. The `with-foreign-float-traps-masked' wrap stays even though the
    ;; callee establishes its own -- the masks nest harmlessly, and every `defmethod' in
    ;; this file carries one by the rule the header states, checked by
    ;; `tools/ci/check-float-traps.lisp'.
    (cl-gbdt/src/lightgbm/api:free-dataset dataset)))

;;; ---------------------------------------------------------------------------
;;; Training

(defun %valid-set-name (backend entry)
  "Return the name half of ENTRY, one element of `train''s :VALID-SETS: NIL when ENTRY is
a bare dataset, or ENTRY's car when ENTRY is a (NAME . DATASET) cons and NAME is a string.

Signals `unsupported-argument' naming :VALID-SETS and ENTRY itself when ENTRY is a cons
whose car is not a string -- before any foreign call, and before the dataset half of
ENTRY is checked against `lightgbm-dataset' by `%check-lightgbm-dataset', which runs
afterward on `%valid-set-dataset''s result. That later check is what turns a cons whose
cdr is not this backend's own kind of dataset into `wrong-backend-reference' instead --
a different mistake from this one, and so a different condition."
  (if (consp entry)
      (let ((name (car entry)))
        (unless (stringp name)
          (error 'unsupported-argument
                 :backend (backend-name backend)
                 :argument "train's :valid-sets"
                 :reason (format nil "each element must be a dataset or a ~
                                      (string . dataset) cons; ~S's car is not a string"
                                 entry)))
        name)
      nil))

(defun %valid-set-dataset (entry)
  "Return the dataset half of ENTRY, one element of `train''s :VALID-SETS: ENTRY itself
when it is a bare dataset, or its cdr when it is a (NAME . DATASET) cons.

Does not check that the result is a `lightgbm-dataset' -- `%check-lightgbm-dataset' does
that afterward, on every element `%valid-set-name' has already let through."
  (if (consp entry) (cdr entry) entry))

(defun %recheck-train-datasets (backend dataset valid-sets)
  "Re-run `train''s own opening checks over BACKEND, DATASET and VALID-SETS, and return
DATASET's freshly read live pointer.

`train''s loop calls this after every `funcall' of a caller-supplied OBJECTIVE and after every
`funcall' of a caller-supplied EVALUATION, which are the only points in the loop where code this
library did not write runs. That code may free the training set -- `free-dataset' from inside
the objective is the case this was found through -- and the pointer `train' read once before the
loop is then a pointer into freed memory that `LGBM_BoosterUpdateOneIterCustom' would
dereference. Measured before this function existed: a memory fault at an arbitrary address,
killing the process rather than signalling. The checks are exactly the ones `train' already ran,
so the caller gets `released-handle-error', `backend-not-open' or `wrong-backend-reference' --
the typed conditions every other freed handle in this library produces -- and never a fault. The
RETURN VALUE is the point of the exercise: re-checking and then going on to use the pointer read
before the loop would fix nothing, so `train' assigns this to the variable it reads from.

The VALID-SETS entries are re-checked for the same reason even though nothing here uses their
pointers: `LGBM_BoosterGetEval', which the same iteration reaches through `%read-evaluation'
when RECORD-HISTORY is true, evaluates each attached validation set through memory that
dataset owns, and `LGBM_DatasetFree' clears nothing in the booster when one is freed -- see
`%check-booster-datasets-live', which exists for that exact hazard on the public
`update-one-iteration' path. `LGBM_BoosterGetNumPredict' and `LGBM_BoosterGetPredict', which
`%custom-evaluation-entries' reaches through `%booster-predictions' for each dataset index in
turn, read the same per-dataset memory and are covered by exactly the same re-check -- which
is why that function calls this one between two consecutive datasets' reads rather than only
once per iteration. That function also calls this one ONCE MORE, before its loop, purely to
have a live pointer to return should the loop run zero times; that call re-checks nothing the
iteration's other calls do not already re-check, and see its own comment for why it is there.

BACKEND itself is re-checked with `%check-backend-open' because `close-backend' unmaps the
shared library and the objective can call it. `handle-live-pointer' already refuses a handle
whose OWN backend has been closed, which covers the ordinary case where DATASET was built by
BACKEND; the check here is what covers the case `%check-lightgbm-dataset' documents as
legitimate and therefore does not catch -- a dataset built by a second `lightgbm-backend'
instance over the same library, whose own backend is still open while BACKEND is not. It
costs one slot read per iteration."
  (%check-backend-open backend)
  (let ((train-data-pointer
          (%check-lightgbm-dataset backend dataset "train's dataset argument"
                                    'lightgbm-dataset)))
    (dolist (valid-set valid-sets)
      (%check-lightgbm-dataset backend valid-set "a train :valid-sets entry" 'lightgbm-dataset))
    train-data-pointer))

(defun %custom-evaluation-entries (backend evaluation booster-pointer dataset valid-sets
                                    row-counts library-entries check-collisions-p name-pin)
  "Call EVALUATION once for each of BOOSTER-POINTER's datasets and return two values: the
(DATASET-INDEX METRIC-NAME VALUE) entries the calls produced, in dataset-index order, and
DATASET's freshly re-read live pointer.

ROW-COUNTS is the row count of each dataset in index order -- `(cons TRAINING-ROWS
VALID-SET-ROWS)' -- and is what says how many datasets there are as well as how wide each
one's prediction buffer is. `train' reads it once before its loop rather than per iteration:
a built LightGBM dataset's row count cannot change, there is no C entry point that appends
rows to one, and an integer read before the loop cannot go stale into a fault the way a
pointer can.

Each dataset's predictions come from `%booster-predictions', which is handed that dataset's
OWN row count -- the 40-row training set's and the 17-row validation set's are different
numbers and `LGBM_BoosterGetNumPredict' reports each separately, so deriving one from the
other would silently mis-shape the array the caller's function is handed.

EVALUATION is called with that array and the dataset's index, and must return two values, a
metric name and a value; `custom-metric-entry' checks both and builds the entry. The SECOND
value returned here is the point of the re-check that follows every call: EVALUATION is
caller code and may free a dataset or close BACKEND, so `%recheck-train-datasets' runs the
moment it returns -- before the NEXT dataset's `LGBM_BoosterGetPredict', and before `train'
uses the training pointer again -- and `train' assigns the pointer it returns rather than
going on with the one it read before the loop. See that function for what each of its three
re-checks covers.

CHECK-COLLISIONS-P runs `check-metric-name-collision' against LIBRARY-ENTRIES, this
iteration's own `%read-evaluation' result. `train' passes true on the first iteration only:
what the LIBRARY reports cannot be known before a booster has produced one real evaluation,
and after that first one the library's own name set does not change.

THAT SECOND HALF IS THE HINGE, and it is worth meeting here rather than deducing later,
because it is what makes ONE comparison cover a whole run. On this backend it is a property
of where the names come from and not of anything this code does: `%read-evaluation' pairs
each dataset's values with the ONE name list `%booster-eval-names' reads out of
`LGBM_BoosterGetEvalNames', which `LGBM_BoosterCreate' fixes from the `metric' parameter and
which takes no per-iteration input at all. NOTHING ASSERTS IT -- it is a fact about the
vendored library, not about this library, so a green suite says nothing about it. An editor
moving this check off round 1, or reaching a LightGBM whose reported names could vary within
a run, is changing what makes round 1 sufficient.

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
caller's object -- see `custom-metric-entry', which measured what that reached."
  (let ((entries '())
        ;; Initialised from a re-check rather than from NIL, so this function returns a live
        ;; pointer even when ROW-COUNTS is empty and the loop below never runs. `train' always
        ;; passes at least the training set's count today, but it assigns TRAIN-DATA-POINTER
        ;; from this value unconditionally, and a NIL reaching `%dataset-num-rows' next
        ;; iteration is a worse failure than this costs: one backend check plus one handle
        ;; check per dataset -- the training set and every VALID-SETS entry -- and not one
        ;; foreign call among them.
        (train-data-pointer (%recheck-train-datasets backend dataset valid-sets)))
    (loop :for index :from 0
          :for rows :in row-counts
          :do (let ((predictions (%booster-predictions booster-pointer index rows)))
                (multiple-value-bind (name value) (funcall evaluation predictions index)
                  (setf train-data-pointer
                        (%recheck-train-datasets backend dataset valid-sets))
                  (let* ((entry (custom-metric-entry (backend-name backend) name value index))
                         (pinned-name (second entry)))
                    (when check-collisions-p
                      (check-metric-name-collision (backend-name backend) pinned-name index
                                                    library-entries))
                    (pin-metric-name (backend-name backend) name-pin pinned-name index)
                    (push entry entries)))))
    (values (nreverse entries) train-data-pointer)))

(defmethod train ((backend lightgbm-backend) dataset
                   &key valid-sets (num-rounds 100) parameters (record-history t)
                        early-stopping objective evaluation)
  "Train a LightGBM booster on DATASET for up to NUM-ROUNDS boosting iterations, and
return it and a `training-report' of the run.

Builds the booster with `LGBM_BoosterCreate' from PARAMETERS, attaches each of
VALID-SETS with `LGBM_BoosterAddValidData', then drives
`LGBM_BoosterUpdateOneIter' NUM-ROUNDS times -- or fewer, when EARLY-STOPPING
ends the run first. See the `train' generic function's docstring for what each
argument means, and for what the secondary value holds; NUM-ROUNDS defaults to
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
iteration through `%read-evaluation': the same function the `evaluation' method calls, on
the same booster pointer and the same dataset count, which is what keeps the history and
what `evaluation' answers afterward from being able to disagree. `training-report-from-history'
folds the run's worth of them into the report once the loop is done; that fold is backend-
neutral and shared with `cl-gbdt/src/xgboost/protocol''s `train', so what the two backends
record cannot drift apart either. It orders series by the (DATASET-INDEX, METRIC-NAME)
pair's first appearance, which for this backend is `%read-evaluation''s own dataset-major
order, so the report's series arrive in exactly the order `evaluation' reports its entries
in without anything being sorted.

RECORD-HISTORY NIL skips that read entirely -- one `LGBM_BoosterGetEval' call per dataset
per iteration, which is what makes recording cost real wall-clock time (see the `train'
generic's docstring for the measured figures). The loop is then exactly the
`LGBM_BoosterUpdateOneIter' loop this method ran before it recorded anything, and the
report it still returns as its secondary value has an empty series list over the same
NUM-ROUNDS -- `training-report-from-history' over an empty history, the same shape a run
with `metric=none' produces.

A read that fails propagates, freeing the booster through the `with-pointer-ownership'
form below rather
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

OBJECTIVE replaces `LGBM_BoosterUpdateOneIter' with `LGBM_BoosterUpdateOneIterCustom' for
every iteration of the loop, driven by the gradient and Hessian the caller's own function
returns -- see the `train' generic function's docstring for what that function is called with
and what it must return. Signals `capability-unavailable' naming `:custom-objective' for a
non-NIL OBJECTIVE when the capability reads false, before any foreign call: see
`%check-custom-objective' above, which reads the capability rather than this backend's name,
and `*optional-symbols*' for why the answer here is probed rather than declared. OBJECTIVE
NIL, the default, reaches no check and runs exactly the `LGBM_BoosterUpdateOneIter' loop this
method has always run.

A non-NIL OBJECTIVE also OVERRIDES any `objective' entry in PARAMETERS, forcing it to
\"none\" through `objective-parameters' before `LGBM_BoosterCreate' ever sees the string.
That is not a convenience: `LGBM_BoosterUpdateOneIterCustom' refuses to run while the booster
holds an objective function at all -- `Check failed: objective_function_ == nullptr', a
non-zero return this method would surface as `foreign-call-error' -- so the combination the
override replaces has no working form to preserve. Every other parameter passes through
untouched and in its original order, `num_class' included, which is still what tells LightGBM
how many output groups a multiclass custom objective has.

Each iteration reads the booster's current raw scores with `%booster-predictions' --
`LGBM_BoosterGetPredict' at `data_idx' 0, the scores LightGBM already holds for the training
data, rather than a fresh `predict' over the training matrix, which this method does not have
and which would cost a full prediction pass per iteration. It hands them to OBJECTIVE as a
(ROWS GROUPS) `double-float' array, where ROWS comes from `%dataset-num-rows' on the training
set's own pointer and GROUPS from `LGBM_BoosterGetNumClasses'. What comes back is checked by
`check-objective-result' -- `cl-gbdt/src/config/objective''s, which is backend-neutral pure
code and names no library, so a second backend refuses the same shapes with the same
`dimension-mismatch' by calling it rather than by restating it -- and only then flattened into
the C buffers, GROUP-MAJOR on this backend (row I of group K at `(+ (* K ROWS) I)') and
converted to `single-float', which is what `LGBM_BoosterUpdateOneIterCustom''s `const float*'
parameters admit. Both the flattening and the score layout are measured; see
`%update-one-iteration-custom' and `%booster-predictions' for the measurements. The flattening is
this method's business and not the caller's: OBJECTIVE is handed, and returns, a (ROWS GROUPS)
array whichever order the library underneath wants it in.

OBJECTIVE is funcalled inside this method's own `with-foreign-float-traps-masked' body wrap,
so the caller's Lisp arithmetic runs under the masked convention on x86-64 as well as on
aarch64 -- `(/ 1.0d0 0.0d0)' yields infinity there rather than signalling
`division-by-zero'. Nothing about that is specific to a custom objective; it is simply where
in `train' the caller's code now runs. A condition the caller's function does signal
propagates out of `train' through the `with-pointer-ownership' form below, freeing the raw
booster handle rather than orphaning it, exactly as a mid-loop foreign failure does.

An objective that frees a handle this loop depends on, or closes BACKEND, is caught rather
than crashed on: `%recheck-train-datasets' re-runs this method's own opening checks the
moment the `funcall' returns, and TRAIN-DATA-POINTER is reassigned from what it returns, so
nothing after the caller's code uses a pointer read before it. See that function for what
each of the three re-checks is for. This is the only place the loop needs it -- the
OBJECTIVE NIL branch beside it runs no caller code at all.

Neither RECORD-HISTORY nor EARLY-STOPPING is disabled by OBJECTIVE, and neither is made
meaningful by it: a metric configured through PARAMETERS relates to the library's own
objective, not to the caller's, and this method neither signals nor warns about that -- see
the `train' generic function's docstring, which states it as the caller's decision.

EVALUATION adds the caller's own metric to what each iteration records, one call per dataset
per iteration -- see the `train' generic function's docstring for what that function is
called with and what it must return. Signals `capability-unavailable' naming
`:custom-evaluation' when the capability reads false, and `unsupported-argument' naming
\"train's :evaluation\" for RECORD-HISTORY NIL or for a non-function, all three before any
foreign call: see `%check-custom-evaluation' above, and `*optional-symbols*' for why the
answer here is probed rather than declared. EVALUATION NIL, the default, reaches no check and
records exactly what this method has always recorded.

The calls happen after this iteration's own `%read-evaluation' and BEFORE the history push
and the watcher, in `%custom-evaluation-entries' -- so the entries the history keeps and the
entries the watcher sees are one list, and `:early-stopping' can watch a custom metric with
nothing here to arrange it. The custom entries are APPENDED after every library entry, which
is what makes `training-report-from-history''s first-seen ordering put the library's series
first, as a prefix, exactly where `evaluation' reports them; a custom metric never reaches
`evaluation' at all, that method reading only `LGBM_BoosterGetEval'.

Each dataset's predictions are read with `%booster-predictions' at that dataset's own
`data_idx' -- `LGBM_BoosterGetNumPredict' then `LGBM_BoosterGetPredict', the values LightGBM
already holds, rather than a fresh `predict' over a matrix this method does not have. They
are `predict :kind :normal''s numbers and not the margin OBJECTIVE is handed, measured on
both datasets; see `%booster-predictions', which records that measurement and the per-dataset
length it rests on. ROW-COUNTS is read once before the loop, from the same pointers this
method already validated, and only when EVALUATION is non-NIL, so a run that asks for no
custom metric makes no extra foreign call at all.

An EVALUATION that frees a handle this loop depends on, or closes BACKEND, is caught the same
way an OBJECTIVE is: `%custom-evaluation-entries' calls `%recheck-train-datasets' the moment
each `funcall' returns -- between two consecutive datasets' reads, not once per iteration --
and TRAIN-DATA-POINTER is reassigned from what it returns.

DATASET and every VALID-SETS entry's dataset half are each run through
`%check-lightgbm-dataset' before any foreign call. `train' dispatches on
BACKEND, not on DATASET, so unlike `dataset-num-rows' or `free-dataset' there
is no CLOS specializer here to rule out the wrong kind of handle first --
without this, `handle-live-pointer' would happily hand `LGBM_BoosterCreate' a
booster's own pointer to use as its training-set `DatasetHandle'. Signals
`wrong-backend-reference' when DATASET or a VALID-SETS entry's dataset half is
not a `lightgbm-dataset', and `released-handle-error' or `backend-not-open'
when one is but has already been freed or had its own backend closed. A
VALID-SETS entry that is a cons with a non-string car never reaches this check
at all: `%valid-set-name' signals `unsupported-argument' for it first, which is
the different mistake a malformed name is, kept distinct from a wrong dataset
handle.

The returned booster retains DATASET as its training set and a fresh copy of
VALID-SETS as its validation sets, keeping all of them alive for the booster's
lifetime and letting `update-one-iteration' notice if any is freed out from
under it -- see `%check-booster-datasets-live'. The copy matters: VALID-SETS is
the caller's own list, and `make-handle' would otherwise store that exact list
object rather than a snapshot of it. A caller who destructively removes an
entry from VALID-SETS after `train' returns -- `delete', `(setf (cdr ...))',
reusing the list elsewhere with `nconc' -- would silently remove it from the
booster's view too, since both would be the same cons cells; the dataset
`LGBM_BoosterAddValidData' already attached would then go unchecked by
`%check-booster-datasets-live' even though LightGBM still holds its pointer.
Free the result with `free-booster' or wrap it in `with-booster'.

The raw booster handle exists in C from the moment `LGBM_BoosterCreate' returns,
but `make-handle' does not take ownership of it until the very end -- a stale
VALID-SETS entry or a mid-loop failure can each signal first.
`with-pointer-ownership' spans exactly that gap: the pointer is owned by nobody
inside its body, and any exit that has not called TAKE-OWNERSHIP frees the raw
booster here instead of orphaning it.

Signals `backend-not-open' before any of that when BACKEND is not open -- see
`%check-backend-open'."
  (with-foreign-float-traps-masked
    (%check-backend-open backend)
    (%check-custom-objective backend objective)
    (%check-custom-evaluation backend evaluation record-history)
    (let* ((valid-set-entries (copy-list valid-sets))
           (train-data-pointer
             (%check-lightgbm-dataset backend dataset "train's dataset argument"
                                       'lightgbm-dataset))
           (valid-set-names
             (mapcar (lambda (entry) (%valid-set-name backend entry)) valid-set-entries))
           (valid-sets (mapcar #'%valid-set-dataset valid-set-entries))
           (valid-set-pointers
             (mapcar (lambda (valid-set)
                       (%check-lightgbm-dataset
                        backend valid-set "a train :valid-sets entry" 'lightgbm-dataset))
                     valid-sets))
           (dataset-count (1+ (length valid-set-pointers)))
           (dataset-names (cons nil valid-set-names))
           ;; Read once, and only for a run that actually has a custom metric to hand
           ;; predictions to: a built dataset's row count cannot change, and an integer read
           ;; here cannot go stale into a fault the way a pointer can. NIL otherwise, so a
           ;; run without EVALUATION makes not one extra foreign call.
           (row-counts (when evaluation
                         (mapcar #'%dataset-num-rows
                                 (cons train-data-pointer valid-set-pointers))))
           ;; One pin for the whole run, so `pin-metric-name' can compare this iteration's
           ;; name against the first iteration's. NIL without EVALUATION, which allocates
           ;; nothing for a run that has no custom metric to hold to a name.
           (name-pin (when evaluation (make-metric-name-pin)))
           ;; Built before `LGBM_BoosterCreate', so a malformed spec -- or one asking for
           ;; early stopping with RECORD-HISTORY NIL -- signals with no raw booster handle
           ;; in existence yet to unwind. NIL when EARLY-STOPPING is NIL, which is what the
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
      ;; Built here rather than by `cl-gbdt/src/lightgbm/api''s `create-booster', which is the
      ;; Layer 1 function for exactly this. `train' is the ONE method in this file whose Layer
      ;; 1 counterpart exists and is not called: five of the thirteen delegate their whole
      ;; procedure -- `make-dataset', `predict', `update-one-iteration', `free-dataset' and
      ;; `free-booster' -- and the other seven have no counterpart to delegate to at all.
      ;; `booster-best-iteration' is what holds this one back, and it is a barrier rather than
      ;; a preference: a `:reader'-only slot (src/handle.lisp) whose sole writer is
      ;; `make-handle''s :BEST-ITERATION initarg, at construction, while the value comes from
      ;; the watcher AFTER the loop -- so this method must still own the raw pointer when the
      ;; loop ends, and `create-booster' builds its handle before the first iteration. Giving
      ;; the slot a writer to merge the two is not this file's to do.
      ;;
      ;; A second reason stood here and is DEMOTED to a preference: that this ownership form
      ;; is what frees the raw booster when the loop signals, where a handle built up front
      ;; would be left unreferenced with a finalizer that only warns. True of the code as
      ;; written, but a `train' that built the handle first could free it from an
      ;; `unwind-protect' just as well, so it argues for the shape this method already has
      ;; rather than against the merge. It also came with a claim about coverage that does not
      ;; hold: the two tests named after leaking --
      ;; `lightgbm-api-make-dataset-wrong-length-label-signals-without-leaking' and its
      ;; XGBoost twin -- are about a raw DATASET pointer, not a booster, and each asserts only
      ;; that `make-dataset' signals. A leaked raw C pointer has no Lisp object or finalizer
      ;; whose absence a test could observe, which is exactly what the LightGBM one's own
      ;; commentary says.
      (let ((booster-pointer
              (%create-booster train-data-pointer
                               (%parameter-string
                                (if objective (objective-parameters parameters) parameters)))))
        (with-pointer-ownership (booster-pointer #'%free-booster-unchecked take-ownership)
          (%add-valid-data booster-pointer valid-set-pointers)
          ;; ROUND is 1-based, which is the numbering `observe-iteration' answers
          ;; `watcher-best-iteration' in and the report publishes.
          (loop :for round :from 1 :to num-rounds
                :do (if objective
                        (let ((scores (%booster-predictions
                                       booster-pointer 0
                                       (%dataset-num-rows train-data-pointer))))
                          (multiple-value-bind (grad hess) (funcall objective scores)
                            ;; Before anything else this iteration does, and before the
                            ;; next one reads TRAIN-DATA-POINTER again: the caller's
                            ;; own code has just run and may have freed a handle this
                            ;; loop holds a raw pointer to.
                            (setf train-data-pointer
                                  (%recheck-train-datasets backend dataset valid-sets))
                            (check-objective-result grad hess
                                                    (array-dimension scores 0)
                                                    (array-dimension scores 1))
                            (%update-one-iteration-custom booster-pointer grad hess)))
                        (%update-one-iteration booster-pointer))
                    (incf completed-rounds)
                    (let ((entries (when record-history
                                     (%read-evaluation booster-pointer dataset-count))))
                      ;; Appended after every library entry, and before the push and
                      ;; the watcher, so the history and the watcher see one list.
                      (when evaluation
                        (multiple-value-bind (custom pointer)
                            (%custom-evaluation-entries
                             backend evaluation booster-pointer dataset valid-sets
                             row-counts entries (= round 1) name-pin)
                          (setf entries (append entries custom)
                                train-data-pointer pointer)))
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
            (values (take-ownership 'lightgbm-booster backend :booster
                                    :training-set dataset
                                    :validation-sets valid-sets
                                    :best-iteration best-iteration)
                    report)))))))

(defmethod update-one-iteration ((booster lightgbm-booster))
  "Advance BOOSTER by one boosting iteration via `LGBM_BoosterUpdateOneIter'.

Returns false when the iteration just run produced no further split, per the generic
function's contract -- about that call, not a latch for the rest of the run; see
`cl-gbdt/src/lightgbm/api''s function of the same name, which spells that
distinction out. Signals `released-handle-error' when BOOSTER's training set,
or any of its validation sets, has already been freed -- see
`%check-booster-datasets-live'.

This method's whole body was procedure -- there was no portable argument here to check or
translate -- so all of it is `cl-gbdt/src/lightgbm/api''s `update-one-iteration', which is
where the liveness check and the `produced_empty_tree' inversion now live."
  (with-foreign-float-traps-masked
    ;; Named in full, not imported: `cl-gbdt/src/protocol''s `update-one-iteration', the
    ;; generic this method is defined on, is a DIFFERENT symbol of the same name, and this
    ;; file imports that one. Not recursion. The `with-foreign-float-traps-masked' wrap stays
    ;; even though the callee establishes its own, for the reason `free-dataset' above gives.
    (cl-gbdt/src/lightgbm/api:update-one-iteration booster)))

(defmethod free-booster ((booster lightgbm-booster))
  "Free BOOSTER via `LGBM_BoosterFree'. Does nothing if it was already freed.

See `free-dataset''s docstring for why this does not signal `backend-not-open' when
BOOSTER's backend has already been closed -- the same `with-booster' cleanup-form
reasoning applies here.

This method's whole body was procedure too, and is `cl-gbdt/src/lightgbm/api''s
`free-booster' entirely, where the closed-backend branch now lives beside the identical one
`free-dataset' takes."
  (with-foreign-float-traps-masked
    ;; Named in full, not imported, and not recursion -- see `update-one-iteration' above,
    ;; which faces the same doubled name for the same reason.
    (cl-gbdt/src/lightgbm/api:free-booster booster)))

;;; ---------------------------------------------------------------------------
;;; Inference

(defmethod predict ((booster lightgbm-booster) matrix
                    &key (kind :normal) num-iteration missing)
  "Predict on MATRIX with BOOSTER -- a dense matrix via `LGBM_BoosterPredictForMat', a
`csr-matrix' via `LGBM_BoosterPredictForCSR'.

KIND and NUM-ITERATION are as the `predict' generic function documents. NUM-ITERATION's
:BEST is resolved HERE, by `%resolve-best-num-iteration', and the integer it produces is what
reaches the procedure: `booster-best-iteration' is written by `train' and by nothing else, so
:BEST is a Layer 2 concept and `cl-gbdt/src/lightgbm/api''s `predict' takes an integer or NIL,
refusing the keyword itself with `unsupported-argument' -- see its docstring, which says so
from the other side. Predictions start from iteration 0 -- the protocol exposes no
start-iteration override.

Signals `capability-unavailable' naming `:missing-value' for a non-NIL MISSING, whatever the
value is and whatever form MATRIX takes -- this backend has no C-API route for a
missing-value sentinel at all, on the prediction path any more than on the ingestion one. See
`%check-missing-value' above, which carries the reasoning; `make-dataset' calls that same
function, and this is a second call site rather than a copy of it, because policy section 7
asks each operation to re-check the capability for itself. MISSING NIL, the default, reaches
no check: every prediction that names no sentinel behaves exactly as it did before the
argument existed.

Signals `capability-unavailable' when MATRIX is a `csr-matrix' and this backend's
`:sparse-input' capability reads false -- see `cl-gbdt/src/lightgbm/api''s
`%check-sparse-input', which checks it before any foreign call. Everything else means exactly
what it means for a dense matrix, `csr-matrix' or not: KIND and NUM-ITERATION are honoured
identically on either path -- all four KINDs included, unlike `cl-gbdt/src/xgboost/api''s
`predict', whose sparse entry point is XGBoost's inplace prediction and covers only two of
them.

Returns the result array and, as a second value, the SHAPE this backend states for it -- a
list of integers in `array-dimensions' order, or NIL where it states none. Nothing here
reports axes the way XGBoost's `out_shape'/`out_dim' pair does, so that value is DERIVED
rather than read back: `:normal' and `:raw' get the result array's own dimensions, `:contrib'
the three axes `contrib-shape' divides the element count into (NIL for any of the four cases
that function's own docstring enumerates), and `:leaf-index' NIL. `%prediction-shape', beside
the procedure in `cl-gbdt/src/lightgbm/api', is where that happens and carries the
measurements. This backend declares `:prediction-shape' in `*provided-capabilities*' to say
the mechanism is here; nothing re-checks that declaration, there being no argument to refuse.

The procedure itself is Layer 1 and lives in `cl-gbdt/src/lightgbm/api''s `predict': the
choice of entry point, the buffer sized from `LGBM_BoosterCalcNumPredict', the OUT-LEN
assertion, the copy-out and the derived shape, together with the `:sparse-input' gate and the
deliberate absence of any NaN or infinity scan over the result. Everything the paragraphs
above promise about those is that function's doing; see its own docstring. What is left here
is the portable contract: the :MISSING gate this backend answers with a refusal, and
resolving :BEST.

Refusing a KIND this backend has no prediction type for moved BELOW BOTH of those, and is
the one thing about this method a caller can observe changing: `%predict-type''s `ecase'
now runs inside the procedure rather than in the same `let' that read the pointer, so a
call wrong in two ways at once is answered by whichever check still runs first. Measured
through `cl-gbdt:predict' against the vendored library, on a booster trained without
:EARLY-STOPPING and so with no best iteration: a bad KIND together with a non-NIL :MISSING
signalled `sb-kernel:case-failure' before the split and signals `capability-unavailable'
now; a bad KIND together with `:num-iteration :best' signalled `sb-kernel:case-failure' and
signals `unsupported-argument'. A bad KIND alone is `sb-kernel:case-failure' either way.
Both changes put a typed `cl-gbdt' condition where an untyped one used to escape, and the
old order could not be restored without calling `%predict-type' here purely for effect,
duplicating a check the procedure already makes."
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
    ;; read the pointer, and the `when' came after. So a booster with no best iteration still
    ;; reports THAT, rather than the `capability-unavailable' :MISSING always signals here.
    (let ((resolved (%resolve-best-num-iteration booster num-iteration
                                                 "predict's :num-iteration")))
      (when missing
        (%check-missing-value (handle-backend booster)))
      ;; Named in full, not imported, and not recursion -- see `update-one-iteration' above,
      ;; which faces the same doubled name for the same reason.
      (cl-gbdt/src/lightgbm/api:predict booster matrix :kind kind :num-iteration resolved))))

;;; ---------------------------------------------------------------------------
;;; Persistence

(defmethod save-model ((booster lightgbm-booster) path &key num-iteration)
  "Save BOOSTER's model to PATH via `LGBM_BoosterSaveModel'.

NUM-ITERATION limits how many trees are saved, :BEST resolved by
`%resolve-best-num-iteration' first; nil saves all of them, which LightGBM spells as 0.
Returns PATH.

The procedure is Layer 1 and lives in `cl-gbdt/src/lightgbm/api''s `save-model'. What is left
here is resolving :BEST, which reads `booster-best-iteration' -- a slot `train' writes and
nothing else does, so the keyword has no meaning below this layer."
  (with-foreign-float-traps-masked
    ;; Read and discarded, and not redundant with the check inside the procedure: the old body
    ;; read the pointer in the same `let' that resolved :BEST, and `let' evaluates its
    ;; initialization forms in order, so a freed BOOSTER handed :BEST reported
    ;; `released-handle-error' rather than `unsupported-argument'. Resolution now runs before
    ;; the delegation, so without this line that order would silently reverse. `predict' above
    ;; carries the same line for the same reason.
    (handle-live-pointer booster)
    (let ((resolved (%resolve-best-num-iteration booster num-iteration
                                                  "save-model's :num-iteration")))
      ;; Named in full, not imported, and not recursion -- see `update-one-iteration' above,
      ;; which faces the same doubled name for the same reason.
      (cl-gbdt/src/lightgbm/api:save-model booster path :num-iteration resolved))))

(defmethod load-model ((backend lightgbm-backend) path)
  "Load a LightGBM model from PATH via `LGBM_BoosterCreateFromModelfile' and
return a new booster.

The returned booster has no training set -- see the `booster' class'
documentation -- since PATH names a model, not a dataset.

The raw booster handle exists in C from the moment `LGBM_BoosterCreateFromModelfile'
returns, but `make-handle' does not take ownership of it until it also succeeds --
mirroring `cl-gbdt/src/xgboost/protocol''s `load-model', which reaches for the same
`with-pointer-ownership' macro for the same reason: nothing here guarantees
`make-handle' cannot signal, and a raw handle it never took ownership of would
otherwise be orphaned rather than freed.

Signals `backend-not-open' before the foreign call when BACKEND is not open --
see `%check-backend-open'.

This method's whole body was procedure -- there was no portable argument here to check or
translate -- so all of it is `cl-gbdt/src/lightgbm/api''s `load-model', which is where the
ownership window, the null-handle check and the backend guard now live."
  (with-foreign-float-traps-masked
    (cl-gbdt/src/lightgbm/api:load-model backend path)))

(defmethod model-to-string ((booster lightgbm-booster) &key num-iteration)
  "Return BOOSTER's model as a string via `LGBM_BoosterSaveModelToString'.

NUM-ITERATION's :BEST is resolved by `%resolve-best-num-iteration' before
`%resolve-num-iteration' ever sees it, exactly as `predict' and `save-model' resolve it.

The procedure is Layer 1 and lives in `cl-gbdt/src/lightgbm/api''s `model-to-string'. What is
left here is resolving :BEST, exactly as `save-model' above."
  (with-foreign-float-traps-masked
    ;; See `save-model' above: discarded, and load-bearing for the order two wrong arguments
    ;; are reported in.
    (handle-live-pointer booster)
    (let ((resolved (%resolve-best-num-iteration booster num-iteration
                                                  "model-to-string's :num-iteration")))
      (cl-gbdt/src/lightgbm/api:model-to-string booster :num-iteration resolved))))

;;; ---------------------------------------------------------------------------
;;; Feature importance

(defmethod feature-importance ((booster lightgbm-booster) &key (kind :split) num-iteration)
  "Return BOOSTER's per-feature importances via `LGBM_BoosterFeatureImportance'.

The result has one entry per feature. The width comes from
`LGBM_BoosterGetNumFeature', which works whether BOOSTER came from `train' or
`load-model' -- unlike a booster's training set, which `load-model' leaves
unbound.

NUM-ITERATION does not accept :BEST, unlike `predict', `save-model' and
`model-to-string' -- `%reject-best-num-iteration' signals `unsupported-argument' for it
rather than letting it reach `LGBM_BoosterFeatureImportance' as raw, uninterpreted data.

The procedure is `cl-gbdt/src/lightgbm/api''s `feature-importance', and so is the :BEST
refusal: unlike `save-model' and `model-to-string', this operation never RESOLVED :BEST, and a
refusal is something Layer 1 can make for itself with the same argument name and the same
condition. Nothing is left here."
  (with-foreign-float-traps-masked
    (cl-gbdt/src/lightgbm/api:feature-importance booster :kind kind
                                                          :num-iteration num-iteration)))

;;; ---------------------------------------------------------------------------
;;; Evaluation

(defmethod evaluation ((booster lightgbm-booster))
  "Return BOOSTER's evaluation metrics via `%read-evaluation', the pointer-level reader this
backend shares between this method and `train''s per-iteration recording loop, so the two
can never disagree -- see the `evaluation' generic function's docstring for the portable
contract this satisfies.

Reads one `LGBM_BoosterGetEval' result per dataset BOOSTER retains, in the order
`train' attached them, and pairs entry N of each with entry N of the single metric-name
list `LGBM_BoosterGetEvalNames' reports for the whole booster -- LightGBM configures
metrics once per booster, not per dataset, so the names are read once and reused for
every dataset rather than re-read per index.

DATASET-INDEX is passed straight through to `LGBM_BoosterGetEval' as its `data_idx':
this backend's own numbering already is the portable contract's numbering, 0 for the
training set and 1 upward for each `:VALID-SETS' entry in order, so nothing here
renumbers anything. The datasets are counted from BOOSTER's own retained handles rather
than asked of the library, which has no entry point reporting how many are attached; a
`load-model' booster retains none, and is the case that count is 0 for. LightGBM does
answer `data_idx' 0 for such a booster -- with an empty result, confirmed against the
vendored library, since a model file carries no metrics -- but there is no dataset behind
that index for the portable contract to name, so it is not evaluated at all rather than
reported as index 0.

The values are `LGBM_BoosterGetEval''s own doubles, returned unmodified, which is what
the secondary value's `:value-source :library-doubles' says; unlike XGBoost's, nothing
here parses text, so there is no :RAW to keep and no VALUE is ever NIL.

`%check-booster-datasets-live' runs before any foreign call this method makes, including
inside `%read-evaluation': `LGBM_BoosterGetEval' evaluates each attached validation set
through the metric objects built over that dataset's own label and weight arrays, none of
which `LGBM_DatasetFree' clears from the booster, so evaluating after one of them was
freed is a use-after-free rather than a catchable condition -- the identical hazard
`update-one-iteration' guards against with the same call.

The procedure is `cl-gbdt/src/lightgbm/api''s `evaluation'. One thing about it changed with
the move and is worth knowing here: the kind check now runs before
`%check-booster-datasets-live' rather than after, so a value that is not a booster at all is
answered with `wrong-backend-reference' where it used to reach `booster-training-set' and
produce a bare CLOS `no-applicable-method'. Every other order is as it was."
  (with-foreign-float-traps-masked
    (cl-gbdt/src/lightgbm/api:evaluation booster)))
