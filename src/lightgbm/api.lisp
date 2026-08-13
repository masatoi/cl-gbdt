;;;; api.lisp --- LightGBM's Layer 1 operations, over the handle classes.
;;;;
;;;; `native.lisp' holds the `%'-functions: each takes and returns raw pointers, and none of them
;;;; is a finished operation. `classes.lisp' holds the CLOS types and the shared library's
;;;; lifetime. This file is what a caller of `cl-gbdt/lightgbm' actually invokes -- operations
;;;; that take a backend or a handle, do the whole job, and hand back a handle or a result.
;;;;
;;;; Every OPERATION here reaches the shared library, so every one wraps its whole body in
;;;; `with-foreign-float-traps-masked' -- see `protocol.lisp''s header for why, and
;;;; `tools/ci/check-float-traps.lisp', which scans this file and holds every name the sibling
;;;; `all.lisp' exports to that rule. The `%'-prefixed helpers are not separately wrapped and
;;;; are not policed by that check: each runs inside the body wrap of an operation here --
;;;; `%dataset-pointer' inside `create-dataset', `%prediction-shape' inside `predict', and
;;;; `%check-sparse-input' inside both -- and each says so where it is defined, since nothing
;;;; in CI would notice a future caller reaching one from outside an already-masked extent.
;;;;
;;;; Nothing here may depend on `cl-gbdt/src/protocol' or the training files: this file is Layer 1,
;;;; and `tools/ci/check-layer-separation.lisp' fails the build if it ever does. That is not a
;;;; stylistic preference. A caller who loaded `cl-gbdt/lightgbm' alone has no unified API in the
;;;; image, and these functions are the whole of what they can call.
;;;;
;;;; Loads after `classes.lisp' and cannot precede it: every operation below takes or returns a
;;;; `lightgbm-dataset' or a `lightgbm-booster', and `%reference-pointer',
;;;; `%check-lightgbm-dataset' and `%check-object-class' are each handed a class name as a
;;;; symbol.
;;;;
;;;; Every operation below that takes a caller-supplied HANDLE checks its kind before letting
;;;; its pointer reach C, and none of them may skip that: each was a `defmethod' specialized on
;;;; `lightgbm-dataset' or `lightgbm-booster' before the Layer 1 split, and that specializer WAS
;;;; the check. A plain `defun' takes whatever it is given. `%check-lightgbm-dataset' and
;;;; `%check-lightgbm-booster' are what the operations that require a LIVE handle use;
;;;; `%check-object-class' below is what `free-dataset' and `free-booster' use instead, they
;;;; being the two that must keep working on a handle that is neither, and what
;;;; `dataset-num-rows' and `dataset-num-features' pair with `handle-live-pointer' for, neither
;;;; having a BACKEND argument of its own to hand `%check-lightgbm-dataset' for its report.
;;;;
;;;; The three operations that take a caller-supplied BACKEND rather than a handle --
;;;; `create-dataset', `create-booster' and `load-model' -- are under the identical rule for the
;;;; identical reason, and `%check-object-class' is their check too, handed `lightgbm-backend'
;;;; where the frees hand it a handle class. Their `defmethod' ancestors specialized on
;;;; `lightgbm-backend', so the backend argument was type-checked by the same mechanism the
;;;; handle arguments were, and `%check-backend-open' does not replace it: that function asks
;;;; whether the object is OPEN, not whose it is.

(uiop:define-package #:cl-gbdt/src/lightgbm/api
  (:use #:cl)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt/src/lightgbm/classes
                #:lightgbm-backend
                #:lightgbm-booster
                #:lightgbm-dataset)
  (:import-from #:cl-gbdt/src/lightgbm/native
                #:%add-valid-data
                #:%booster-num-features
                #:%calc-num-predict
                #:%check-backend-open
                #:%check-booster-datasets-live
                #:%check-lightgbm-booster
                #:%check-lightgbm-dataset
                #:%create-booster
                #:%create-booster-from-modelfile
                #:%create-dataset
                #:%create-dataset-from-csr
                #:%create-dataset-from-file
                #:%data-type
                #:%dataset-num-features
                #:%dataset-num-rows
                #:%feature-importance
                #:%feature-importance-type
                #:%free-booster
                #:%free-booster-unchecked
                #:%free-dataset
                #:%free-dataset-unchecked
                #:%parameter-string
                #:%predict-for-csr
                #:%predict-for-mat
                #:%predict-ncol
                #:%predict-type
                #:%read-evaluation
                #:%reference-pointer
                #:%resolve-num-iteration
                #:%save-model
                #:%save-model-to-string
                #:%set-feature-names
                #:%set-group-field
                #:%set-info-field
                #:%update-one-iteration)
  (:import-from #:cl-gbdt/src/backend
                #:backend-name
                #:backend-open-p
                #:backend-supports-p)
  (:import-from #:cl-gbdt/src/conditions
                #:capability-unavailable
                #:foreign-call-error
                #:unsupported-argument
                #:wrong-backend-reference)
  (:import-from #:cl-gbdt/src/config/prediction-shape
                #:contrib-shape)
  (:import-from #:cl-gbdt/src/data
                #:csr-matrix
                #:csr-matrix-indices
                #:csr-matrix-indptr
                #:csr-matrix-num-columns
                #:csr-matrix-num-rows
                #:csr-matrix-values
                #:with-foreign-matrix)
  (:import-from #:cl-gbdt/src/foreign
                #:with-foreign-float-traps-masked)
  (:import-from #:cl-gbdt/src/handle
                #:%reject-best-num-iteration
                #:booster-training-set
                #:booster-validation-sets
                #:handle-backend
                #:handle-live-pointer
                #:handle-released-p
                #:release-handle
                #:with-pointer-ownership)
  (:export #:create-booster
           #:create-dataset
           #:create-dataset-from-file
           #:dataset-num-features
           #:dataset-num-rows
           #:evaluation
           #:feature-importance
           #:free-booster
           #:free-dataset
           #:load-model
           #:model-to-string
           #:predict
           #:save-model
           #:update-one-iteration))

(in-package #:cl-gbdt/src/lightgbm/api)

;;; ---------------------------------------------------------------------------
;;; The `:sparse-input' gate

(defun %check-sparse-input (backend)
  "Signal `capability-unavailable' when BACKEND's `:sparse-input' capability reads false.

Policy section 7 requires the operation itself to re-check a capability rather than trusting
the caller to have asked `backend-supports-p' first, so a caller who never asked gets a typed
condition instead of a missing-symbol crash. Both operations this backend gates on
`:sparse-input' call this -- `%dataset-pointer' below, on `create-dataset''s behalf, and
`predict' at the end of this file -- so the two cannot come to disagree about which capability
they name or which backend they blame.

Not exported, unlike every other name this file defines that a caller reaches. It was, while
`cl-gbdt/src/lightgbm/protocol''s `predict' held the prediction procedure and imported this
gate from here; that procedure is now `predict' below, both call sites are in this file, and
an export nothing outside it needs is one more claim to keep true.

Only a `csr-matrix' argument ever reaches this. A dense matrix needs neither
`LGBM_DatasetCreateFromCSR' nor `LGBM_BoosterPredictForCSR' to exist, and must keep working
on a library that has neither."
  (unless (backend-supports-p backend :sparse-input)
    (error 'capability-unavailable
           :backend (backend-name backend) :capability :sparse-input)))

;;; ---------------------------------------------------------------------------
;;; The class gate, for the frees and the creators

(defun %check-object-class (object class noun argument-description)
  "Signal `wrong-backend-reference' unless OBJECT is of type CLASS, reporting NOUN as the kind
of thing that was wanted and ARGUMENT-DESCRIPTION as the argument OBJECT came from. Returns
no useful value, and reads nothing else about OBJECT at all.

Seven callers, in three groups, and OBJECT is a handle for two of the groups and a BACKEND
for the third -- which is why this is not named for handles. Each group reaches it for its
own reason:

`free-dataset' and `free-booster' cannot use `%check-lightgbm-dataset' or
`%check-lightgbm-booster', which every operation in this file that requires a LIVE handle
uses: both of those end in `handle-live-pointer', so both REQUIRE the handle to be unreleased
and its backend still open. Those are precisely the two states a free is documented to
tolerate -- freeing an already-freed handle is a no-op, and freeing one whose backend has
since been closed `warn's the memory leaked and returns -- so routing either free through a
check that signals `released-handle-error' or `backend-not-open' would break both promises for
the sake of the kind check. This checks the kind and nothing else.

`create-dataset', `create-booster' and `load-model' have no handle to check at all: their
caller-supplied object is the BACKEND, and each calls this FIRST, ahead of
`%check-backend-open'. That order is the point rather than a detail. `%check-backend-open'
asks only whether the object it is handed is OPEN, and for another backend's object that is a
truthful answer to a question about the wrong shared library, which lets the call proceed; for
a value that is no backend at all, it reaches `backend-open-p' and produces a bare CLOS
`no-applicable-method' instead of a typed condition.

`dataset-num-rows' and `dataset-num-features' need a LIVE handle, unlike the frees, but have
no BACKEND argument of their own to hand `%check-lightgbm-dataset' for its report -- neither
signature takes one, both being readers of an existing DATASET rather than builders -- so each
pairs this check with `handle-live-pointer' instead, calling it only once this one has already
confirmed DATASET's kind: `wrong-backend-reference' for the wrong kind, and whatever
`handle-live-pointer' signals otherwise, `released-handle-error' for an already-freed DATASET
or `backend-not-open' for one whose backend has since closed.

A `typep' against the CONCRETE class subsumes kind and backend together, where
`cl-gbdt/src/handle''s `%check-handle-kind' needs a (kind, backend-name) pair: that function
takes the backend-agnostic `dataset'/`booster' classes because `native.lisp' may not depend on
`classes.lisp' (policy section 11), whereas this file already names `lightgbm-dataset',
`lightgbm-booster' and `lightgbm-backend' -- and a `lightgbm-dataset' is by construction a
dataset built by LightGBM, no other backend's operations building one.

NOUN is passed rather than derived from CLASS: `wrong-backend-reference''s report reads \"~A
must be a ~A built by ~A itself\", where the class name would make it \"must be a
lightgbm-dataset built by LIGHTGBM\" and say the backend twice. \"backend\" is the third value
NOUN takes, after \"dataset\" and \"booster\", and the one that condition's `:report' words
differently: a backend object is not something the backend BUILT, so it reads as its OWN.

All seven call sites were `defmethod's specialized on the concrete class before the Layer 1
split, and that specializer WAS this check; a plain `defun' takes whatever it is given, so it
is made here or nowhere. What the frees and the creators let through without it was measured
against the vendored libraries, and both measurements are worth reading for their SPREAD
rather than for any one number.

The frees hand the pointer of whatever handle they are given straight to `LGBM_DatasetFree' or
`LGBM_BoosterFree': a `lightgbm-booster' passed to `free-dataset' faulted at
#xFFFFFFFFFFFFFFF8, and the image then aborted -- glibc's \"free(): invalid pointer\", SIGABRT
-- when the caller's own cleanup form went on to free that booster properly. An
`xgboost-dataset' passed to the same function faulted at that same address in one measurement
and RETURNED SILENTLY in another.

The creators build against a library their backend argument had no say in: `create-dataset'
handed the XGBOOST backend ACCEPTED it and returned a `lightgbm-dataset' --
`LGBM_DatasetCreateFromMat' built it, that C entry point never seeing a backend object at all
-- whose `handle-backend' is the XGBoost backend that did not build it. Every later liveness
question about that handle then consults the wrong library's state: after `close-backend' on
the LightGBM backend that DID build it, the handle still read live (`handle-released-p' NIL,
its recorded backend open), and `free-dataset' on it called `LGBM_DatasetFree' into a library
`close-backend' had unloaded -- which RETURNED NORMALLY.

Those spreads are the argument for checking in Lisp. Reaching a foreign library through
another library's entry point, or through one the caller has unloaded, is undefined behaviour,
so what it does is not a property either library reports and not something a caller can be
told to handle -- and a silent return is the dangerous outcome, not the safe one.

`%check-lightgbm-dataset' remains what `create-booster' checks its DATASET with, and it stays
deliberately identity-blind: two `lightgbm-backend' instances over the same shared library are
a legitimate way for a caller to hold datasets, so a dataset built by one is not refused when
the other trains on it. The backend check here rules out the other backend CLASS, a different
fault -- a different library entirely, not a second handle on the same one."
  (unless (typep object class)
    (error 'wrong-backend-reference
           :backend :lightgbm
           :given (class-name (class-of object))
           :argument argument-description
           :expected noun)))

;;; ---------------------------------------------------------------------------
;;; Datasets

(defun %dataset-pointer (backend matrix parameter-string reference-pointer)
  "Return two values: the raw LightGBM dataset pointer built from MATRIX, and the name
of the C function that produced it, for the null-handle check `create-dataset' makes
afterward.

MATRIX is either a `csr-matrix' -- `LGBM_DatasetCreateFromCSR', through
`%create-dataset-from-csr' -- or anything `with-foreign-matrix' accepts --
`LGBM_DatasetCreateFromMat', through `%create-dataset'. PARAMETER-STRING and
REFERENCE-POINTER reach both entry points identically: neither argument means anything
different for a sparse matrix than for a dense one, which is exactly what `make-dataset''s
own contract promises about them.

The `:sparse-input' capability is re-checked on the sparse branch rather than assumed --
`%check-sparse-input' above, which carries the reasoning.

A `defun', not a second `create-dataset' method specialized on `csr-matrix': `create-dataset'
is a plain function taking MATRIX as a required argument, so a method pair would first have to
make it generic and would then split every one of the shared steps it holds -- the ownership
dance, LABEL, WEIGHT, GROUP, FEATURE-NAMES -- across two bodies that must not drift, to vary
one call. This keeps that whole procedure single and varies only the call that actually
differs."
  (if (typep matrix 'csr-matrix)
      (progn
        (%check-sparse-input backend)
        (values (%create-dataset-from-csr (csr-matrix-indptr matrix)
                                          (csr-matrix-indices matrix)
                                          (csr-matrix-values matrix)
                                          (csr-matrix-num-columns matrix)
                                          parameter-string reference-pointer)
                "LGBM_DatasetCreateFromCSR"))
      (values (%create-dataset matrix parameter-string reference-pointer)
              "LGBM_DatasetCreateFromMat")))

(defun create-dataset (backend matrix &key label weight group feature-names parameters
                                        reference)
  "Build and return a `lightgbm-dataset' from MATRIX on BACKEND.

MATRIX is a dense matrix -- built via `LGBM_DatasetCreateFromMat' -- or a `csr-matrix', built
via `LGBM_DatasetCreateFromCSR'. Nothing else about this function varies with which of the two
it is: see `%dataset-pointer' above, the only form here that branches on it.

LABEL and WEIGHT are attached to the finished dataset with `LGBM_DatasetSetField' under those
LightGBM field names, GROUP with the same call under `group', and FEATURE-NAMES with
`LGBM_DatasetSetFeatureNames'. Each is attached only when supplied; a NIL one is not written
as an empty field. REFERENCE is another `lightgbm-dataset' whose bin mapper this one should
align to -- what `LGBM_DatasetCreateFromMat' calls its `reference' argument, and what a
validation set needs to be binned the same way as the training set it will be scored against
-- or NIL for none.

PARAMETERS is a plist in LightGBM'S OWN vocabulary, rendered by `%parameter-string' and handed
to the creation call verbatim; nothing here translates a key or a value, and no key is added.
`:categorical-feature' is one such key among the rest -- this function does not know it as a
concept, and a caller who wants columns read as categories writes that key here, already
rendered as the string LightGBM expects. `cl-gbdt/src/lightgbm/protocol''s `make-dataset' is
what turns the portable :CATEGORICAL-FEATURES argument into exactly that entry before calling
this.

Signals `wrong-backend-reference' when BACKEND is not a `lightgbm-backend' -- the other
backend's object, or not a backend at all -- before anything else is read from it and ahead of
the openness check below, which for another backend's object would answer about the wrong
shared library; see `%check-object-class'. Signals `backend-not-open' before any foreign
call when BACKEND is not open -- see `%check-backend-open'. Signals `capability-unavailable'
naming `:sparse-input' when MATRIX is a `csr-matrix' and that capability reads false,
`wrong-backend-reference' when REFERENCE is supplied and is not a `lightgbm-dataset',
`released-handle-error' when it has already been freed, and `backend-not-open' when its own
backend has since been closed -- see `%reference-pointer'. Signals `foreign-call-error' when
the creation call reports success but writes a null handle: a library-contract violation, but
one every later call through this handle would otherwise dereference blindly.

The raw dataset handle exists in C from the moment the creation call returns, but
`make-handle' does not take ownership of it until the very end -- attaching LABEL, WEIGHT,
GROUP or FEATURE-NAMES can each signal first (a wrong-length LABEL is the commonest way).
`with-pointer-ownership' spans exactly that gap: the pointer is owned by nobody inside its
body, and any exit that has not called TAKE-OWNERSHIP frees the raw dataset here instead of
orphaning it."
  (with-foreign-float-traps-masked
    (%check-object-class backend 'lightgbm-backend "backend"
                         "create-dataset's backend argument")
    (%check-backend-open backend)
    (let ((reference-pointer (%reference-pointer backend reference 'lightgbm-dataset))
          (parameter-string (%parameter-string parameters)))
      (multiple-value-bind (dataset-pointer function-name)
          (%dataset-pointer backend matrix parameter-string reference-pointer)
        (when (cffi:null-pointer-p dataset-pointer)
          (error 'foreign-call-error
                 :function-name function-name
                 :code 0
                 :message "reported success but returned a null dataset handle"))
        (with-pointer-ownership (dataset-pointer #'%free-dataset-unchecked take-ownership)
          (when label
            (%set-info-field dataset-pointer "label" label))
          (when weight
            (%set-info-field dataset-pointer "weight" weight))
          (when group
            (%set-group-field dataset-pointer group))
          (when feature-names
            (%set-feature-names dataset-pointer feature-names))
          (take-ownership 'lightgbm-dataset backend :dataset))))))

(defun %check-file-path (path)
  "Signal `unsupported-argument' when PATH is a wild pathname.

`create-dataset-from-file' passes PATH's `sb-ext:native-namestring' to
`LGBM_DatasetCreateFromFile', not its `namestring' -- `namestring' backslash-escapes a
character that is a CL pathname wildcard marker but literal in a real filename (an
asterisk, concretely), so a file genuinely named `star*file.libsvm' namestrings as
`star\\*file.libsvm' and LightGBM refuses to open a file that exists, reporting a path
the caller never wrote. Review round 4's Finding N9 fixed this on XGBoost's side only;
LightGBM's `create-dataset-from-file' landed two rounds earlier and was never revisited.

But `native-namestring' signals `sb-kernel:no-native-namestring-error' -- untyped by this
project, and uncatchable by name -- for a PATH that IS wild (`wildcard*.libsvm', where the
asterisk really is a CL wildcard marker rather than the literal-character case above), so
switching to it is not a bare substitution: this check has to run FIRST, refusing a wild
PATH with a typed condition this project's own callers can catch, or the fix trades one
edge case for a worse one. Mirrors `cl-gbdt/src/xgboost/file-input''s
`%check-file-uri-arguments', which pairs the identical two calls for the identical reason
-- LightGBM has no URI to compose and so needs none of that function's other checks
(reserved query characters, a smuggled `format' key), only this one."
  (when (wild-pathname-p path)
    (error 'unsupported-argument
           :backend :lightgbm
           :argument "create-dataset-from-file's path"
           :reason (format nil "~S is a wild pathname; LGBM_DatasetCreateFromFile takes a ~
                                plain filename, and sb-ext:native-namestring signals its ~
                                own untyped error for a wild pathname rather than ~
                                composing one" path))))

(defun %best-effort-resolve-path (path)
  "Resolve PATH the way `open' resolves a pathname designator: `truename' when that
succeeds -- symlinks followed, a leading `~' expanded, PATH merged against
`*default-pathname-defaults*' the way an existing file's `truename' always is -- or,
when `truename' signals `file-error' (most concretely a missing file), `MERGE-PATHNAMES'
against `*default-pathname-defaults*' instead, which applies the identical merging rule
without requiring PATH to exist. Never signals and never refuses: every PATH this
function is given comes back as some pathname.

`create-dataset-from-file' passes this function's result, not PATH itself, to
`sb-ext:native-namestring'. Measured directly (not assumed): `native-namestring' on a
bare relative pathname designator prints PATH's own relative spelling verbatim, with NO
merge against `*default-pathname-defaults*' at all. A caller who rebinds that special --
an ordinary CL pattern this project does not itself use, but places no restriction
against -- and passes a relative PATH would therefore have had
`LGBM_DatasetCreateFromFile' open it relative to the OS PROCESS's own working directory
instead, which can differ from `*default-pathname-defaults*' arbitrarily: silently
training on an unrelated file of the same name if one happens to exist there, or failing
to find a file that does exist at the location `open' itself would have resolved PATH to.
This is the third occurrence of a hazard whose definitive analysis lives in
`cl-gbdt/src/xgboost/file-input''s `%resolve-file-path' docstring (review round 3,
Finding N4) -- reproduced here for LightGBM rather than cross-referenced only, since a
reader of one function should not have to open the other backend's source to trust either.

Deliberately not that function transplanted: `%resolve-file-path' REFUSES, returning NIL,
when it cannot resolve to an existing file, because XGBoost's `detect-file-format' gate
must classify the file's own bytes and has nothing to classify without one. This function
has no such gate to feed -- `create-dataset-from-file''s own docstring already promises a
missing PATH is `LGBM_DatasetCreateFromFile''s own message to report, not a `probe-file'
pre-check made here -- so this function never refuses: `MERGE-PATHNAMES' applies CL's
merging rule to a pathname that need not exist, and the missing-file case still reaches C
exactly as it always has, only correctly resolved first, so the C library's own error
names the location `open' would actually have looked at.

Measured, not assumed, that the two branches do not disagree about a leading `~':
`(pathname \"~/x.txt\")' already carries an ABSOLUTE `:HOME' directory component from
parsing alone, before either `truename' or `merge-pathnames' ever sees it, so
`sb-ext:native-namestring' expands it identically from either branch's result -- `~' never
reaches C unexpanded either way.

Runs after `%check-file-path' at every call site, never before: a wild PATH also makes
`truename' signal `file-error' (per the standard), which would route it to the
`merge-pathnames' branch here -- `merge-pathnames' does not resolve a wildcard, so the
result would still be wild, and `native-namestring' would then signal its own untyped
`sb-kernel:no-native-namestring-error' for it. `%check-file-path' refuses a wild PATH
first, with a typed condition, so this function is never actually reached with one."
  (handler-case (truename path)
    (file-error () (merge-pathnames path *default-pathname-defaults*))))

(defun create-dataset-from-file (backend path &key parameters reference)
  "Build and return a `lightgbm-dataset' from PATH on BACKEND, via
`LGBM_DatasetCreateFromFile'.

PARAMETERS is a plist in LightGBM'S OWN vocabulary, rendered by `%parameter-string' and
handed to the creation call verbatim -- exactly as `create-dataset''s own PARAMETERS is,
nothing here translating a key or a value either. `header', `label_column' and `label' are
three such keys among the rest, measured in
`docs/superpowers/specs/2026-08-13-file-input-measurements.md' section 5: `header=true'
consumes the file's first line as a header rather than a data row, `label_column=N' and its
alias `label=N' pick a 0-based column as the label (default 0). REFERENCE behaves exactly as
`create-dataset''s REFERENCE: another `lightgbm-dataset' whose bin mapper this one should
align to, or NIL to build its own.

There is no format argument: unlike XGBoost's counterpart, LightGBM infers PATH's format
-- CSV, TSV or libsvm -- from its own content, so nothing here declares or checks one. A file
whose format the library cannot infer, or that does not exist, is reported through
`check-lgbm' exactly as any other failed foreign call is -- LightGBM's own message to report,
not a `probe-file' pre-check made here.

PATH reaches `LGBM_DatasetCreateFromFile' as `sb-ext:native-namestring' of its OWN
`%best-effort-resolve-path' -- `truename' when PATH resolves, `merge-pathnames' against
`*default-pathname-defaults*' when it does not -- not PATH's bare `native-namestring' and
not its `namestring' either. `namestring' backslash-escapes a literal asterisk in a real
filename, which LightGBM would then fail to open (see `%check-file-path' for why the
`native-namestring' substitution needed a wild-pathname guard rather than being bare:
`native-namestring' signals its own untyped error for a path that is genuinely wild).
`native-namestring' ALONE, without resolving first, would also print a relative PATH's
own relative spelling verbatim, with no merge against `*default-pathname-defaults*' at
all, so `LGBM_DatasetCreateFromFile' would open it relative to the OS process's own
working directory instead -- silently reading an unrelated file of the same name there if
one exists, or failing to find a file that does, at the location `open' itself would have
resolved PATH to. See `%best-effort-resolve-path''s own docstring for the measurement
behind this and for why it never refuses, unlike XGBoost's `%resolve-file-path'.

Signals `wrong-backend-reference' when BACKEND is not a `lightgbm-backend' before anything
else is read from it, and `backend-not-open' before any foreign call when BACKEND is not
open -- see `%check-object-class' and `%check-backend-open'. Signals `unsupported-argument'
when PATH is a wild pathname -- see `%check-file-path'. Signals `wrong-backend-reference'
when REFERENCE is supplied and is not a `lightgbm-dataset', `released-handle-error' when it
has already been freed, and `backend-not-open' when its own backend has since been closed --
see `%reference-pointer'. Signals `foreign-call-error' when the creation call reports success
but writes a null handle, exactly as `create-dataset' does.

The raw dataset handle exists in C from the moment the creation call returns, but there is
nothing left to attach to it afterward -- no LABEL, WEIGHT, GROUP or FEATURE-NAMES argument,
the file itself already carrying whatever `create-dataset''s caller would otherwise pass
separately. `with-pointer-ownership' still spans the gap between that return and
`make-handle', short as it is: policy section 10 asks for it wherever a fresh pointer exists
with nothing in Lisp yet referencing it, not only where the gap is wide."
  (with-foreign-float-traps-masked
    (%check-object-class backend 'lightgbm-backend "backend"
                         "create-dataset-from-file's backend argument")
    (%check-backend-open backend)
    (%check-file-path path)
    (let ((reference-pointer (%reference-pointer backend reference 'lightgbm-dataset))
          (parameter-string (%parameter-string parameters)))
      (let ((dataset-pointer (%create-dataset-from-file
                              (sb-ext:native-namestring (%best-effort-resolve-path path))
                              parameter-string reference-pointer)))
        (when (cffi:null-pointer-p dataset-pointer)
          (error 'foreign-call-error
                 :function-name "LGBM_DatasetCreateFromFile"
                 :code 0
                 :message "reported success but returned a null dataset handle"))
        (with-pointer-ownership (dataset-pointer #'%free-dataset-unchecked take-ownership)
          (take-ownership 'lightgbm-dataset backend :dataset))))))

(defun free-dataset (dataset)
  "Free DATASET via `LGBM_DatasetFree'. Does nothing if it was already freed, and returns no
useful value.

Signals `wrong-backend-reference' when DATASET is not a `lightgbm-dataset' -- a booster, a
dataset built by another backend, or not a handle at all -- before anything is read from it
and before any foreign call. This function dispatches on nothing, so `%check-object-class' is
the only thing between a wrong-kind pointer and `LGBM_DatasetFree' dereferencing it; see that
function for why the frees cannot use the same check the rest of this file uses, and for what
the two wrong handles measured against the vendored library did without it.

Unlike every other operation that reads an existing handle -- `free-booster' below excepted,
which takes this same path for this same reason -- this does not go through
`handle-live-pointer' and so does not signal `backend-not-open' when DATASET's backend has
already been closed. A free runs from a cleanup form -- the `unwind-protect' a Layer 1 caller
writes for itself, as tests/functional/lightgbm-standalone.lisp does, or the one inside
`cl-gbdt''s `with-dataset', which reaches this function through the method that delegates to
it and which a caller of `cl-gbdt/lightgbm' alone does not have -- and a non-local exit is
exactly when that cleanup runs; signalling there would replace whatever condition is already
unwinding the stack instead of letting it propagate. So when
the backend is closed, the handle is instead marked released without calling
`LGBM_DatasetFree' -- the shared library may no longer be mapped into the process, so that
call cannot be trusted not to crash -- and a `warn' reports the foreign memory as leaked,
since it is genuinely unreclaimable at that point."
  (with-foreign-float-traps-masked
    (%check-object-class dataset 'lightgbm-dataset "dataset"
                         "free-dataset's dataset argument")
    (if (backend-open-p (handle-backend dataset))
        (release-handle dataset (lambda (pointer) (%free-dataset pointer)))
        (let ((already-released (handle-released-p dataset)))
          (release-handle dataset (lambda (pointer) (declare (ignore pointer))))
          (unless already-released
            (warn "Freeing a LightGBM dataset after its backend was closed: the foreign ~
                   dataset was not freed and its memory is leaked."))))))

;;; ---------------------------------------------------------------------------
;;; Boosters
;;;
;;; `cl-gbdt/src/lightgbm/protocol''s `train' calls `create-booster' for its whole
;;; construction, the same way every other Layer 1 operation's method already called its own
;;; counterpart. What used to hold `train' back was `booster-best-iteration' being a
;;; `:reader'-only slot writable only at construction, while the value `train' has to write
;;; comes from its own loop's watcher afterward -- `cl-gbdt/src/handle''s
;;; `%set-booster-best-iteration' removed that barrier, so `train' now holds a full handle
;;; from this function's own `make-handle' call and writes the best iteration back once its
;;; loop ends, rather than building the handle itself only at the end. See `train''s own call
;;; site for that write and the reasoning behind it -- and note the same is true of
;;; `cl-gbdt/src/xgboost/api''s `create-booster': this was a property of the shared `handle'
;;; class, not of either library, which is why both backends' `train' carried the same
;;; barrier and both lost it together. The two small functions below
;;; ARE what their methods call, wholesale.

(defun create-booster (backend dataset &key parameters valid-sets)
  "Create a booster over DATASET on BACKEND via `LGBM_BoosterCreate', returning a
`lightgbm-booster'.

The result is UNTRAINED: `LGBM_BoosterCreate' allocates the model and fixes its parameters,
and every boosting iteration comes from a later `update-one-iteration'. Free it with
`free-booster'.

PARAMETERS is a plist in LightGBM'S OWN vocabulary, rendered by `%parameter-string' and
handed to the creation call verbatim; nothing here translates a key or a value, and no key
is added. VALID-SETS is a list of `lightgbm-dataset's -- bare datasets, not the
(NAME . DATASET) entries `cl-gbdt''s `train' accepts, a name being a training-report concept
with no meaning at this layer -- attached afterward with `LGBM_BoosterAddValidData' in the
order given. The booster retains DATASET and a COPY of VALID-SETS, which keeps them alive
for its lifetime and lets `update-one-iteration' notice a dataset freed out from under it.
The copy is what makes that promise hold: were the caller's own list object stored, a later
`delete' or `(setf (cdr ...))' on it would remove an entry from the booster's view while
LightGBM still held that dataset's pointer.

Signals `wrong-backend-reference' when BACKEND is not a `lightgbm-backend' -- the other
backend's object, or not a backend at all -- before anything else is read from it and ahead of
the openness check, which for another backend's object would answer about the wrong shared
library; see `%check-object-class'. Signals `backend-not-open' before any foreign call
when BACKEND is not open -- see `%check-backend-open' -- `wrong-backend-reference' when
DATASET or a VALID-SETS entry is not a `lightgbm-dataset', and `released-handle-error' or
`backend-not-open' when one is but has already been freed or had its own backend closed; see
`%check-lightgbm-dataset', which is what rules out `LGBM_BoosterCreate' being handed a
booster's own pointer as its training set. This function dispatches on nothing, so those two
checks are the only thing standing between a wrong-kind handle -- or another backend's object
where this one belongs -- and a segfault. Signals `foreign-call-error' when creation reports
success but writes a null handle -- that check lives in `%create-booster', beside the call it
guards, and is not repeated here.

Every check runs before the creation call, so a rejected VALID-SETS entry leaves no booster
in existence at all. The raw handle then exists in C from the moment that call returns and
nothing in Lisp references it until `make-handle' runs -- `with-pointer-ownership' spans
exactly that gap, so a validation set that fails to attach frees the booster rather than
orphaning it.

PARAMETERS cannot fail in that gap here, and this is where the two backends' ownership forms
differ. `%parameter-string' is an ARGUMENT to `%create-booster', so a malformed plist -- an
odd-length one, which `normalize-parameters' signals `data-error' for -- signals before any
handle exists; measured, `:parameters '(:eta)' frees no booster at all, because there was
none. On `cl-gbdt/src/xgboost/api' the same plist frees exactly one: that library applies
parameters AFTER creation, so its `%set-parameters' runs inside its ownership form. See its
`create-booster', which carries both measurements."
  (with-foreign-float-traps-masked
    (%check-object-class backend 'lightgbm-backend "backend"
                         "create-booster's backend argument")
    (%check-backend-open backend)
    ;; `let', not `let*': no binding here reads another, so the sequential scope `let*' adds
    ;; would claim a dependency that is not there. Ordering is NOT what separates the two --
    ;; both evaluate their init forms left to right -- and the ordering this function depends
    ;; on is not among these bindings at all: the checks precede the creation call because
    ;; that call sits in the nested form below, in this `let''s body. `%create-booster' is
    ;; deliberately NOT among these bindings, for exactly that reason -- it belongs inside its
    ;; own form, where the raw handle's lifetime begins.
    (let ((train-data-pointer
            (%check-lightgbm-dataset backend dataset "create-booster's dataset argument"
                                      'lightgbm-dataset))
          (valid-set-pointers
            (mapcar (lambda (valid-set)
                      (%check-lightgbm-dataset backend valid-set
                                                "a create-booster :valid-sets entry"
                                                'lightgbm-dataset))
                    valid-sets))
          (validation-sets (copy-list valid-sets)))
      (let ((booster-pointer (%create-booster train-data-pointer
                                              (%parameter-string parameters))))
        (with-pointer-ownership (booster-pointer #'%free-booster-unchecked take-ownership)
          (%add-valid-data booster-pointer valid-set-pointers)
          (take-ownership 'lightgbm-booster backend :booster
                          :training-set dataset
                          :validation-sets validation-sets))))))

(defun update-one-iteration (booster)
  "Advance BOOSTER by one boosting iteration via `LGBM_BoosterUpdateOneIter'.

Returns false when THIS iteration produced no further split -- LightGBM's own
`produced_empty_tree' out parameter, read by `%update-one-iteration' and inverted here, so
that a true return means the iteration did produce one. Per call, and about that call only:
the C function recomputes the flag every time and promises nothing about the next one, so a
false return is not a latch and a loop that stops on it is making its own decision, not
reading one the library has made. `cl-gbdt/src/protocol''s generic says the same in the
portable words \"returns false when no further split was possible\".

Signals `wrong-backend-reference' when BOOSTER is not a booster built by this backend -- a
dataset, an XGBoost booster, or not a handle at all -- and this function dispatches on
nothing, so `%check-lightgbm-booster' is the only thing between such a handle's pointer and
`LGBM_BoosterUpdateOneIter'. Measured against the vendored library with the check absent, an
`xgboost-booster' came back as a `foreign-call-error' rather than a fault, but only because
LightGBM happened to return -1 for that particular pointer: nothing in the C API examines a
handle it is given, so that is a fact about one measurement, not a contract.

Signals `released-handle-error' when BOOSTER's training set, or any of its validation sets,
has already been freed -- see `%check-booster-datasets-live', which runs before any foreign
call because `LGBM_BoosterUpdateOneIter' dereferences those datasets' pointers itself and a
freed one is a segfault rather than a catchable condition. Signals `released-handle-error'
for a freed BOOSTER, and `backend-not-open' when its backend has since been closed -- both
from the `handle-live-pointer' inside `%check-lightgbm-booster', which is why the kind check
is first: a handle this backend never built is the wrong handle whatever its state, and
`%check-booster-datasets-live' would otherwise read slots off it before anything questioned
what it was.

PRECEDENCE, when more than one of those is true at once. BOOSTER's own state is examined
before its datasets', so a fault in the booster or its backend WINS over a freed training set.
Measured against the vendored library, before and after the kind check moved ahead of
`%check-booster-datasets-live':

  booster freed + training set freed   was `released-handle-error' naming the DATASET,
                                       is `released-handle-error' naming the BOOSTER
  training set freed + backend closed  was `released-handle-error' naming the dataset,
                                       is `backend-not-open'
  training set freed alone             `released-handle-error' naming the dataset, unchanged

Both changes are deliberate and neither is a widening: a released handle, or a shared library
`close-backend' has unmapped, is a more fundamental fault than the dataset a still-usable
booster points at, and reporting the dataset while the booster itself was unusable named the
lesser of the two. `cl-gbdt/src/xgboost/api''s `update-one-iteration' records the same
precedence and one further row of its own."
  (with-foreign-float-traps-masked
    (let ((pointer (%check-lightgbm-booster booster "update-one-iteration's booster argument")))
      (%check-booster-datasets-live booster)
      (zerop (%update-one-iteration pointer)))))

(defun free-booster (booster)
  "Free BOOSTER via `LGBM_BoosterFree'. Does nothing if it was already freed, and returns no
useful value.

Signals `wrong-backend-reference' when BOOSTER is not a `lightgbm-booster' -- a dataset, a
booster built by another backend, or not a handle at all -- before anything is read from it
and before any foreign call, for the reason `free-dataset' above states and by the same
`%check-object-class'.

See `free-dataset' above for why this does not signal `backend-not-open' when BOOSTER's
backend has already been closed, but marks the handle released and `warn's the foreign
memory leaked instead -- the same cleanup-form reasoning applies here, whether the
`unwind-protect' is the caller's own or the one inside `cl-gbdt''s `with-booster'."
  (with-foreign-float-traps-masked
    (%check-object-class booster 'lightgbm-booster "booster"
                         "free-booster's booster argument")
    (if (backend-open-p (handle-backend booster))
        (release-handle booster (lambda (pointer) (%free-booster pointer)))
        (let ((already-released (handle-released-p booster)))
          (release-handle booster (lambda (pointer) (declare (ignore pointer))))
          (unless already-released
            (warn "Freeing a LightGBM booster after its backend was closed: the foreign ~
                   booster was not freed and its memory is leaked."))))))

;;; ---------------------------------------------------------------------------
;;; Inference

(defun %prediction-shape (kind result element-count nrow booster-pointer)
  "Return the shape `predict' states for its KIND result -- a list of integers in
`array-dimensions' order -- or NIL where this backend states none.

RESULT is the array `predict' is about to return, ELEMENT-COUNT the count
`LGBM_BoosterCalcNumPredict' gave for it, NROW the row count whichever entry point just ran was
handed, and BOOSTER-POINTER the booster it ran against.

No SHAPE is read back from the library here, because there is no shape to read: this backend's
prediction entry points report an element count and no axes at all, where XGBoost's write an
`out_shape'/`out_dim' pair `cl-gbdt/src/xgboost/api''s `predict' hands back verbatim. Every
value below is DERIVED, and each KIND gets only what can be derived from what is known.

That is not to say this function makes no foreign call. The `:contrib' arm calls
`%booster-num-features' -- `LGBM_BoosterGetNumFeature' -- for the feature count it derives a
width from. That is the one library call this function makes, and it runs inside `predict''s
`with-foreign-float-traps-masked' body wrap, which is the whole of this function's trap
protection: `%prediction-shape' is `%'-prefixed and named by no `:export' clause, so
`tools/ci/check-float-traps.lisp' does not police it -- that check holds `defun's the backend's
PUBLIC package exports, read from the second `define-package' in `src/lightgbm/all.lisp'. So the
constraint any future change has to preserve is stated here and nowhere else: a second library
call added below, or a caller reaching this function from outside an already-masked dynamic
extent, must establish the mask itself, and nothing in CI will notice if it does not.

What each KIND gets:

  `:normal', `:raw'  RESULT's own `array-dimensions'. One column per output group is the whole
                     of what these hold and the array already says so, so there is nothing
                     further to state.
  `:contrib'         `contrib-shape''s (NROW classes width), width being one contribution per
                     feature plus the bias and classes what is left of ELEMENT-COUNT once the
                     other two divide out. That function answers NIL, never signalling and never
                     guessing, for any of the FOUR cases its own docstring enumerates: NROW zero
                     or negative, a negative feature count, a division leaving a remainder, and
                     an exact division whose quotient is zero. The last two are its way of
                     saying the layout is not the one this arithmetic describes; the first two,
                     that the inputs never described a layout at all. Read that docstring rather
                     than this summary -- the contract is the four cases, not just the division.
                     The CLASS-MAJOR ordering the shape implies does not follow from
                     the count, which divides identically with the last two axes swapped: it is
                     a measured claim, held by
                     `lightgbm-s-derived-contrib-shape-is-the-one-the-numbers-support' in
                     tests/functional/prediction-shape.lisp, whose feature-major arm is the
                     control that makes it a measurement rather than a restatement.
  `:leaf-index'      NIL. ELEMENT-COUNT divides by iterations and output groups much as
                     `:contrib''s divides by output groups and width, but this project has no
                     property that tells the resulting orderings apart -- a leaf index is an
                     opaque identifier, summing to nothing and agreeing with nothing -- and an
                     ordering asserted without one is a guess. NIL means what it means
                     everywhere `predict''s second value appears: no shape is stated. It is not
                     an error and nothing signals.

NROW rather than RESULT's first dimension, though the two are equal, because it is the count
the entry point that just ran was handed and the one ELEMENT-COUNT was computed against --
which for a `csr-matrix' is `csr-matrix-num-rows' and not any dimension of MATRIX."
  (ecase kind
    ((:normal :raw) (array-dimensions result))
    (:contrib (contrib-shape element-count nrow (%booster-num-features booster-pointer)))
    (:leaf-index nil)))

(defun predict (booster matrix &key (kind :normal) num-iteration)
  "Predict on MATRIX with BOOSTER, returning two values: the result array and the SHAPE this
backend states for it.

MATRIX is a dense matrix -- predicted through `LGBM_BoosterPredictForMat' -- or a `csr-matrix',
through `LGBM_BoosterPredictForCSR'. KIND is `:normal', `:raw', `:leaf-index' or `:contrib',
mapped onto LightGBM's own `C_API_PREDICT_*' constant by `%predict-type', which signals for
anything else. Predictions start from iteration 0; nothing here exposes a start-iteration
override.

NUM-ITERATION is a positive integer, or NIL for every iteration -- which LightGBM spells as 0,
and `%resolve-num-iteration' is what writes it that way. :BEST is REFUSED, with
`unsupported-argument' naming this backend and \"predict's :num-iteration\": only `train' writes
a booster's `best-iteration', and a booster built by `create-booster' has none, so at this layer
the keyword would name an empty slot. `%reject-best-num-iteration' is what refuses it, and its
own docstring measures why the refusal has to be explicit -- `%resolve-num-iteration' is
`(or num-iteration 0)', so :BEST would otherwise reach a foreign call expecting an integer as
uninterpreted data and come back a raw CFFI `type-error' rather than a `cl-gbdt' condition. The
refusal is invisible to Layer 2: `cl-gbdt/src/lightgbm/protocol''s `predict' method resolves
:BEST first -- by `%resolve-best-num-iteration', which signals `unsupported-argument' when the
booster has no best iteration to resolve it against -- and calls this with the integer that
resolution produced, so the keyword itself never arrives from there.

Signals `capability-unavailable' naming `:sparse-input' when MATRIX is a `csr-matrix' and that
capability reads false -- see `%check-sparse-input' above, which checks it before any foreign
call. Everything else means exactly what it means for a dense matrix: both entry points take
the same PREDICT-TYPE, the same START-ITERATION/NUM-ITERATION pair and the same parameter
string, and both fill the same buffer in the same row-major order, so KIND and NUM-ITERATION
are honoured identically on either path -- all four KINDs included, unlike
`cl-gbdt/src/xgboost/api''s `predict', whose sparse entry point is XGBoost's inplace
prediction and covers only two of them. A `csr-matrix' whose NUM-COLUMNS is not BOOSTER's own
feature count is LightGBM's own mistake to catch, and it does, with a clean nonzero return this
reports as `foreign-call-error' (\"The number of features in data (N) is not the same as it was
in training data (M).\"); nothing here pre-empts that check.

Signals `wrong-backend-reference' when BOOSTER is not a booster built by this backend -- a
dataset, an XGBoost booster, or not a handle at all. This function dispatches on nothing, so
`%check-lightgbm-booster' is the only thing between such a handle's pointer and
`LGBM_BoosterPredictForMat' or `LGBM_BoosterPredictForCSR'. In the measurement that found this
defect, an `xgboost-booster' with the check absent produced an SBCL corruption warning and
`Signal 7' -- a bus error, which kills the process rather than signalling anything a caller
could handle; in another it returned without faulting at all. See `%check-object-class' above
on why the spread, and not either outcome, is the argument for checking in Lisp.

Signals `released-handle-error' for a freed BOOSTER, and `backend-not-open' when its backend
has since been closed -- both from the `handle-live-pointer' inside `%check-lightgbm-booster',
which is read before anything is allocated, and before NUM-ITERATION is examined, so a freed
booster handed :BEST is reported as freed rather than as having no best iteration.

The output buffer's element count comes from `LGBM_BoosterCalcNumPredict', not from the row
count alone: the row count is only correct for a single-class objective. That count is read the
same way for either matrix kind -- it depends on BOOSTER, the row count, KIND and
NUM-ITERATION, and on nothing about how the rows are laid out. The second array dimension is
that count divided by the row count, guarded by `%predict-ncol'. Whichever entry point ran also
writes its own element count back through OUT-LEN; this is asserted equal to
`LGBM_BoosterCalcNumPredict''s count rather than trusted silently, since the buffer was sized
from the latter and a mismatch would mean either an under-filled result or a write past the
allocated buffer going unnoticed.

No prediction call here ever states the result's SHAPE, and that count is the whole of what one
reports bearing on it: `LGBM_BoosterCalcNumPredict' returns a number and nothing else, and
neither entry point reports axes the way XGBoost's `out_shape'/`out_dim' pair does -- so the
SECOND value is DERIVED rather than reported. `%prediction-shape' above is where that happens,
from the element count, the row count and BOOSTER's own feature count -- the last read by a
further library call, `LGBM_BoosterGetNumFeature', which runs inside this function's own
`with-foreign-float-traps-masked' body wrap like every other call it makes. It states a shape
only for the KINDs those three determine one for: `:normal' and `:raw' get the result array's
own dimensions, `:contrib' the three axes `contrib-shape' divides the count into (NIL for any
of the four cases that function's own docstring enumerates), and `:leaf-index' NIL. This
backend declares `:prediction-shape' in `*provided-capabilities*' to say the mechanism is here;
nothing re-checks that declaration, there being no argument to refuse. The first value is
untouched by all of it -- same dimensions, same elements, every KIND, either entry point.

Deliberately does not scan the result for NaN or infinity -- see
`cl-gbdt/src/xgboost/api''s `predict' for the identical reasoning, which applies here
unchanged: `with-foreign-float-traps-masked' restores the C calling convention around this
call, it does not and should not decide what counts as a valid model output."
  (with-foreign-float-traps-masked
    ;; POINTER is bound first, and a `let' evaluates its init forms left to right, so a
    ;; wrong-kind or freed BOOSTER is refused by `%check-lightgbm-booster' before
    ;; `%reject-best-num-iteration' can report a :BEST it would also have refused -- the same
    ;; precedence the `predict' method above this layer keeps between a bad handle and its own
    ;; argument checks.
    (let ((pointer (%check-lightgbm-booster booster "predict's booster argument"))
          (predict-type (%predict-type kind))
          (iteration-count (%resolve-num-iteration
                            (%reject-best-num-iteration booster num-iteration
                                                        "predict's :num-iteration"))))
      ;; The buffer sizing, the OUT-LEN check and the copy-out are identical for both entry
      ;; points and live here once; CALL is the only thing that differs between them, which
      ;; is exactly how much of this function a `csr-matrix' changes.
      (flet ((predict-into (nrow function-name call)
               (let* ((element-count
                        (%calc-num-predict pointer nrow predict-type 0 iteration-count))
                      (ncol-result (%predict-ncol element-count nrow))
                      (result (make-array (list nrow ncol-result)
                                          :element-type 'double-float)))
                 (cffi:with-foreign-string (parameter-cstring "")
                   (cffi:with-foreign-objects ((out-len :int64)
                                               (buffer :double element-count))
                     (funcall call parameter-cstring out-len buffer)
                     (assert (= element-count (cffi:mem-ref out-len :int64)) ()
                             "~A wrote ~D elements, expected ~D from ~
                              LGBM_BoosterCalcNumPredict"
                             function-name (cffi:mem-ref out-len :int64) element-count)
                     (dotimes (row nrow)
                       (dotimes (col ncol-result)
                         (setf (aref result row col)
                               (cffi:mem-aref buffer :double
                                              (+ (* row ncol-result) col)))))))
                 ;; The second value. Derived from ELEMENT-COUNT and NROW, both of which this
                 ;; function already held for its buffer sizing, plus BOOSTER's own feature
                 ;; count -- nothing about how MATRIX is laid out enters into it, which is why
                 ;; one call here serves both entry points. See `%prediction-shape' above.
                 (values result (%prediction-shape kind result element-count nrow pointer)))))
        (if (typep matrix 'csr-matrix)
            (progn
              (%check-sparse-input (handle-backend booster))
              (predict-into (csr-matrix-num-rows matrix) "LGBM_BoosterPredictForCSR"
                            (lambda (parameter-cstring out-len buffer)
                              (%predict-for-csr pointer
                                                (csr-matrix-indptr matrix)
                                                (csr-matrix-indices matrix)
                                                (csr-matrix-values matrix)
                                                (csr-matrix-num-columns matrix)
                                                predict-type iteration-count
                                                parameter-cstring out-len buffer))))
            (with-foreign-matrix (data-pointer nrow ncol element-type) matrix
              (predict-into nrow "LGBM_BoosterPredictForMat"
                            (lambda (parameter-cstring out-len buffer)
                              (%predict-for-mat pointer data-pointer
                                                (%data-type element-type) nrow ncol
                                                predict-type iteration-count
                                                parameter-cstring out-len buffer)))))))))

;;; ---------------------------------------------------------------------------
;;; Persistence
;;;
;;; `save-model' and `model-to-string' below repeat the same three-step shape: a class guard
;;; on BOOSTER, then `%reject-best-num-iteration', then the foreign call. That repetition is
;;; load-bearing, not an oversight to fold into one helper. `tools/ci/check-layer-1-guards.lisp'
;;; requires the guard to be the very FIRST form each body evaluates, a `%CHECK-'-prefixed call
;;; on BOOSTER itself -- and neither a shared macro nor a body-taking function wrapping that
;;; sequence is a shape its walk recognises, so factoring the duplication out would report both
;;; entry points unguarded rather than remove it.

(defun save-model (booster path &key num-iteration)
  "Save BOOSTER's model to PATH via `LGBM_BoosterSaveModel', and return PATH.

NUM-ITERATION is a positive integer limiting how many trees are written, or NIL for all of
them -- which LightGBM spells as 0, and `%resolve-num-iteration' is what writes it that way.
:BEST is REFUSED, with `unsupported-argument' naming this backend and \"save-model's
:num-iteration\": only `train' writes a booster's `best-iteration', so at this layer the
keyword would name an empty slot. The refusal is invisible to Layer 2, whose method resolves
:BEST first and calls this with the integer that resolution produced.

Signals `wrong-backend-reference' when BOOSTER is not a booster built by this backend -- a
dataset, an XGBoost booster, or not a handle at all. This function dispatches on nothing, so
`%check-lightgbm-booster' is the only thing between such a handle's pointer and
`LGBM_BoosterSaveModel'; see `%check-object-class' above on what wrong-kind pointers did when
measured against the vendored library. It also signals `released-handle-error' for a freed
BOOSTER and `backend-not-open' for a closed backend, both from the `handle-live-pointer'
inside that check, which runs before NUM-ITERATION is examined."
  (with-foreign-float-traps-masked
    (let ((pointer (%check-lightgbm-booster booster "save-model's booster argument")))
      (%reject-best-num-iteration booster num-iteration "save-model's :num-iteration")
      (cffi:with-foreign-string (filename (namestring path))
        (%save-model pointer (%resolve-num-iteration num-iteration) filename)))
    path))

(defun load-model (backend path)
  "Load a LightGBM model from PATH via `LGBM_BoosterCreateFromModelfile' and return a new
booster built against BACKEND.

The returned booster has no training set -- see the `booster' class' documentation -- since
PATH names a model, not a dataset. `evaluation' on it therefore reports nothing, and
`update-one-iteration' signals `missing-training-set'.

The raw booster handle exists in C from the moment `LGBM_BoosterCreateFromModelfile' returns,
but `make-handle' does not take ownership of it until it also succeeds; `with-pointer-ownership'
spans exactly that gap, freeing the raw handle on any exit that has not taken ownership rather
than orphaning it.

Signals `wrong-backend-reference' when BACKEND is not a `lightgbm-backend' -- the other
backend's, a handle, or not a backend at all -- checked FIRST, ahead of `%check-backend-open',
which asks only whether the object is open and answers that truthfully for the wrong library.
Signals `backend-not-open' when BACKEND is closed, and `foreign-call-error' when the library
reports success but returns a null handle."
  (with-foreign-float-traps-masked
    (%check-object-class backend 'lightgbm-backend "backend" "load-model's backend argument")
    (%check-backend-open backend)
    (let ((booster-pointer
            (cffi:with-foreign-string (filename (namestring path))
              (cffi:with-foreign-objects ((out-num-iterations :int) (out :pointer))
                (%create-booster-from-modelfile filename out-num-iterations out)
                (cffi:mem-ref out :pointer)))))
      (when (cffi:null-pointer-p booster-pointer)
        (error 'foreign-call-error
               :function-name "LGBM_BoosterCreateFromModelfile"
               :code 0
               :message "reported success but returned a null booster handle"))
      (with-pointer-ownership (booster-pointer #'%free-booster-unchecked take-ownership)
        (take-ownership 'lightgbm-booster backend :booster)))))

(defun model-to-string (booster &key num-iteration)
  "Return BOOSTER's model as a string via `LGBM_BoosterSaveModelToString'.

NUM-ITERATION means what it means for `save-model' above, :BEST refused on the same terms and
by the same call. The text this returns is the text `save-model' writes, so it can be written
to a file and handed back to `load-model'.

Signals `wrong-backend-reference', `released-handle-error' and `backend-not-open' exactly as
`save-model' does, and for the same reason: this function dispatches on nothing."
  (with-foreign-float-traps-masked
    (let ((pointer (%check-lightgbm-booster booster "model-to-string's booster argument")))
      (%reject-best-num-iteration booster num-iteration "model-to-string's :num-iteration")
      (%save-model-to-string pointer (%resolve-num-iteration num-iteration)))))

;;; ---------------------------------------------------------------------------
;;; Feature importance

(defun feature-importance (booster &key (kind :split) num-iteration)
  "Return BOOSTER's per-feature importances via `LGBM_BoosterFeatureImportance', as a fresh
`(simple-array double-float (*))' with one entry per feature, indexed by column.

The width comes from `LGBM_BoosterGetNumFeature', which answers whether BOOSTER came from
`create-booster' or from `load-model' -- unlike a booster's training set, which `load-model'
leaves unbound. A feature never used in a split is reported as zero rather than omitted.

KIND is `:split' or `:gain', mapped onto LightGBM's own `C_API_FEATURE_IMPORTANCE_*' constant
by `%feature-importance-type', which signals for anything else. NUM-ITERATION is a positive
integer or NIL for every iteration; :BEST is refused by `%reject-best-num-iteration', as it is
in `save-model' and `predict' and for the same reason. KIND is checked BEFORE NUM-ITERATION,
preserving the order the Layer 2 method's own `let*' established.

Signals `wrong-backend-reference' when BOOSTER is not a booster built by this backend, and
`released-handle-error' or `backend-not-open' from the `handle-live-pointer' inside that
check, which runs before anything else is read."
  (with-foreign-float-traps-masked
    (let* ((pointer (%check-lightgbm-booster booster "feature-importance's booster argument"))
           (importance-type (%feature-importance-type kind))
           (count (%booster-num-features pointer))
           (result (make-array count :element-type 'double-float))
           (resolved (%reject-best-num-iteration booster num-iteration
                                                  "feature-importance's :num-iteration")))
      (cffi:with-foreign-object (buffer :double count)
        (%feature-importance pointer (%resolve-num-iteration resolved) importance-type buffer)
        (dotimes (index count)
          (setf (aref result index) (cffi:mem-aref buffer :double index))))
      result)))

;;; ---------------------------------------------------------------------------
;;; Evaluation

(defun evaluation (booster)
  "Return BOOSTER's evaluation metrics as two values: a list of (DATASET-INDEX METRIC-NAME
VALUE) entries, and a plist stating where the values came from.

Reads one `LGBM_BoosterGetEval' result per dataset BOOSTER retains, in the order they were
attached, through `%read-evaluation' -- the same pointer-level reader `train''s per-iteration
recording loop uses, so the two can never disagree. DATASET-INDEX is 0 for the training set
and 1 upward for each validation set, which is `LGBM_BoosterGetEval''s own `data_idx'; nothing
here renumbers anything. The datasets are counted from BOOSTER's own retained handles rather
than asked of the library, which has no entry point reporting how many are attached. A
`load-model' booster retains none, and is the case the count is 0 for.

The second value is always `(:value-source :library-doubles)': these are
`LGBM_BoosterGetEval''s own doubles, returned unmodified. Unlike `cl-gbdt/src/xgboost/api''s
`evaluation', nothing here parses text, so there is no :RAW to keep and no VALUE is ever NIL.

The kind check runs FIRST and `%check-booster-datasets-live' second, which is the opposite of
the order the Layer 2 method used. It has to be: `%check-booster-datasets-live' reads
`booster-training-set' off whatever it is handed, so a DATASET reaching it produced a bare
CLOS `no-applicable-method' error instead of a typed condition -- the same defect
`update-one-iteration' was measured to have and fixed the same way. That check is still owed
before any foreign call, `LGBM_BoosterGetEval' evaluating each attached validation set through
metric objects built over that dataset's own arrays, none of which `LGBM_DatasetFree' clears
from the booster."
  (with-foreign-float-traps-masked
    (let ((pointer (%check-lightgbm-booster booster "evaluation's booster argument")))
      (%check-booster-datasets-live booster)
      (let ((dataset-count (if (booster-training-set booster)
                               (1+ (length (booster-validation-sets booster)))
                               0)))
        (values (%read-evaluation pointer dataset-count)
                (list :value-source :library-doubles))))))

;;; ---------------------------------------------------------------------------
;;; Dataset metadata

(defun dataset-num-rows (dataset)
  "Return DATASET's row count, read via `LGBM_DatasetGetNumData'.

Signals `wrong-backend-reference' when DATASET is not a `lightgbm-dataset' -- a booster, the
other backend's dataset, or not a handle at all. This function dispatches on nothing, so
`%check-object-class' is the only thing between a wrong-kind pointer and the foreign call.
Unlike the frees, which use that same check and then tolerate a released handle, this requires
a LIVE one and reads it through `handle-live-pointer' immediately after: `released-handle-error'
for a freed DATASET, `backend-not-open' for a closed backend."
  (with-foreign-float-traps-masked
    (%check-object-class dataset 'lightgbm-dataset "dataset"
                         "dataset-num-rows's dataset argument")
    (%dataset-num-rows (handle-live-pointer dataset))))

(defun dataset-num-features (dataset)
  "Return DATASET's feature count, read via `LGBM_DatasetGetNumFeature'.

Signals what `dataset-num-rows' above signals, on the same terms and through the same two
calls."
  (with-foreign-float-traps-masked
    (%check-object-class dataset 'lightgbm-dataset "dataset"
                         "dataset-num-features's dataset argument")
    (%dataset-num-features (handle-live-pointer dataset))))
