# Data and prediction

How `make-dataset` and `predict` handle sparse input, missing values, categorical features, and the shape of a prediction result.

## Sparse input: CSR matrices

`make-csr-matrix` builds a `csr-matrix`, the one sparse form this API accepts. `make-dataset`
and `predict` each take one wherever they take a dense matrix -- neither generic's lambda list
changed to allow it -- and the dataset a `csr-matrix` builds is an ordinary dataset that
nothing downstream, `train` included, can distinguish from a densely-built one.

`INDPTR`, `INDICES` and `VALUES` may each be **any sequence**, and are validated and coerced
once, at construction: `INDPTR` and `INDICES` to `(simple-array (signed-byte 32) (*))`,
`VALUES` to `(simple-array double-float (*))`. A backend method therefore only has to pin
what the struct already holds. The five slots are **read-only**: `csr-matrix-indptr` and its
four siblings are readers with no `setf` expander, so that construction-time validation
cannot be undone afterwards. **`NUM-COLUMNS` is required and never inferred** from the
largest index `INDICES` happens to hold: a matrix's declared width and its largest stored
index are different facts -- the trailing columns can legitimately hold nothing at all, and
the stored indices cannot tell that apart from a matrix that simply is not that wide. Only
the caller knows the first. `NUM-ROWS` is not a slot; it is `(1- (length indptr))`, which
`csr-matrix-num-rows` returns, so there is no second copy of the row count to keep in sync.

A malformed matrix signals `dimension-mismatch`, and a value that cannot be coerced signals
`unsupported-element-type` -- both from `make-csr-matrix` itself, next to the mistake, rather
than from a foreign call several frames later:

```lisp
(ql:quickload :cl-gbdt :silent t)

;; Four rows, four columns. Row 2 stores nothing at all -- a repeated INDPTR entry is an
;; empty row, which is legal. INDPTR, INDICES and VALUES may each be any sequence.
(let ((csr (cl-gbdt:make-csr-matrix :indptr '(0 2 3 3 5)
                                    :indices '(0 3 1 0 2)
                                    :values '(1.0 2.0 3.0 4.0 5.0)
                                    :num-columns 4)))
  (format t "indptr:      ~S~%  ~S~%" (cl-gbdt:csr-matrix-indptr csr)
          (type-of (cl-gbdt:csr-matrix-indptr csr)))
  (format t "indices:     ~S~%" (cl-gbdt:csr-matrix-indices csr))
  (format t "values:      ~S~%  ~S~%" (cl-gbdt:csr-matrix-values csr)
          (type-of (cl-gbdt:csr-matrix-values csr)))
  (format t "num-columns: ~S~%" (cl-gbdt:csr-matrix-num-columns csr))
  (format t "num-rows:    ~S~%" (cl-gbdt:csr-matrix-num-rows csr)))

(dolist (bad (list (list :indptr '(0 2) :indices '(0 3) :values '(1.0) :num-columns 4)
                   (list :indptr '(0 1) :indices '(9) :values '(1.0) :num-columns 4)
                   (list :indptr '(0 1) :indices '(0) :values '("x") :num-columns 4)))
  (handler-case (apply #'cl-gbdt:make-csr-matrix bad)
    (error (c) (format t "SIGNALED ~A~%  ~A~%" (type-of c) c))))
```

Output:

```
indptr:      #(0 2 3 3 5)
  (SIMPLE-ARRAY (SIGNED-BYTE 32) (5))
indices:     #(0 3 1 0 2)
values:      #(1.0d0 2.0d0 3.0d0 4.0d0 5.0d0)
  (SIMPLE-ARRAY DOUBLE-FLOAT (5))
num-columns: 4
num-rows:    4
SIGNALED DIMENSION-MISMATCH
  Dimension mismatch. Expected: INDICES and VALUES to have the same length, got: (2
                                                                                  1)
SIGNALED DIMENSION-MISMATCH
  Dimension mismatch. Expected: a column index in [0, 4), got: 9
SIGNALED UNSUPPORTED-ELEMENT-TYPE
  Element type (SIMPLE-ARRAY CHARACTER (1)) is not supported. Use DOUBLE-FLOAT or SINGLE-FLOAT.
```

Sparse input is a capability, `:sparse-input`, true on both vendored backends. Both
`make-dataset` and `predict` re-check it for themselves rather than trusting the caller to
have asked first, and signal `capability-unavailable` when it is false -- never a silent
conversion to a dense matrix, exactly as [the capability
model](backends.md#asking-a-backend-what-it-can-do) requires everywhere else. The dataset's
feature
count is the declared `NUM-COLUMNS`; a `NUM-COLUMNS` that disagrees with the booster's own
feature count at prediction time is the library's own refusal to report, reaching the caller
as `foreign-call-error` in that library's words rather than as a check invented here.

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

(defparameter *dense*
  (make-array '(8 2) :element-type 'double-float
                      :initial-contents '((0.0d0 0.0d0) (0.0d0 1.0d0)
                                           (0.0d0 2.0d0) (0.0d0 3.0d0)
                                           (5.0d0 0.0d0) (5.0d0 1.0d0)
                                           (5.0d0 2.0d0) (5.0d0 3.0d0))))
(defparameter *label*
  (make-array 8 :element-type 'single-float
                 :initial-contents '(0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0)))

;; The same eight rows as CSR, every element stored explicitly -- zeros included. See
;; "An absent entry is not a zero" below for why dropping them is not the same matrix.
(defparameter *sparse*
  (cl-gbdt:make-csr-matrix
   :indptr '(0 2 4 6 8 10 12 14 16)
   :indices '(0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1)
   :values '(0 0 0 1 0 2 0 3 5 0 5 1 5 2 5 3)
   :num-columns 2))

(defun show-sparse (name backend dataset-parameters booster-parameters)
  (format t "~A backend-supports-p :sparse-input => ~S~%"
          name (cl-gbdt:backend-supports-p backend :sparse-input))
  (cl-gbdt:with-dataset (dataset (apply #'cl-gbdt:make-dataset backend *sparse*
                                        :label *label* dataset-parameters))
    (format t "~A dataset from the csr-matrix: rows=~D features=~D~%"
            name (cl-gbdt:dataset-num-rows dataset)
            (cl-gbdt:dataset-num-features dataset))
    (cl-gbdt:with-booster (booster (cl-gbdt:train backend dataset :num-rounds 10
                                                   :parameters booster-parameters))
      (dolist (kind '(:normal :raw :contrib :leaf-index))
        ;; The dense call first, so its success is on the record before the sparse one
        ;; is attempted -- materialising the rows densely is the documented workaround.
        (let ((dense (cl-gbdt:predict booster *dense* :kind kind)))
          (handler-case
              (format t "~A ~S: dense ok; sparse equals dense => ~S~%" name kind
                      (equalp (cl-gbdt:predict booster *sparse* :kind kind) dense))
            ;; XGBoost's message carries a multi-line stack trace; line 1 is the refusal.
            (error (c) (let ((text (princ-to-string c)))
                         (format t "~A ~S: dense ok; sparse SIGNALED ~A~%  ~A~%" name kind
                                 (type-of c)
                                 (subseq text 0 (position #\Newline text)))))))))))

(let ((lgbm (cl-gbdt:open-backend :lightgbm))
      (xgb (cl-gbdt:open-backend :xgboost)))
  (show-sparse "LightGBM" lgbm
               '(:parameters (:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1))
               '(:objective "binary" :num-leaves 2 :min-data-in-leaf 1 :min-data-in-bin 1
                 :verbose -1))
  (show-sparse "XGBoost " xgb '()
               '(:objective "binary:logistic" :max-depth 2 :verbosity 0))
  (cl-gbdt:close-backend lgbm)
  (cl-gbdt:close-backend xgb))
```

Output:

```
LightGBM backend-supports-p :sparse-input => T
LightGBM dataset from the csr-matrix: rows=8 features=2
LightGBM :NORMAL: dense ok; sparse equals dense => T
LightGBM :RAW: dense ok; sparse equals dense => T
LightGBM :CONTRIB: dense ok; sparse equals dense => T
LightGBM :LEAF-INDEX: dense ok; sparse equals dense => T
XGBoost  backend-supports-p :sparse-input => T
XGBoost  dataset from the csr-matrix: rows=8 features=2
XGBoost  :NORMAL: dense ok; sparse equals dense => T
XGBoost  :RAW: dense ok; sparse equals dense => T
XGBoost  :CONTRIB: dense ok; sparse SIGNALED FOREIGN-CALL-ERROR
  XGBoosterPredictFromCSR returned -1: [17:33:19] /__w/xgboost/xgboost/src/learner.cc:1264: Unsupported prediction type:2
XGBoost  :LEAF-INDEX: dense ok; sparse SIGNALED FOREIGN-CALL-ERROR
  XGBoosterPredictFromCSR returned -1: [17:33:19] /__w/xgboost/xgboost/src/learner.cc:1264: Unsupported prediction type:6
```

The bracketed time in XGBoost's two messages is XGBoost's own wall-clock stamp, so those two
lines are the only part of this output that differs from one run to the next.

### `predict`'s KIND on a `csr-matrix`: XGBoost serves two of the four

`make-dataset` takes a `csr-matrix` identically on both backends. `predict` does not:

| `predict`'s KIND | dense, both backends | `csr-matrix`, LightGBM | `csr-matrix`, XGBoost |
|---|---|---|---|
| `:normal` | works | works | works |
| `:raw` | works | works | works |
| `:contrib` | works | works | **`foreign-call-error`** |
| `:leaf-index` | works | works | **`foreign-call-error`** |

`XGBoosterPredictFromCSR` is not the CSR spelling of `XGBoosterPredictFromDMatrix`: the
vendored header (`ffi-spec/xgboost/include/xgboost/c_api.h`) documents it as *inplace
prediction from CPU CSR matrix*, a different code path, and `learner.cc` refuses the prediction
type codes `:contrib` and `:leaf-index` map to -- `2` and `6`, the two numbers the messages
above name -- while accepting `:normal`'s `0` and `:raw`'s `1`. Those refusals reach the caller
as `foreign-call-error` naming the call that failed. Nothing here emulates around it: routing
those two KINDs through a transient DMatrix instead would mean this wrapper, not the library,
deciding which C entry point a KIND gets, and would leave the very symbol `:sparse-input`
declares for prediction unused. LightGBM's `LGBM_BoosterPredictForCSR` has no such restriction
and serves all four, with the same values the dense path produces.

**The workaround is to materialise the rows as a dense matrix** -- a 2D `double-float` or
`single-float` array, or a `foreign-matrix` -- and predict on that, which is what the block
above does for every KIND before trying the sparse call. Note what the workaround is *not*:
`predict`'s MATRIX argument accepts a 2D array, a `foreign-matrix` or a `csr-matrix`, and
**a dataset is not one of them**, so building the rows into a dataset with `make-dataset`
leads to no prediction at all. Materialising is a real cost -- avoiding it is the reason to
pass a `csr-matrix` in the first place -- and this API charges it rather than hiding it.

### An absent entry is not a zero, and the two libraries disagree about it

An entry a `csr-matrix` does not store is *absent*, and the two libraries read absence
differently: **LightGBM reads an absent entry as `0.0`** while its own `zero_as_missing` flag
is off, which is the default -- that flag *can* change what LightGBM reads, measured below in
[LightGBM's `zero_as_missing`, measured](#lightgbms-zero_as_missing-measured). **XGBoost reads
one as missing**, and no key this project has found changes that -- measured as none found,
not established that none exists. This matters because dropping zeros is exactly what a CSR
conversion normally does, and the disagreement changes results silently rather than erroring:

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

;; *dense*, *label* and *sparse* as defined in the previous block.
;; The same eight rows with the zeros dropped -- the conversion a sparse format normally
;; performs. Row 0 is (0.0 0.0) and now stores nothing at all.
(defparameter *zeros-dropped*
  (cl-gbdt:make-csr-matrix
   :indptr '(0 0 1 2 3 4 6 8 10)
   :indices '(1 1 1 0 0 1 0 1 0 1)
   :values '(1 2 3 5 5 1 5 2 5 3)
   :num-columns 2))

(defun first-column (predictions)
  (loop :for row :below (array-dimension predictions 0)
        :collect (aref predictions row 0)))

(defun compare (name backend dataset-parameters booster-parameters)
  (flet ((train-on (matrix)
           (cl-gbdt:with-dataset (dataset (apply #'cl-gbdt:make-dataset backend matrix
                                                 :label *label* dataset-parameters))
             (cl-gbdt:with-booster (booster (cl-gbdt:train backend dataset :num-rounds 10
                                                            :parameters booster-parameters))
               (first-column (cl-gbdt:predict booster *dense*))))))
    (let ((stored (train-on *sparse*))
          (dropped (train-on *zeros-dropped*)))
      (format t "~A every element stored: ~{~,4F~^ ~}~%" name stored)
      (format t "~A zeros dropped:        ~{~,4F~^ ~}~%" name dropped)
      (format t "~A the two agree: ~S~%" name (equal stored dropped)))))

(let ((lgbm (cl-gbdt:open-backend :lightgbm))
      (xgb (cl-gbdt:open-backend :xgboost)))
  (compare "LightGBM" lgbm
           '(:parameters (:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1))
           '(:objective "binary" :num-leaves 2 :min-data-in-leaf 1 :min-data-in-bin 1
             :verbose -1))
  (compare "XGBoost " xgb '()
           '(:objective "binary:logistic" :max-depth 2 :verbosity 0))
  (cl-gbdt:close-backend lgbm)
  (cl-gbdt:close-backend xgb))
```

Output:

```
LightGBM every element stored: 0.1793 0.1793 0.1793 0.1793 0.8207 0.8207 0.8207 0.8207
LightGBM zeros dropped:        0.1793 0.1793 0.1793 0.1793 0.8207 0.8207 0.8207 0.8207
LightGBM the two agree: T
XGBoost  every element stored: 0.4256 0.4256 0.4256 0.4256 0.5744 0.5744 0.5744 0.5744
XGBoost  zeros dropped:        0.5744 0.5744 0.5744 0.5744 0.5744 0.5744 0.5744 0.5744
XGBoost  the two agree: NIL
```

Two matrices describing the same eight rows, differing only in whether the zeros are stored.
LightGBM's two boosters agree to the last digit, which is the absent-entry-is-`0.0` reading
demonstrated rather than asserted. XGBoost's do not: dropping the zeros took feature 0 away
from all four class-0 rows, and the run that trained on what was left no longer separates the
two classes at all -- every row comes back with the positive class's value. Nothing signalled;
the numbers simply changed.

So a `csr-matrix` is not a portable compression of a dense matrix. It is portable when every
element is stored -- as `*sparse*` above does, and as the functional suite's `dense-to-csr`
helper does for exactly this reason -- and it means two different things when entries are
omitted. Omit them when *missing* is what you mean and you are on XGBoost, or when `0.0` is
what you mean and you are on LightGBM; store them when the same matrix has to mean the same
thing on both. Both halves are asserted per backend by the functional suite's
`an-omitted-entry-is-zero-to-lightgbm-and-missing-to-xgboost`, on a fixture that also sends
each library a row storing nothing at all.

`make-csr-matrix` takes `:implicit-value`, so a caller can state once what an absent entry
means in their own data and have the wrapper refuse a call that disagrees with it, rather than
silently train the different numbers the comparison above just measured. The declaration is
`NIL` (the default), a real for which `zerop` is true, `:missing`, or `:none` -- the claim that
nothing is absent, every element stored. Both `make-dataset` and `predict` check it, two
separate code paths each gated on its own, and refuse with `unsupported-argument` before any
foreign call when the declaration disagrees with the backend's own reading of absence:

| Declared | LightGBM | XGBoost |
|---|---|---|
| `NIL` (default, undeclared) | accept | accept |
| a real that is `zerop` (stored as `0.0d0`) | accept | **refuse** |
| `:missing` | **refuse** | accept |
| `:none` (nothing is absent) | accept | accept |

`:none` is verified **structurally**, at `make-csr-matrix` construction, rather than merely
counted: a row's element count alone does not establish that nothing is absent, because
`make-csr-matrix` does not reject a column index repeated within a row -- duplicates are legal
CSR that both libraries accept -- so a row storing one column twice and another not at all has
exactly the right length while still leaving one of its columns absent. Declaring `:none` on
such a matrix signals `dimension-mismatch`, not `unsupported-argument`: the declaration itself
is legal and the matrix fails it, a size claim like the ones `make-csr-matrix` already checks.

An undeclared matrix -- `:implicit-value` left `NIL`, the default -- is checked by **nothing**.
Everything measured above still applies to it exactly as it always did.

### LightGBM's `zero_as_missing`, measured

The obvious question about the LightGBM refusal above is LightGBM's own `zero_as_missing`
parameter: if it can make an absent entry mean *missing*, doesn't the refusal turn away a true
claim? Three runs, reusing `*sparse*` and `*zeros-dropped*` from above, settle what the flag
actually does:

```lisp
(ql:quickload :cl-gbdt/lightgbm/unified :silent t)

;; *sparse*, *zeros-dropped* and *label* as defined above.
(defun train-model-string (backend matrix &key dataset-zam booster-zam)
  (let ((dataset-parameters (append '(:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1)
                                     (when dataset-zam '(:zero-as-missing "true"))))
        (booster-parameters (append '(:objective "binary" :num-leaves 2 :min-data-in-leaf 1
                                       :min-data-in-bin 1 :verbose -1)
                                     (when booster-zam '(:zero-as-missing "true")))))
    (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset backend matrix :label *label*
                                                          :parameters dataset-parameters))
      (cl-gbdt:with-booster (booster (cl-gbdt:train backend dataset :num-rounds 10
                                                     :parameters booster-parameters))
        (cl-gbdt:model-to-string booster)))))

(defun recorded-flag (model-string)
  (let ((pos (search "[zero_as_missing:" model-string)))
    (subseq model-string pos (+ pos 19))))

;; The parameter string itself is built correctly regardless of what follows.
(format t "parameter string: ~S~%"
        (cl-gbdt/src/lightgbm/native::%parameter-string '(:zero-as-missing "true" :verbose -1)))

(let ((lgbm (cl-gbdt:open-backend :lightgbm)))
  (let ((plain (train-model-string lgbm *sparse*))
        (dataset-only (train-model-string lgbm *sparse* :dataset-zam t)))
    (format t "1. set on make-dataset's :parameters alone: recorded ~A, byte-identical: ~S~%"
            (recorded-flag dataset-only) (string= plain dataset-only)))
  (let ((plain (train-model-string lgbm *sparse*))
        (both (train-model-string lgbm *sparse* :dataset-zam t :booster-zam t)))
    (format t "2. set on both make-dataset's and train's :parameters: recorded ~A, ~
              byte-identical: ~S~%"
            (recorded-flag both) (string= plain both)))
  (let ((stored (train-model-string lgbm *sparse* :dataset-zam t :booster-zam t))
        (dropped (train-model-string lgbm *zeros-dropped* :dataset-zam t :booster-zam t)))
    (format t "3. with it set in both places, stored vs. zeros-dropped byte-identical: ~S~%"
            (string= stored dropped)))
  (cl-gbdt:close-backend lgbm))
```

Output:

```
parameter string: "zero_as_missing=true verbose=-1"
1. set on make-dataset's :parameters alone: recorded [zero_as_missing: 0, byte-identical: T
2. set on both make-dataset's and train's :parameters: recorded [zero_as_missing: 1, byte-identical: NIL
3. with it set in both places, stored vs. zeros-dropped byte-identical: T
```

Setting `zero_as_missing` on `make-dataset`'s `:parameters` alone does nothing: the model
records `[zero_as_missing: 0]` and is byte-identical to the run without it, even though the
parameter string itself is built correctly, as run 1's own first line confirms. The flag has
to be set on `train`'s `:parameters` too before it bites -- run 2 does that, and the model
both records `[zero_as_missing: 1]` and differs. With it set in both places, run 3 trains
`*sparse*` and `*zeros-dropped*` to **byte-identical** models: an absent entry and a stored
zero become the same thing, exactly what the flag is documented to do.

So `:implicit-value :missing` genuinely **is** a true claim on LightGBM under that
configuration, and this project's refusal turns it away anyway. The reason is not that
LightGBM cannot do it -- that would be false, and a false statement does not belong on the
public surface -- it is that neither call site checking the declaration can see the
configuration that makes it true: `make-dataset` never sees the parameters `train` is later
called with, and `predict` sees neither dataset's nor booster's. A check reading
`make-dataset`'s own `:parameters` would answer confidently and wrongly, since the flag alone
there -- run 1 above -- changes nothing at all. A caller who has genuinely set
`zero_as_missing` on both the dataset and the booster should simply leave `:implicit-value`
undeclared: the declaration is opt-in, and undeclared costs nothing under that configuration
either.

### Why CSR only, and not CSC

XGBoost has `XGDMatrixCreateFromCSC` -- it is bound in `src/xgboost/c-api.lisp` like every
other emitted function -- but there is **no `XGBoosterPredictFromCSC`** anywhere in its C API;
`XGBoosterPredictFromCSR` is the only sparse prediction entry point it offers. Supporting CSC
would therefore put a format into this API that a caller could build a dataset from and then
not predict with: `make-dataset` would take it, `predict` would refuse it, and the refusal
would be a property of one backend rather than of the format. LightGBM does have both
(`LGBM_DatasetCreateFromCSC` and `LGBM_BoosterPredictForCSC`), which would make the gap
backend-specific in a way nothing else in the unified API is. CSR is the format both libraries
can do both halves of, so CSR is the format this API takes.

## Missing values

`make-dataset` and `predict` both take `:missing`, the value in the caller's own matrix that
means *missing* -- the datum a caller wrote in place of one they do not have, such as the
`-999.0` a CSV convention often uses. `:missing` is a `real` or `NIL`; anything else signals
`unsupported-argument` naming `:missing` and the backend. `NIL`, the default on both
operations, is exactly today's behaviour: the wrapper sends `"missing":NaN` unconditionally,
the same as every call made before either operation took this argument at all. This applies
identically whether MATRIX is dense or a `csr-matrix` -- `make-dataset`'s own `:missing`
reaches the same creation-config key either way.

A non-`NIL` `:missing` needs the `:missing-value` capability, answerable through
`backend-supports-p`, and **both operations re-check it themselves** rather than trusting a
caller who asked first. XGBoost declares it -- the sentinel is a key its creation and
prediction config JSONs already read -- and LightGBM does not: its C API has no
missing-value key at all, so `:missing` there signals `capability-unavailable` for *any*
non-`NIL` value, a `NaN` included, even though a `NaN` is in fact what LightGBM's own
ingestion path already treats as missing. A capability whose meaning depended on which value
was passed could not be stated by `backend-supports-p` at all -- it answers about the
backend, and never sees the argument. The capability gate also fires *first*, by design: a
non-`real` `:missing` on LightGBM signals `capability-unavailable`, not
`unsupported-argument` -- the renderer's own type check is never reached there, since only a
backend that passed the gate has a renderer left to reach.

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

(defparameter *mv-matrix*
  (make-array '(8 3) :element-type 'double-float
                      :initial-contents '((0.0d0 1.0d0 2.0d0) (0.0d0 2.0d0 1.0d0)
                                           (0.0d0 1.0d0 2.0d0) (0.0d0 2.0d0 1.0d0)
                                           (5.0d0 1.0d0 2.0d0) (5.0d0 2.0d0 1.0d0)
                                           (5.0d0 1.0d0 2.0d0) (5.0d0 2.0d0 1.0d0))))
(defparameter *mv-label*
  (make-array 8 :element-type 'single-float
                 :initial-contents '(0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0)))
;; Column 0 alone carries the class -- 0.0 for the first four rows, 5.0 for the last four --
;; while columns 1 and 2 repeat the same two values on both halves and carry nothing. Row 7 is
;; positive-class, so punching its column-0 cell takes away the only signal that row has.
(defparameter *mv-sentinel* -999.0d0)

(defun mv-holed (&optional (value *mv-sentinel*) (row 7))
  "*MV-MATRIX*, with ROW's column 0 replaced by VALUE."
  (let ((matrix (make-array '(8 3) :element-type 'double-float)))
    (dotimes (r 8) (dotimes (c 3) (setf (aref matrix r c) (aref *mv-matrix* r c))))
    (setf (aref matrix row 0) value)
    matrix))

(defun mv-quiet-nan ()
  "A quiet double-float NaN, built from its bits so no arithmetic can trap."
  (sb-kernel:make-double-float -524288 0))

(defun mv-train-predict (xgb matrix &key missing)
  "Train an XGBoost booster on MATRIX/*MV-LABEL* and return its row-7 prediction on the
clean *MV-MATRIX*."
  (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset xgb matrix :label *mv-label*
                                                        :missing missing))
    (cl-gbdt:with-booster (booster (cl-gbdt:train xgb dataset :num-rounds 5
                                      :parameters '(:objective "binary:logistic" :max-depth 2
                                                    :eta 0.5 :verbosity 0)))
      (aref (cl-gbdt:predict booster *mv-matrix*) 7 0))))

(let ((lgbm (cl-gbdt:open-backend :lightgbm))
      (xgb (cl-gbdt:open-backend :xgboost)))
  (format t "LightGBM backend-supports-p :missing-value => ~S~%"
          (cl-gbdt:backend-supports-p lgbm :missing-value))
  (format t "XGBoost  backend-supports-p :missing-value => ~S~%"
          (cl-gbdt:backend-supports-p xgb :missing-value))

  ;; LightGBM signals regardless of the value -- even a NaN, which is in fact what its own
  ;; ingestion path already treats as missing -- because its C API has no missing-value key at
  ;; all: the capability gate fires before the value is even looked at.
  (dolist (value (list *mv-sentinel* (mv-quiet-nan)))
    (handler-case (cl-gbdt:make-dataset lgbm (mv-holed value) :label *mv-label* :missing value
                                         :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                                        :verbose -1))
      (error (c) (format t "LightGBM make-dataset :missing ~A SIGNALED ~A: ~A~%"
                          (if (sb-ext:float-nan-p value) "<NaN>" value) (type-of c) c))))

  ;; The gate fires first even for a non-real :missing: LightGBM never reaches the renderer's
  ;; own type check at all, so this is CAPABILITY-UNAVAILABLE too, not UNSUPPORTED-ARGUMENT.
  (handler-case (cl-gbdt:make-dataset lgbm (mv-holed) :label *mv-label* :missing "-999.0"
                                       :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                                      :verbose -1))
    (error (c) (format t "LightGBM make-dataset :missing \"-999.0\" SIGNALED ~A: ~A~%"
                        (type-of c) c)))

  ;; XGBoost provides the capability, so a non-real :missing reaches the renderer's own check
  ;; instead of the capability gate.
  (handler-case (cl-gbdt:make-dataset xgb (mv-holed) :label *mv-label* :missing "-999.0")
    (error (c) (format t "XGBoost  make-dataset :missing \"-999.0\" SIGNALED ~A: ~A~%"
                        (type-of c) c)))

  ;; :missing selects a sentinel VALUE, not a policy: it changes what the model learns.
  (format t "XGBoost  row 7, trained with :missing ~S: ~S~%" *mv-sentinel*
          (mv-train-predict xgb (mv-holed) :missing *mv-sentinel*))
  (format t "XGBoost  row 7, trained with that same cell read literally: ~S~%"
          (mv-train-predict xgb (mv-holed)))

  ;; The wrapper renders the sentinel itself rather than letting the Lisp printer choose: a
  ;; bare `princ' of a double would emit "1.0d-5", and XGBoost's JSON config parser rejects
  ;; that exponent marker outright. A caller may still pass a `d0'-marked double; only what
  ;; reaches the library is affected.
  (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset xgb *mv-matrix* :label *mv-label*
                                                        :missing 1.0d-5))
    (format t "XGBoost  make-dataset :missing 1.0d-5 (a Lisp exponent marker) works: rows=~S~%"
            (cl-gbdt:dataset-num-rows dataset)))

  (cl-gbdt:close-backend lgbm)
  (cl-gbdt:close-backend xgb))
```

Output:

```
LightGBM backend-supports-p :missing-value => NIL
XGBoost  backend-supports-p :missing-value => T
LightGBM make-dataset :missing -999.0d0 SIGNALED CAPABILITY-UNAVAILABLE: LIGHTGBM does not provide :MISSING-VALUE in the library that is loaded.
LightGBM make-dataset :missing <NaN> SIGNALED CAPABILITY-UNAVAILABLE: LIGHTGBM does not provide :MISSING-VALUE in the library that is loaded.
LightGBM make-dataset :missing "-999.0" SIGNALED CAPABILITY-UNAVAILABLE: LIGHTGBM does not provide :MISSING-VALUE in the library that is loaded.
XGBoost  make-dataset :missing "-999.0" SIGNALED UNSUPPORTED-ARGUMENT: :missing is not supported by XGBOOST: the value that means missing must be a real number, or NIL for the backend's own default.
XGBoost  row 7, trained with :missing -999.0d0: 0.622459352016449d0
XGBoost  row 7, trained with that same cell read literally: 0.5d0
XGBoost  make-dataset :missing 1.0d-5 (a Lisp exponent marker) works: rows=8
```

`:missing` selects a sentinel *value*, not a policy. It does not turn missing-value handling
on or off, and it does not make `0.0` mean missing -- LightGBM's `use_missing` and
`zero_as_missing` flags are unchanged by any of this, and stay exactly where they were,
reachable through `make-dataset`'s `:parameters`. The last line of the output above is a
separate, JSON-rendering fact worth knowing on its own: XGBoost's config parser rejects a
Lisp double's own exponent marker outright (`1.0d-5` fails with `json.cc:409: Expecting:
","`, measured against the vendored library) but accepts `1.0e-5`. The wrapper renders
`:missing` itself for exactly this reason, so a caller may still write `1.0d-5` and have it
reach the library correctly.

### `predict`'s own `:missing`

`predict` takes the identical argument, checked against the `:missing-value` capability
**separately** from `make-dataset`'s own check -- the two operations reach two different
config sites, and a backend could in principle gate one and not the other. A dense matrix's
sentinel becomes a key in the transient DMatrix's own *creation* config, the same one
`make-dataset` fills; a `csr-matrix`'s sentinel goes into `XGBoosterPredictFromCSR`'s
*inplace predict* config instead, since that call builds no DMatrix at all -- shown below
with `:kind :normal`, one of the two kinds XGBoost's sparse `predict` serves at all (see
[Sparse input](#sparse-input-csr-matrices) above). Both config sites are demonstrated first,
on one booster trained with no `:missing` anywhere, so nothing about how the model was
trained can account for what changes; the single-precision measurement that follows trains a
second booster, for a reason its own comment explains:

```lisp
(defparameter *mv-narrowing-sentinel* 16777217.0d0
  "Not exactly representable in single-float: 16777217 is 2^24+1, and single-float spacing at
2^24 is 2, so this narrows to 16777216.0.")
(defparameter *mv-shared-float32* 16777216.0d0
  "A different double-float from *MV-NARROWING-SENTINEL*, but the same value once both are
narrowed to single-float.")
(defparameter *mv-own-float32* 16777224.0d0
  "2^24+8, a multiple of single-float's spacing there, so this is exactly representable and is
therefore its own single-float, distinct from *MV-NARROWING-SENTINEL*'s.")

(defun mv-csr (matrix)
  "MATRIX as a `cl-gbdt:csr-matrix' with every element stored explicitly."
  (let* ((rows (array-dimension matrix 0))
         (cols (array-dimension matrix 1))
         (indptr (make-array (1+ rows)))
         (indices (make-array (* rows cols)))
         (values (make-array (* rows cols)))
         (pos 0))
    (dotimes (r rows)
      (setf (aref indptr r) pos)
      (dotimes (c cols)
        (setf (aref indices pos) c)
        (setf (aref values pos) (aref matrix r c))
        (incf pos)))
    (setf (aref indptr rows) pos)
    (cl-gbdt:make-csr-matrix :indptr indptr :indices indices :values values :num-columns cols)))

(let ((xgb (cl-gbdt:open-backend :xgboost)))
  ;; predict's own :missing, re-checked separately from make-dataset's -- trained on the CLEAN
  ;; matrix, so nothing about how the model was trained can account for what follows.
  (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset xgb *mv-matrix* :label *mv-label*))
    (cl-gbdt:with-booster (booster (cl-gbdt:train xgb dataset :num-rounds 5
                                      :parameters '(:objective "binary:logistic" :max-depth 2
                                                    :eta 0.5 :verbosity 0)))
      (format t "predict row 7, :missing ~S on the holed matrix: ~S~%" *mv-sentinel*
              (aref (cl-gbdt:predict booster (mv-holed) :missing *mv-sentinel*) 7 0))
      (format t "predict row 7, that same holed matrix read literally: ~S~%"
              (aref (cl-gbdt:predict booster (mv-holed)) 7 0))
      ;; The other config site: a csr-matrix's :missing reaches XGBoosterPredictFromCSR's own
      ;; inplace predict config rather than a transient DMatrix's creation config.
      (format t "predict on a csr-matrix, :missing ~S: ~S~%" *mv-sentinel*
              (aref (cl-gbdt:predict booster (mv-csr (mv-holed)) :kind :normal
                                     :missing *mv-sentinel*) 7 0))
      (format t "predict on that csr-matrix, the same cell read literally: ~S~%"
              (aref (cl-gbdt:predict booster (mv-csr (mv-holed)) :kind :normal) 7 0))))

  ;; Single precision: XGBoost gives every split a default direction for a value it reads as
  ;; missing. A booster trained on the clean fixture sends both 2^24-sized values the same
  ;; direction it already sends a genuine missing value, so it cannot tell the two readings
  ;; apart below. Training instead with a NaN hole in a NEGATIVE-class row (row 3, unlike row
  ;; 7's positive class) teaches a default direction the two readings do separate on.
  (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset xgb (mv-holed (mv-quiet-nan) 3)
                                                        :label *mv-label*))
    (cl-gbdt:with-booster (booster (cl-gbdt:train xgb dataset :num-rounds 5
                                      :parameters '(:objective "binary:logistic" :max-depth 2
                                                    :eta 0.5 :verbosity 0)))
      (format t "predict row 7, :missing ~S vs a stored ~S (shares its float32): ~S~%"
              *mv-narrowing-sentinel* *mv-shared-float32*
              (aref (cl-gbdt:predict booster (mv-holed *mv-shared-float32*)
                                     :missing *mv-narrowing-sentinel*) 7 0))
      (format t "predict row 7, :missing ~S vs a stored ~S (its own float32): ~S~%"
              *mv-narrowing-sentinel* *mv-own-float32*
              (aref (cl-gbdt:predict booster (mv-holed *mv-own-float32*)
                                     :missing *mv-narrowing-sentinel*) 7 0))
      (format t "predict row 7, a stored NaN, for comparison: ~S~%"
              (aref (cl-gbdt:predict booster (mv-holed (mv-quiet-nan))) 7 0))))

  (cl-gbdt:close-backend xgb))
```

Output:

```
predict row 7, :missing -999.0d0 on the holed matrix: 0.622459352016449d0
predict row 7, that same holed matrix read literally: 0.3775406777858734d0
predict on a csr-matrix, :missing -999.0d0: 0.622459352016449d0
predict on that csr-matrix, the same cell read literally: 0.3775406777858734d0
predict row 7, :missing 1.6777217d7 vs a stored 1.6777216d7 (shares its float32): 0.3775406777858734d0
predict row 7, :missing 1.6777217d7 vs a stored 1.6777224d7 (its own float32): 0.622459352016449d0
predict row 7, a stored NaN, for comparison: 0.3775406777858734d0
```

The last three lines are the single-precision fact: XGBoost compares `:missing` against the
data at **single precision**, whatever the matrix's own element type. `16777217.0d0` is
2^24 + 1, one past single-float's spacing of 2 at that magnitude, so it narrows to
`16777216.0`; a stored `16777216.0d0` -- a genuinely different `double-float` -- shares that
narrowing and so reads as missing, while a stored `16777224.0d0` -- 2^24 + 8, itself exactly
representable in `single-float` -- does not. Two `double-float`s that round to the same
`single-float` therefore both count as missing against a sentinel that narrows to it.
Measured directly at the raw XGBoost level too, over the identical 24-cell fixture, before
either functional test in `tests/functional/missing-value.lisp` existed:
`XGDMatrixNumNonMissing` keeps 22 of the 24 entries when the sentinel matches the
shared-float32 datum, and all 24 when it does not -- the same distinction the predictions
above show, at the level a caller actually observes it.

### Training and prediction sentinels are not tied together

Nothing connects the `:missing` a dataset was built with to the `:missing` a later `predict`
call names. XGBoost does not record a dataset's sentinel on the booster trained from it, and
none is written into a saved model either, so predicting with a *different* sentinel than
training used -- or with none at all -- is a call the library accepts and never reports.
Keeping the two consistent is the caller's own responsibility; nothing here detects that they
are not:

```lisp
;; Nothing ties predict's :missing to the sentinel :missing used at training time. XGBoost does
;; not record a dataset's sentinel on the booster trained from it, and none is written into a
;; saved model either -- so a caller who predicts with a DIFFERENT sentinel than training used,
;; or with none at all, gets no error: just silently different numbers.
(let ((xgb (cl-gbdt:open-backend :xgboost)))
  (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset xgb (mv-holed -999.0d0) :label *mv-label*
                                                        :missing -999.0d0))
    (cl-gbdt:with-booster (booster (cl-gbdt:train xgb dataset :num-rounds 5
                                      :parameters '(:objective "binary:logistic" :max-depth 2
                                                    :eta 0.5 :verbosity 0)))
      (format t "trained with :missing -999.0d0; predict :missing -999.0d0 (matches): ~S~%"
              (aref (cl-gbdt:predict booster (mv-holed -999.0d0) :missing -999.0d0) 7 0))
      (format t "same booster; predict :missing -1.0d0 (a DIFFERENT sentinel; no error): ~S~%"
              (aref (cl-gbdt:predict booster (mv-holed -999.0d0) :missing -1.0d0) 7 0))
      (format t "same booster; predict with no :missing at all (no error either): ~S~%"
              (aref (cl-gbdt:predict booster (mv-holed -999.0d0)) 7 0))))
  (cl-gbdt:close-backend xgb))
```

Output:

```
trained with :missing -999.0d0; predict :missing -999.0d0 (matches): 0.622459352016449d0
same booster; predict :missing -1.0d0 (a DIFFERENT sentinel; no error): 0.3775406777858734d0
same booster; predict with no :missing at all (no error either): 0.3775406777858734d0
```

The booster above was trained with `:missing -999.0d0`. Asking `predict` for the matching
sentinel reads row 7 as missing, exactly as training did. Asking for `-1.0d0` instead -- a
sentinel training never used -- signals nothing at all; row 7 is simply read literally, the
same result omitting `:missing` from `predict` entirely already gives. A caller who trains
with one sentinel and predicts with another, or forgets `:missing` on one side, gets a
booster and a prediction that both ran without complaint, and numbers that silently do not
mean what the caller intended.

## Categorical features

`make-dataset` takes `:categorical-features`, a list of 0-based column indices naming which
columns of the caller's own matrix hold *categories* rather than *quantities* -- so a split on
one of them partitions the category set instead of thresholding an ordinal that has no order.
`NIL`, the default, is exactly today's behaviour: every column is read as a quantity, the same
as every call made before the argument existed.

Each backend renders the list its own way. XGBoost attaches it to the finished DMatrix as the
`"feature_type"` field -- `"c"` for each named column, `"q"` for every other -- through the
same `XGDMatrixSetStrFeatureInfo` call `:feature-names` already uses, under a different key.
LightGBM instead composes a `categorical_feature` entry into the parameter string that builds
the dataset -- see [LightGBM: `categorical_feature` and its four
aliases](#lightgbm-categorical_feature-and-its-four-aliases) below for what that means when a
caller also writes the key by hand.

`predict` takes no such argument at all, on either backend. A booster trained from a dataset
built with categorical columns predicts correctly from a plain matrix regardless -- measured
below -- because the trained trees already carry the category sets they split on. XGBoost in
particular records nothing about which columns were categorical on the booster itself: a model
it saves carries an empty `"feature_types":[]`, the same field a model trained with no
categorical column at all would save.

`:categorical-features` needs the `:categorical-features` capability, answerable through
`backend-supports-p` and true on both vendored backends, and **`make-dataset` re-checks it
itself** rather than trusting a caller who asked first, exactly as [the capability
model](backends.md#asking-a-backend-what-it-can-do) requires everywhere else. It applies
identically
whether the caller's matrix is dense or a `csr-matrix` -- the column count checked against is
the matrix's own, and for a `csr-matrix` that is its **declared** `NUM-COLUMNS`, not the
largest index actually stored. An index that is not an integer, is negative, is at or beyond
the matrix's last column, or was named more than once, signals `unsupported-argument` naming
`:categorical-features` and the backend, from `make-dataset` itself rather than from either
library:

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

;; Six categories in column 0, alternating class by category -- 0, 2 and 4 positive, 1, 3 and 5
;; negative -- four rows apiece, 24 rows in all. No threshold on the ordinal 0..5 separates an
;; alternating pattern; a categorical split choosing the subset {0, 2, 4} does. Column 1 is
;; noise: it alternates independently of the class and carries no signal.
;; Taken from tests/functional/categorical-features.lisp's own fixture and parameters.
(defparameter *num-categories* 6)
(defparameter *rows-per-category* 4)

(defun category-matrix ()
  (let* ((rows (* *num-categories* *rows-per-category*))
         (matrix (make-array (list rows 2) :element-type 'double-float)))
    (dotimes (row rows)
      (setf (aref matrix row 0) (coerce (floor row *rows-per-category*) 'double-float))
      (setf (aref matrix row 1) (coerce (mod row 2) 'double-float)))
    matrix))

(defun category-labels ()
  (let* ((rows (* *num-categories* *rows-per-category*))
         (label-vector (make-array rows :element-type 'single-float)))
    (dotimes (row rows)
      (setf (aref label-vector row)
            (if (evenp (floor row *rows-per-category*)) 1.0 0.0)))
    label-vector))

(defun category-means (predictions)
  (loop :for category :below *num-categories*
        :collect (/ (loop :for row :from (* category *rows-per-category*)
                            :below (* (1+ category) *rows-per-category*)
                          :sum (row-major-aref predictions row))
                    *rows-per-category*)))

(defun demo (name backend dataset-parameters booster-parameters)
  (format t "~A backend-supports-p :categorical-features => ~S~%"
          name (cl-gbdt:backend-supports-p backend :categorical-features))
  (flet ((run (categorical-features)
           (cl-gbdt:with-dataset
               (dataset (apply #'cl-gbdt:make-dataset backend (category-matrix)
                               :label (category-labels)
                               (append (when dataset-parameters
                                         (list :parameters dataset-parameters))
                                       (when categorical-features
                                         (list :categorical-features categorical-features)))))
             (cl-gbdt:with-booster (booster (cl-gbdt:train backend dataset :num-rounds 20
                                                           :parameters booster-parameters))
               (category-means (cl-gbdt:predict booster (category-matrix)))))))
    (format t "~A category means, :categorical-features '(0): ~S~%" name (run '(0)))
    (format t "~A category means, the same matrix read as quantities: ~S~%" name (run nil))))

(let ((lgbm (cl-gbdt:open-backend :lightgbm))
      (xgb (cl-gbdt:open-backend :xgboost)))
  (demo "XGBoost " xgb nil
        '(:objective "binary:logistic" :max-depth 1 :verbosity 0 :min-child-weight 0
          :tree-method "hist"))
  (demo "LightGBM" lgbm '(:min-data-in-leaf 1 :min-data-in-bin 1 :min-data-per-group 1
                           :cat-smooth 0 :cat-l2 0 :verbose -1)
        '(:objective "binary" :max-depth 1 :verbose -1 :min-data-in-leaf 1 :min-data-per-group 1
          :cat-smooth 0 :cat-l2 0))

  ;; predict never takes :categorical-features -- every predict call above already omits it,
  ;; and still routes each row down the trained categorical splits correctly. XGBoost's own
  ;; saved model confirms it records nothing about which columns were categorical:
  (cl-gbdt:with-dataset
      (dataset (cl-gbdt:make-dataset xgb (category-matrix) :label (category-labels)
                                      :categorical-features '(0)))
    (cl-gbdt:with-booster (booster (cl-gbdt:train xgb dataset :num-rounds 5
                                      :parameters '(:objective "binary:logistic" :max-depth 1
                                                    :verbosity 0 :min-child-weight 0
                                                    :tree-method "hist")))
      (let* ((json (cl-gbdt:model-to-string booster))
             (pos (search "\"feature_types\"" json)))
        (format t "XGBoost  saved model's own feature_types field: ~A~%"
                (subseq json pos (+ pos 25))))))

  ;; The same comparison over a csr-matrix, the other form make-dataset accepts -- every element
  ;; stored explicitly, so nothing here is about an absent CSR entry (see Sparse input above).
  (flet ((as-csr (matrix)
           (let* ((rows (array-dimension matrix 0)) (columns (array-dimension matrix 1))
                  (indptr (make-array (1+ rows))) (indices (make-array (* rows columns)))
                  (values (make-array (* rows columns))) (position 0))
             (dotimes (row rows)
               (setf (aref indptr row) position)
               (dotimes (column columns)
                 (setf (aref indices position) column)
                 (setf (aref values position) (aref matrix row column))
                 (incf position)))
             (setf (aref indptr rows) position)
             (cl-gbdt:make-csr-matrix :indptr indptr :indices indices :values values
                                      :num-columns columns))))
    (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset xgb (as-csr (category-matrix))
                                                          :label (category-labels)
                                                          :categorical-features '(0)))
      (cl-gbdt:with-booster (booster (cl-gbdt:train xgb dataset :num-rounds 20
                                        :parameters '(:objective "binary:logistic" :max-depth 1
                                                      :verbosity 0 :min-child-weight 0
                                                      :tree-method "hist")))
        (format t "XGBoost  category means from a csr-matrix dataset, ~
                   :categorical-features '(0): ~S~%"
                (category-means (cl-gbdt:predict booster (category-matrix)))))))

  ;; A bad index signals unsupported-argument naming :categorical-features and the backend,
  ;; from make-dataset itself, before either backend's own creation call is reached.
  (dolist (indices (list '("0") '(-1) '(2) '(0 0)))
    (handler-case
        (cl-gbdt:free-dataset
         (cl-gbdt:make-dataset xgb (category-matrix) :label (category-labels)
                                :categorical-features indices))
      (error (c) (format t "XGBoost  :categorical-features ~S SIGNALED ~A: ~A~%"
                          indices (type-of c) c))))

  ;; A :valid-sets entry built WITHOUT :categorical-features, alongside a training set built
  ;; WITH it, provokes nothing: training succeeds and both entries evaluate the same way.
  (cl-gbdt:with-dataset
      (xgb-train (cl-gbdt:make-dataset xgb (category-matrix) :label (category-labels)
                                        :categorical-features '(0)))
    (cl-gbdt:with-dataset (xgb-valid (cl-gbdt:make-dataset xgb (category-matrix)
                                                            :label (category-labels)))
      (cl-gbdt:with-booster
          (booster (cl-gbdt:train xgb xgb-train :num-rounds 5 :valid-sets (list xgb-valid)
                                   :parameters '(:objective "binary:logistic" :max-depth 1
                                                 :verbosity 0 :min-child-weight 0
                                                 :tree-method "hist" :eval-metric "logloss")))
        (format t "XGBoost  evaluation, train :categorical-features '(0), valid without it: ~S~%"
                (cl-gbdt:evaluation booster)))))
  (cl-gbdt:with-dataset
      (lgbm-train (cl-gbdt:make-dataset lgbm (category-matrix) :label (category-labels)
                                         :categorical-features '(0)
                                         :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                                       :min-data-per-group 1 :cat-smooth 0
                                                       :cat-l2 0 :verbose -1)))
    (cl-gbdt:with-dataset (lgbm-valid (cl-gbdt:make-dataset lgbm (category-matrix)
                                                             :label (category-labels)
                                                             :reference lgbm-train
                                                             :parameters '(:min-data-in-leaf 1
                                                                           :min-data-in-bin 1
                                                                           :min-data-per-group 1
                                                                           :cat-smooth 0 :cat-l2 0
                                                                           :verbose -1)))
      (cl-gbdt:with-booster
          (booster (cl-gbdt:train lgbm lgbm-train :num-rounds 5 :valid-sets (list lgbm-valid)
                                   :parameters '(:objective "binary" :max-depth 1 :verbose -1
                                                 :min-data-in-leaf 1 :min-data-per-group 1
                                                 :cat-smooth 0 :cat-l2 0
                                                 :metric "binary_logloss")))
        (format t "LightGBM evaluation, train :categorical-features '(0), valid without it: ~S~%"
                (cl-gbdt:evaluation booster)))))

  (cl-gbdt:close-backend lgbm)
  (cl-gbdt:close-backend xgb))
```

Output:

```
XGBoost  backend-supports-p :categorical-features => T
XGBoost  category means, :categorical-features '(0): (0.9739266037940979d0
                                                      0.026073377579450607d0
                                                      0.9739266037940979d0
                                                      0.026073377579450607d0
                                                      0.9739266037940979d0
                                                      0.026073377579450607d0)
XGBoost  category means, the same matrix read as quantities: (0.8159463405609131d0
                                                              0.40605124831199646d0
                                                              0.5471441149711609d0
                                                              0.46124157309532166d0
                                                              0.5562390089035034d0
                                                              0.19264821708202362d0)
LightGBM backend-supports-p :categorical-features => T
LightGBM category means, :categorical-features '(0): (0.9344864001786668d0
                                                      0.0655135998213332d0
                                                      0.9344864001786668d0
                                                      0.0655135998213332d0
                                                      0.9344864001786668d0
                                                      0.0655135998213332d0)
LightGBM category means, the same matrix read as quantities: (0.8154497898954673d0
                                                              0.4733477173729823d0
                                                              0.5002265133575413d0
                                                              0.5002265133575413d0
                                                              0.5265399565045455d0
                                                              0.184374537772783d0)
XGBoost  saved model's own feature_types field: "feature_types":[],"gradi
XGBoost  category means from a csr-matrix dataset, :categorical-features '(0): (0.9739266037940979d0
                                                                                0.026073377579450607d0
                                                                                0.9739266037940979d0
                                                                                0.026073377579450607d0
                                                                                0.9739266037940979d0
                                                                                0.026073377579450607d0)
XGBoost  :categorical-features ("0") SIGNALED UNSUPPORTED-ARGUMENT: :categorical-features is not supported by XGBOOST: each categorical column must be a non-negative integer index.
XGBoost  :categorical-features (-1) SIGNALED UNSUPPORTED-ARGUMENT: :categorical-features is not supported by XGBOOST: each categorical column must be a non-negative integer index.
XGBoost  :categorical-features (2) SIGNALED UNSUPPORTED-ARGUMENT: :categorical-features is not supported by XGBOOST: categorical column index 2 is beyond the matrix's 2 columns.
XGBoost  :categorical-features (0 0) SIGNALED UNSUPPORTED-ARGUMENT: :categorical-features is not supported by XGBOOST: the same categorical column was named more than once.
XGBoost  evaluation, train :categorical-features '(0), valid without it: ((0
                                                                           "logloss"
                                                                           0.17660880088806152d0)
                                                                          (1
                                                                           "logloss"
                                                                           0.17660880088806152d0))
LightGBM evaluation, train :categorical-features '(0), valid without it: ((0
                                                                           "binary_logloss"
                                                                           0.35374722486733495d0)
                                                                          (1
                                                                           "binary_logloss"
                                                                           0.353747224867335d0))
```

Six categories in one column, alternating class by category, is a fixture where no
threshold on the ordinal separates the two classes but a categorical split choosing the subset
`{0, 2, 4}` does -- taken directly from `tests/functional/categorical-features.lisp`, whose own
comments measure why the small-fixture parameters above are needed, one setting at a time, on
rows this few: `max_depth` `1` on both backends, or spare tree capacity rebuilds the
alternating pattern from the plain ordinal and hides what this fixture measures; XGBoost's
`min_child_weight` `0`, since its default's per-leaf hessian check over four rows rejects the
split; and LightGBM's `min_data_in_leaf`, `min_data_in_bin`, `min_data_per_group`, `cat_smooth`
and `cat_l2`, each of which blocks or weakens a categorical split at its own default on a
fixture this small. On both backends the categorical arm answers one score per class -- `0.974`
positive / `0.026` negative on XGBoost, `0.934`/`0.066` on LightGBM -- while the plain arm, the
identical matrix read as quantities, cannot express that split: six different scores that
barely separate the classes on XGBoost, and on LightGBM two categories (2 and 3) that tie
exactly at `0.500`.

`predict` above is called with no argument naming the categorical column at all -- there is no
such argument to give it -- and every prediction still comes out right: on the dense matrix,
and identically on a `csr-matrix` built from the same rows (`0.9739266037940979d0` and its five
siblings again, digit for digit). XGBoost's own saved model confirms the mechanism: an empty
`"feature_types":[]`, the same field a model trained with no categorical column at all would
save -- the category sets a split needs live in the trees themselves, not on the booster.

The four `unsupported-argument` signals above are the renderer's own rejections
(`cl-gbdt/src/config/categorical-features`), shared by both backends, so the identical four
checks refuse a bad index on LightGBM as well -- naming `LIGHTGBM` in place of `XGBOOST` and
nothing else.

The last two lines are the answer to a question the [missing values](#missing-values) section
above invites: a `:valid-sets` entry built *without* `:categorical-features`, alongside a
training set built *with* it, provokes nothing at all. Training succeeds, and the two entries
evaluate the same way on both backends -- exactly, on XGBoost, and on LightGBM to within
floating-point noise on the order of `1d-17`. That noise is not from this feature: repeating an
otherwise identical comparison with no categorical column anywhere shows the same run-to-run
jitter in LightGBM's `evaluation`, from one run of this section to the next.

### XGBoost: `tree_method` must be `hist` or `approx`

Measured directly: a dataset built with `:categorical-features` trains successfully with
`tree_method` `hist` and with `approx`, but `exact` refuses it -- and only once `train` reaches
`XGBoosterUpdateOneIter`, not at `make-dataset`. The wrapper does not pre-validate
`tree_method`; it is ordinary `:parameters` business, set on the booster rather than the
dataset, and nothing here stops a caller from setting it wrong:

```lisp
(let ((xgb (cl-gbdt:open-backend :xgboost)))
  (unwind-protect
      (cl-gbdt:with-dataset
          (dataset (cl-gbdt:make-dataset xgb (category-matrix) :label (category-labels)
                                          :categorical-features '(0)))
        (format t "make-dataset with :categorical-features '(0) succeeded: rows=~D~%"
                (cl-gbdt:dataset-num-rows dataset))
        (handler-case
            (cl-gbdt:with-booster
                (booster (cl-gbdt:train xgb dataset :num-rounds 5
                                        :parameters '(:objective "binary:logistic" :max-depth 1
                                                      :verbosity 0 :min-child-weight 0
                                                      :tree-method "exact")))
              (declare (ignore booster))
              (format t "train with tree_method exact succeeded (unexpected)~%"))
          (error (c)
            (let ((text (princ-to-string c)))
              (format t "train with tree_method exact SIGNALED ~A:~%  ~A~%" (type-of c)
                      (subseq text 0 (position #\Newline text))))))
        (cl-gbdt:with-booster
            (booster (cl-gbdt:train xgb dataset :num-rounds 5
                                    :parameters '(:objective "binary:logistic" :max-depth 1
                                                  :verbosity 0 :min-child-weight 0
                                                  :tree-method "hist")))
          (declare (ignore booster))
          (format t "train with tree_method hist succeeded~%"))
        (cl-gbdt:with-booster
            (booster (cl-gbdt:train xgb dataset :num-rounds 5
                                    :parameters '(:objective "binary:logistic" :max-depth 1
                                                  :verbosity 0 :min-child-weight 0
                                                  :tree-method "approx")))
          (declare (ignore booster))
          (format t "train with tree_method approx succeeded~%")))
    (cl-gbdt:close-backend xgb)))
```

Output:

```
make-dataset with :categorical-features '(0) succeeded: rows=24
train with tree_method exact SIGNALED FOREIGN-CALL-ERROR:
  XGBoosterUpdateOneIter returned -1: [13:50:17] /__w/xgboost/xgboost/src/tree/updater_colmaker.cc:107: Updater `grow_colmaker` or `exact` tree method doesn't support categorical features.
train with tree_method hist succeeded
train with tree_method approx succeeded
```

The bracketed time in XGBoost's message is its own wall-clock stamp, the only part of this
output that changes between runs -- the same caveat [Sparse input](#sparse-input-csr-matrices)
makes about a different message above. The dataset is built and the feature types attached
without complaint whatever `tree_method` will later be; it is `train`, not `make-dataset`, that
finds out `exact` cannot use them.

### LightGBM: `categorical_feature` and its four aliases

LightGBM's own name for the categorical column list is a parameter-string key,
`categorical_feature`, and the library also honours four synonyms for it -- measured against
the vendored 4.7.0: `cat_feature`, `categorical_column`, `cat_column` and `categorical_features`.
Supplying `:categorical-features` and any of those five spellings in `:parameters` together
signals `unsupported-argument` naming `make-dataset`'s own `:parameters` argument -- not
because the wrapper owns the key outright, but because LightGBM keeps the *first* occurrence of
a duplicated key while `make-dataset` appends its own entry *last*, so the argument the caller
explicitly named would be the one silently discarded. `:parameters` **on its own is
unaffected**: a caller who names no categorical column and writes `categorical_feature` there
by hand -- policy section 6's escape hatch for a backend's own vocabulary -- keeps working
exactly as it did before this argument existed. A near-miss such as `cat_features`, the plural
of the honoured `cat_feature`, is not itself an alias -- LightGBM does not honour it -- and is
never refused, alongside `:categorical-features` or without it: two ways of saying the same
thing remain reachable on this backend, by design.

```lisp
(defparameter *dataset-parameters*
  '(:min-data-in-leaf 1 :min-data-in-bin 1 :min-data-per-group 1 :cat-smooth 0 :cat-l2 0
    :verbose -1))

(let ((lgbm (cl-gbdt:open-backend :lightgbm)))
  (unwind-protect
      (progn
        ;; Both :categorical-features and one of the five spellings LightGBM honours for the
        ;; same key in :parameters: refused, naming make-dataset's own :parameters argument.
        (dolist (key '(:categorical-feature :cat-feature :categorical-column :cat-column
                       :categorical-features))
          (handler-case
              (cl-gbdt:free-dataset
               (cl-gbdt:make-dataset lgbm (category-matrix) :label (category-labels)
                                      :categorical-features '(0)
                                      :parameters (append (list key "0") *dataset-parameters*)))
            (error (c) (format t ":categorical-features '(0) alongside :parameters ~S SIGNALED ~A~%"
                                key (type-of c)))))
        ;; :parameters alone, naming no :categorical-features, is untouched -- the escape hatch.
        (cl-gbdt:with-dataset
            (dataset (cl-gbdt:make-dataset lgbm (category-matrix) :label (category-labels)
                                            :parameters (append '(:categorical-feature "0")
                                                                *dataset-parameters*)))
          (format t ":parameters :categorical-feature alone (no :categorical-features) built ~
                     a dataset: rows=~D~%"
                  (cl-gbdt:dataset-num-rows dataset)))
        ;; cat_features, the plural of the honoured cat_feature, is not itself an alias and is
        ;; never refused -- it reaches the library untouched even alongside :categorical-features.
        (cl-gbdt:with-dataset
            (dataset (cl-gbdt:make-dataset lgbm (category-matrix) :label (category-labels)
                                            :categorical-features '(0)
                                            :parameters (append '(:cat-features "0")
                                                                *dataset-parameters*)))
          (format t ":categorical-features '(0) alongside :parameters :cat-features (not an ~
                     alias) built a dataset: rows=~D~%"
                  (cl-gbdt:dataset-num-rows dataset))))
    (cl-gbdt:close-backend lgbm)))
```

Output:

```
:categorical-features '(0) alongside :parameters :CATEGORICAL-FEATURE SIGNALED UNSUPPORTED-ARGUMENT
:categorical-features '(0) alongside :parameters :CAT-FEATURE SIGNALED UNSUPPORTED-ARGUMENT
:categorical-features '(0) alongside :parameters :CATEGORICAL-COLUMN SIGNALED UNSUPPORTED-ARGUMENT
:categorical-features '(0) alongside :parameters :CAT-COLUMN SIGNALED UNSUPPORTED-ARGUMENT
:categorical-features '(0) alongside :parameters :CATEGORICAL-FEATURES SIGNALED UNSUPPORTED-ARGUMENT
:parameters :categorical-feature alone (no :categorical-features) built a dataset: rows=24
:categorical-features '(0) alongside :parameters :cat-features (not an alias) built a dataset: rows=24
```

### The values inside a categorical column are passed through unvalidated

`:categorical-features` validates which *columns* are categorical -- the four checks above --
never what *values* sit inside them. Measured on LightGBM: a fractional category id truncates
to an integer one silently, and a negative one is converted to missing, with a warning on the
library's own stderr and nothing that reaches Lisp as a condition either way.

```lisp
(defun quiet-nan ()
  "A quiet double-float NaN, built from its bits so no arithmetic can trap."
  (sb-kernel:make-double-float -524288 0))

(defun category-matrix-with-cell (row value)
  "CATEGORY-MATRIX, with ROW's column 0 replaced by VALUE."
  (let ((matrix (category-matrix)))
    (setf (aref matrix row 0) value)
    matrix))

(defun train-predict (backend matrix)
  (cl-gbdt:with-dataset
      (dataset (cl-gbdt:make-dataset backend matrix :label (category-labels)
                                      :categorical-features '(0)
                                      :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                                    :min-data-per-group 1 :cat-smooth 0
                                                    :cat-l2 0 :verbose 1)))
    (cl-gbdt:with-booster
        (booster (cl-gbdt:train backend dataset :num-rounds 20
                                :parameters '(:objective "binary" :max-depth 1 :verbose -1
                                              :min-data-in-leaf 1 :min-data-per-group 1
                                              :cat-smooth 0 :cat-l2 0)))
      (category-means (cl-gbdt:predict booster (category-matrix))))))

(let ((lgbm (cl-gbdt:open-backend :lightgbm)))
  (unwind-protect
      (progn
        ;; A fractional category id truncates to an integer one: every row's column 0 gets
        ;; +0.7, and the categories it truncates to (0..5) are exactly the same as before.
        (format t "integer category ids:                ~S~%"
                (train-predict lgbm (category-matrix)))
        (let ((offset (category-matrix)))
          (dotimes (row (array-dimension offset 0)) (incf (aref offset row 0) 0.7d0))
          (format t "the same ids, each +0.7 (fractional): ~S~%" (train-predict lgbm offset)))

        ;; A negative category value prints a warning on LightGBM's own stderr and is converted
        ;; to missing -- verified against an explicit NaN in the identical cell, which reaches
        ;; the same prediction and prints no warning at all: the two are the same event.
        (format t "row 1 (category 0) set to -1.0d0 (negative):~%  ~S~%"
                (train-predict lgbm (category-matrix-with-cell 1 -1.0d0)))
        (format t "row 1 (category 0) set to an explicit NaN instead:~%  ~S~%"
                (train-predict lgbm (category-matrix-with-cell 1 (quiet-nan)))))
    (cl-gbdt:close-backend lgbm)))
```

Output:

```
integer category ids:                (0.9344864001786668d0 0.0655135998213332d0
                                      0.9344864001786668d0 0.0655135998213332d0
                                      0.9344864001786668d0 0.0655135998213332d0)
the same ids, each +0.7 (fractional): (0.9344864001786668d0
                                       0.0655135998213332d0
                                       0.9344864001786668d0
                                       0.0655135998213332d0
                                       0.9344864001786668d0
                                       0.0655135998213332d0)
[LightGBM] [Warning] Met negative value in categorical features, will convert it to NaN
row 1 (category 0) set to -1.0d0 (negative):
  (0.9344864001786668d0 0.06551359982133317d0 0.9344864001786668d0
   0.06551359982133317d0 0.9344864001786668d0 0.06551359982133317d0)
row 1 (category 0) set to an explicit NaN instead:
  (0.9344864001786668d0 0.06551359982133317d0 0.9344864001786668d0
   0.06551359982133317d0 0.9344864001786668d0 0.06551359982133317d0)
```

Every fractional id in the second run truncates to the same integer category the first run
used, so the two predictions match digit for digit. The negative id in the third run is the
only line that prints a warning -- and its own predictions match the fourth run's, where the
same cell holds an explicit NaN instead of `-1.0d0`, digit for digit as well: LightGBM's
"convert it to NaN" is not a figure of speech, and the fourth run reaches the identical code
path silently, an actual NaN never having been negative to begin with. Both of those runs
differ from the first two only in the three negative categories' shared score
(`0.0655135998213332d0` becomes `0.06551359982133317d0`; category 0's own score, positive and
untouched by the corrupted row, stays bit-identical) -- a real difference, if a small one on
this fixture, from the model having one fewer valid example of column 0 to learn category 0
from. The wrapper validates the *indices* `:categorical-features` names; it never validates
the *values* sitting in the columns those indices point at.

## Prediction shape

`predict` returns two values. The FIRST is exactly what it has always been -- the same
`(simple-array double-float (* *))`, same dimensions, same elements, for every `KIND`, dense
or sparse -- untouched by anything below. The SECOND is new: the SHAPE the backend states for
the result it just wrote, as a list of integers in `array-dimensions` order, or `NIL` where
the backend states none. A caller who ignores it sees behaviour identical to before this
feature existed.

`:prediction-shape` is the capability, answerable through `backend-supports-p` and true on
both vendored backends -- but **no operation refuses on it**. There is no argument asking for
a shape, so a false answer would mean only that the second value is always `NIL`; `predict`
would keep predicting exactly as it does today, on every `KIND`, dense or sparse. That is not
the general rule for a capability in this API -- six of the ten registered capabilities
are re-checked by the operation they gate and signal `capability-unavailable` when they read
false: `:sparse-input`, `:missing-value`, `:categorical-features`, `:custom-objective` and
`:custom-evaluation` (each documented above, on the operation that checks it -- the last two
at [Custom objective](custom-training.md#custom-objective) and
[Custom evaluation](custom-training.md#custom-evaluation)) and
`:model-slicing` (see [Asking a backend what it
can do](backends.md#asking-a-backend-what-it-can-do)). `:prediction-shape` is simply not one of
those
six, because it gates nothing a caller asks for.

The two backends fill the second value from opposite directions. XGBoost's prediction entry
points write an `out_shape`/`out_dim` pair, and `predict` reads that pair straight back and
states exactly what the library said. LightGBM's do not -- `LGBM_BoosterCalcNumPredict`
returns an element count and nothing else -- so LightGBM's second value is DERIVED, and only
as far as the derivation can go: `:normal` and `:raw` state the result array's own
`array-dimensions` (there is nothing to add to what the array already says), `:contrib` is
derived from that element count, the row count, and a further library call this derivation
makes, `LGBM_BoosterGetNumFeature`, and `:leaf-index` states `NIL` -- see below for why.

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

;; Eighteen rows, four columns, three classes, six rows per class -- large enough that
;; :leaf-index and :contrib's extra axes (rounds, output groups, features+1) are all
;; different numbers, so a reader cannot mistake one axis for another by coincidence.
(defparameter *shape-matrix*
  (let* ((rows-per-class 6) (num-classes 3) (cols 4) (rows (* rows-per-class num-classes))
         (matrix (make-array (list rows cols) :element-type 'double-float)))
    (dotimes (row rows)
      (let ((class (floor row rows-per-class)) (offset (mod row rows-per-class)))
        (dotimes (col cols)
          (setf (aref matrix row col) (coerce (+ (* class 10) offset col) 'double-float)))))
    matrix))
(defparameter *shape-label*
  (let* ((rows-per-class 6) (rows (* rows-per-class 3))
         (label (make-array rows :element-type 'single-float)))
    (dotimes (row rows) (setf (aref label row) (coerce (floor row rows-per-class) 'single-float)))
    label))

(defun show-shapes (name backend dataset-parameters booster-parameters)
  (format t "~A backend-supports-p :prediction-shape => ~S~%"
          name (cl-gbdt:backend-supports-p backend :prediction-shape))
  (cl-gbdt:with-dataset (dataset (apply #'cl-gbdt:make-dataset backend *shape-matrix*
                                        :label *shape-label* dataset-parameters))
    (cl-gbdt:with-booster (booster (cl-gbdt:train backend dataset :num-rounds 4
                                                   :parameters booster-parameters))
      (dolist (kind '(:normal :raw :leaf-index :contrib))
        (multiple-value-bind (result shape) (cl-gbdt:predict booster *shape-matrix* :kind kind)
          (format t "~A ~S: array-dimensions ~S, shape ~S~%"
                  name kind (array-dimensions result) shape))))))

(let ((lgbm (cl-gbdt:open-backend :lightgbm))
      (xgb (cl-gbdt:open-backend :xgboost)))
  (show-shapes "LightGBM" lgbm
               '(:parameters (:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1))
               '(:objective "multiclass" :num-class 3 :num-leaves 2 :min-data-in-leaf 1
                 :min-data-in-bin 1 :verbose -1))
  (show-shapes "XGBoost " xgb '()
               '(:objective "multi:softprob" :num-class 3 :max-depth 3 :eta 0.5 :verbosity 0))
  (cl-gbdt:close-backend lgbm)
  (cl-gbdt:close-backend xgb))
```

Output:

```
LightGBM backend-supports-p :prediction-shape => T
LightGBM :NORMAL: array-dimensions (18 3), shape (18 3)
LightGBM :RAW: array-dimensions (18 3), shape (18 3)
LightGBM :LEAF-INDEX: array-dimensions (18 12), shape NIL
LightGBM :CONTRIB: array-dimensions (18 15), shape (18 3 5)
XGBoost  backend-supports-p :prediction-shape => T
XGBoost  :NORMAL: array-dimensions (18 3), shape (18 3)
XGBoost  :RAW: array-dimensions (18 3), shape (18 3)
XGBoost  :LEAF-INDEX: array-dimensions (18 12), shape (18 4 3 1)
XGBoost  :CONTRIB: array-dimensions (18 15), shape (18 3 5)
```

`:normal` and `:raw` state `(18 3)` on both backends -- the array's own `array-dimensions`,
so there is nothing here beyond what the first value already said. `:leaf-index` and
`:contrib` are where the two backends diverge. XGBoost states four and three axes
respectively, RICHER than the `18x12` and `18x15` arrays `predict`'s first value returns for
them: before this branch, `predict` folded those same axes into the array's own two,
discarding the structure the library had already reported. LightGBM's `:contrib` derives the
identical three axes arithmetically from a count and two further numbers; its `:leaf-index`
states `NIL` -- LightGBM's `predict` still returns the `18x12` array for it, exactly as
before, since no operation refuses on this capability and a `NIL` second value changes
nothing about the first.

### Binary models are multidimensional too

The case a reader guesses wrong: `:leaf-index` and `:contrib`'s extra axes look like a
multiclass artifact in the block above, where every shape happens to mention 3. They are not.

```lisp
;; A trivially separable eight-row three-column fixture, in the same spirit as
;; tests/functional/support.lisp's make-separable-dataset -- one output group,
;; unlike *SHAPE-MATRIX*'s three.
(defparameter *shape-bin-matrix*
  (let ((rows 8) (cols 3))
    (let ((m (make-array (list rows cols) :element-type 'double-float)))
      (dotimes (i rows)
        (dotimes (j cols) (setf (aref m i j) (coerce (/ (+ i j) 10) 'double-float))))
      m)))
(defparameter *shape-bin-label*
  (let ((rows 8))
    (let ((l (make-array rows :element-type 'single-float)))
      (dotimes (i rows)
        (setf (aref l i) (if (> (aref *shape-bin-matrix* i 0) 0.35d0) 1.0 0.0)))
      l)))

(let ((xgb (cl-gbdt:open-backend :xgboost)))
  (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset xgb *shape-bin-matrix*
                                                        :label *shape-bin-label*))
    (cl-gbdt:with-booster (booster (cl-gbdt:train xgb dataset :num-rounds 4
                                     :parameters '(:objective "binary:logistic" :max-depth 2
                                                   :eta 0.5 :verbosity 0)))
      (dolist (kind '(:leaf-index :contrib))
        (multiple-value-bind (result shape)
            (cl-gbdt:predict booster *shape-bin-matrix* :kind kind)
          (format t "XGBoost binary model ~S: array-dimensions ~S, shape ~S~%"
                  kind (array-dimensions result) shape)))))
  (cl-gbdt:close-backend xgb))
```

Output:

```
XGBoost binary model :LEAF-INDEX: array-dimensions (8 4), shape (8 4 1 1)
XGBoost binary model :CONTRIB: array-dimensions (8 4), shape (8 1 4)
```

One output group, and both shapes are still multidimensional -- `:leaf-index` four axes,
`:contrib` three. This fixture is also where the first value alone stops being enough: at
four rounds over three columns, `:leaf-index`'s folded width (4 rounds x 1 class) and
`:contrib`'s (1 class x (3 features + 1)) are both 4, so the array `predict` returns is
`8x4` for either `KIND` -- the same shape, from two calls that mean completely different
things. The second value is what tells them apart: `(8 4 1 1)` against `(8 1 4)`, not the same
list even though both multiply out to 4 x 8 elements.

### What backs LightGBM's derived ordering, and the view built from it

`:contrib`'s three axes are `(rows classes features+1)`, CLASS-MAJOR: every output group's
own `features+1` contributions sit together, one group after another. The arithmetic in
`contrib-shape` divides the element count into three numbers exactly as well with the last
two axes swapped -- `(rows features+1 classes)`, FEATURE-MAJOR -- so the ordering is a claim
the division alone cannot support. What supports it is a property of what SHAP contributions
mean: the contributions for one output group sum to that group's own `:raw` score. Grouped
the way the shape claims, they should; grouped the other way, they should not.

```lisp
;; *SHAPE-MATRIX* and *SHAPE-LABEL* as defined above.
(let ((lgbm (cl-gbdt:open-backend :lightgbm)))
  (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset lgbm *shape-matrix* :label *shape-label*
                                   :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                                 :verbose -1)))
    (cl-gbdt:with-booster (booster (cl-gbdt:train lgbm dataset :num-rounds 4
                                     :parameters '(:objective "multiclass" :num-class 3
                                                   :num-leaves 2 :min-data-in-leaf 1
                                                   :min-data-in-bin 1 :verbose -1)))
      (let ((raw (cl-gbdt:predict booster *shape-matrix* :kind :raw)))
        (multiple-value-bind (contrib shape) (cl-gbdt:predict booster *shape-matrix* :kind :contrib)
          (format t "contrib array-dimensions ~S, derived shape ~S~%"
                  (array-dimensions contrib) shape)
          ;; The one-form N-dimensional view: SHAPE describes CONTRIB's own buffer, so the
          ;; displaced array below reads it three-dimensionally without copying anything.
          (let ((view (make-array shape :element-type 'double-float :displaced-to contrib))
                (worst 0.0d0))
            (destructuring-bind (rows classes width) shape
              (dotimes (row rows)
                (dotimes (class classes)
                  (let ((summed (loop :for feature :below width
                                       :sum (aref view row class feature))))
                    (setf worst (max worst (abs (- summed (aref raw row class)))))))))
            (format t "class-major sums vs :raw, worst absolute difference: ~,3E~%" worst))
          ;; The control: the same buffer, read with the last two axes swapped -- what
          ;; :contrib's shape would be if the derivation had guessed the wrong order.
          (let* ((bad-shape (list (first shape) (third shape) (second shape)))
                 (view (make-array bad-shape :element-type 'double-float :displaced-to contrib))
                 (worst 0.0d0))
            (destructuring-bind (rows width classes) bad-shape
              (dotimes (row rows)
                (dotimes (class classes)
                  (let ((summed (loop :for feature :below width
                                       :sum (aref view row feature class))))
                    (setf worst (max worst (abs (- summed (aref raw row class)))))))))
            (format t "feature-major control vs :raw, worst absolute difference: ~,3E~%" worst))))))
  (cl-gbdt:close-backend lgbm))
```

Output:

```
contrib array-dimensions (18 15), derived shape (18 3 5)
class-major sums vs :raw, worst absolute difference: 2.220d-16
feature-major control vs :raw, worst absolute difference: 7.783d-1
```

`(make-array shape :displaced-to contrib)` is the one-form N-dimensional view: `SHAPE`
already describes `CONTRIB`'s own storage, so the displaced array reads that same buffer
three-dimensionally, `(aref view row class feature)` in place of hand-rolled row/column
arithmetic on the flat array -- the point of returning a shape at all, not a footnote to it.
Read that way, class-major reproduces the raw scores to `2.220d-16`, floating-point roundoff
and nothing more; read the other way, the feature-major control misses every one of the 54
`(row, class)` sums by up to `7.783d-1`, some fifteen orders of magnitude larger. That gap is
what turns the ordering from an assumption into a measurement, held by
`lightgbm-s-derived-contrib-shape-is-the-one-the-numbers-support` in
`tests/functional/prediction-shape.lisp`.

`:leaf-index` gets no such derivation. Its element count divides by iterations and output
groups exactly as `:contrib`'s divides by output groups and width, but a leaf index is an
opaque identifier -- it sums to nothing and agrees with nothing, so there is no SHAP-sum-style
property here to check a guessed ordering against. Asserting an ordering with nothing to check
it against would be exactly the mistake the measurement above exists to avoid: a shape stated
on arithmetic alone, with no SHAP-sum-style test and no feature-major control to catch it if
the axes were transposed. `NIL` is what `predict`'s second value means everywhere a backend
states none, and LightGBM's `:leaf-index` result -- the `18x12` array -- is entirely
unaffected by stating it.

### On a `csr-matrix`, XGBoost states a shape for two kinds out of four

[Sparse input](#sparse-input-csr-matrices) above measures that XGBoost's sparse entry point,
`XGBoosterPredictFromCSR`, serves only `:normal` and `:raw`, refusing `:contrib` and
`:leaf-index` with `foreign-call-error` before either produces a result. A shape is read only
from a call that returned one, so the same split holds here: a `csr-matrix` reaches a shape
for the two `KIND`s that succeed, and never reaches one for the two that do not.

```lisp
;; *SHAPE-MATRIX* and *SHAPE-LABEL* as defined above.
(defun dense-to-csr (matrix)
  "MATRIX as a `csr-matrix' with every element stored explicitly -- see 'An absent entry
is not a zero' above for why dropping the zeros would describe a different matrix to XGBoost."
  (let* ((rows (array-dimension matrix 0)) (cols (array-dimension matrix 1))
         (indptr (make-array (1+ rows))) (indices (make-array (* rows cols)))
         (values (make-array (* rows cols))) (pos 0))
    (dotimes (r rows)
      (setf (aref indptr r) pos)
      (dotimes (c cols)
        (setf (aref indices pos) c) (setf (aref values pos) (aref matrix r c)) (incf pos)))
    (setf (aref indptr rows) pos)
    (cl-gbdt:make-csr-matrix :indptr indptr :indices indices :values values :num-columns cols)))

(let ((xgb (cl-gbdt:open-backend :xgboost))
      (csr (dense-to-csr *shape-matrix*)))
  (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset xgb *shape-matrix* :label *shape-label*))
    (cl-gbdt:with-booster (booster (cl-gbdt:train xgb dataset :num-rounds 4
                                     :parameters '(:objective "multi:softprob" :num-class 3
                                                   :max-depth 3 :eta 0.5 :verbosity 0)))
      (dolist (kind '(:normal :raw :leaf-index :contrib))
        (handler-case
            (multiple-value-bind (result shape) (cl-gbdt:predict booster csr :kind kind)
              (format t "XGBoost csr-matrix ~S: array-dimensions ~S, shape ~S~%"
                      kind (array-dimensions result) shape))
          ;; XGBoost's message carries a multi-line stack trace; line 1 is the refusal.
          (error (c) (let ((text (princ-to-string c)))
                       (format t "XGBoost csr-matrix ~S: SIGNALED ~A~%  ~A~%" kind (type-of c)
                               (subseq text 0 (position #\Newline text)))))))))
  (cl-gbdt:close-backend xgb))
```

Output:

```
XGBoost csr-matrix :NORMAL: array-dimensions (18 3), shape (18 3)
XGBoost csr-matrix :RAW: array-dimensions (18 3), shape (18 3)
XGBoost csr-matrix :LEAF-INDEX: SIGNALED FOREIGN-CALL-ERROR
  XGBoosterPredictFromCSR returned -1: [08:23:10] /__w/xgboost/xgboost/src/learner.cc:1264: Unsupported prediction type:6
XGBoost csr-matrix :CONTRIB: SIGNALED FOREIGN-CALL-ERROR
  XGBoosterPredictFromCSR returned -1: [08:23:10] /__w/xgboost/xgboost/src/learner.cc:1264: Unsupported prediction type:2
```

As in Sparse input above, the bracketed time in XGBoost's two messages is XGBoost's own
wall-clock stamp, the only part of this output that differs run to run. `:normal` and `:raw`
state `(18 3)`, identical to the dense call earlier in this section; `:leaf-index` and
`:contrib` never reach a shape, or a result, at all. LightGBM has no such split -- its CSR
entry point serves all four `KIND`s, and states, or declines to state, a shape for each of
them exactly as it does on a dense matrix.
