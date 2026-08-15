# Custom training

How `train`'s `:objective` and `:evaluation` callbacks let a run boost against a caller's own loss and record a caller's own metrics.

## Custom objective

`train` also takes `:objective`, a function that turns the current raw scores into a gradient
and a Hessian, so a run boosts against the caller's own loss instead of one built into the
library. It needs the `:custom-objective` capability, answerable through `backend-supports-p`
and true on both vendored backends; `train` re-checks it itself and signals
`capability-unavailable` for a non-`NIL` `:objective` when it reads false, before any foreign
call.

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified) :silent t)

(defparameter *co-matrix*
  (make-array '(8 1) :element-type 'double-float
              :initial-contents '((0.0d0) (1.0d0) (2.0d0) (3.0d0)
                                   (4.0d0) (5.0d0) (6.0d0) (7.0d0))))
(defparameter *co-label*
  (make-array 8 :element-type 'double-float
              :initial-contents '(0.0d0 1.0d0 4.0d0 9.0d0 16.0d0 25.0d0 36.0d0 49.0d0)))

(defun squared-error (scores)
  "GRAD = prediction - label, HESS = 1 -- squared error's own derivatives."
  (let* ((rows (array-dimension scores 0))
         (grad (make-array (list rows 1) :element-type 'double-float))
         (hess (make-array (list rows 1) :element-type 'double-float :initial-element 1.0d0)))
    (dotimes (row rows (values grad hess))
      (setf (aref grad row 0) (- (aref scores row 0) (aref *co-label* row))))))

(let ((backend (cl-gbdt:open-backend :lightgbm)))
  (unwind-protect
      (cl-gbdt:with-dataset
          (dataset (cl-gbdt:make-dataset backend *co-matrix* :label *co-label*
                                          :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                                        :verbose -1)))
        (cl-gbdt:with-booster
            (built-in (cl-gbdt:train backend dataset :num-rounds 5
                                      :parameters '(:objective "regression" :num-leaves 4
                                                    :min-data-in-leaf 1 :min-data-in-bin 1
                                                    :verbose -1 :boost-from-average nil)))
          (cl-gbdt:with-booster
              (custom (cl-gbdt:train backend dataset :num-rounds 5
                                      :parameters '(:num-leaves 4 :min-data-in-leaf 1
                                                    :min-data-in-bin 1 :verbose -1
                                                    :boost-from-average nil)
                                      :objective #'squared-error))
            (format t "built-in :raw:~%~S~%" (cl-gbdt:predict built-in *co-matrix* :kind :raw))
            (format t "custom  :raw:~%~S~%" (cl-gbdt:predict custom *co-matrix* :kind :raw)))))
    (cl-gbdt:close-backend backend)))
```

Output:

```
built-in :raw:
#2A((0.9006424501538277d0)
    (0.9006424501538277d0)
    (0.9006424501538277d0)
    (3.7816945374011977d0)
    (5.989342808723447d0)
    (11.236049175262446d0)
    (13.940538883209221d0)
    (19.681847476959206d0))
custom  :raw:
#2A((0.9006424501538277d0)
    (0.9006424501538277d0)
    (0.9006424501538277d0)
    (3.7816945374011977d0)
    (5.989342808723447d0)
    (11.236049175262446d0)
    (13.940538883209221d0)
    (19.681847476959206d0))
```

`SQUARED-ERROR` above is squared error's own gradient and Hessian, `grad = prediction - label`
and `hess = 1` -- the same derivatives LightGBM's built-in `"regression"` objective uses -- and
the two runs land on the identical model, digit for digit: a custom objective is not an
approximation of the library's own, it drives the same trees when it computes the same thing.

`:objective` is called once per iteration, before that iteration's update, with **one
argument**: the booster's current raw scores for its training set, as a `(ROWS GROUPS)`
`double-float` array -- the margin, before any sigmoid or softmax transform, and the same shape
and element type `predict` returns. `GROUPS` is 1 for regression and binary classification and
`num_class` for multiclass. It must return **two values**, the gradient and the Hessian, each a
`(ROWS GROUPS)` array. The **shape** is what is checked -- the wrong rank, the wrong
dimensions, or one value instead of two signals `dimension-mismatch` before any foreign call,
so a wrongly shaped array is never read as though it had the right shape:

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified) :silent t)

(defparameter *co-matrix*
  (make-array '(8 1) :element-type 'double-float
              :initial-contents '((0.0d0) (1.0d0) (2.0d0) (3.0d0)
                                   (4.0d0) (5.0d0) (6.0d0) (7.0d0))))
(defparameter *co-label*
  (make-array 8 :element-type 'double-float
              :initial-contents '(0.0d0 1.0d0 4.0d0 9.0d0 16.0d0 25.0d0 36.0d0 49.0d0)))

(let ((backend (cl-gbdt:open-backend :lightgbm)))
  (unwind-protect
      (cl-gbdt:with-dataset
          (dataset (cl-gbdt:make-dataset backend *co-matrix* :label *co-label*
                                          :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                                        :verbose -1)))
        ;; A flat (ROWS) vector instead of the required (ROWS GROUPS) array -- the shape a
        ;; caller who thinks in one dimension returns.
        (handler-case
            (cl-gbdt:train backend dataset :num-rounds 1
                            :objective (lambda (scores)
                                         (declare (ignore scores))
                                         (values (make-array 8 :element-type 'double-float
                                                                :initial-element 0.0d0)
                                                 (make-array '(8 1) :element-type 'double-float
                                                                    :initial-element 1.0d0))))
          (error (c) (format t "SIGNALED ~A~%  ~A~%" (type-of c) c))))
    (cl-gbdt:close-backend backend)))
```

Output:

```
SIGNALED DIMENSION-MISMATCH
  Dimension mismatch. Expected: (8 1), got: (GRADIENT (8) HESSIAN (8 1))
```

The **element type is not** part of that check, and deliberately. `double-float`,
`single-float` and a general array whose elements are reals -- what `(make-array (list rows 1))`
with no `:element-type` gives, the most natural thing to write -- are all accepted, and all
three train the identical model on both backends, because each element is coerced to the
`single-float` the C signature's `const float*` takes as the buffer is written. An element that
is *not* a real -- a string, `NIL`, a complex -- signals `unsupported-element-type` naming that
element's own type, at the write and before the library has been called: the same condition,
with the same value in `unsupported-element-type-given`, that a `csr-matrix` holding a non-real
value already signals from `make-csr-matrix`. Nothing scans either array a second time to say
so; the check rides along with the coercion that was happening anyway.

The two libraries want that `(ROWS GROUPS)` array flattened into their C buffers in opposite
orders -- LightGBM **group-major** (row I of group K at `(+ (* K ROWS) I)`), XGBoost
**row-major** (row I of group K at `(+ (* I GROUPS) K)`, what an `__array_interface__` of shape
`[ROWS, GROUPS]` means) -- and each backend's own code absorbs that difference. **The
flattening is the wrapper's job, not the caller's**: `:objective` is handed, and returns, one
`(ROWS GROUPS)` array on both backends, whichever order the library underneath actually wants
it in. Both orderings are measured rather than assumed -- a gradient confined to one output
group moves only that group's raw score under the correct layout and smears across every group
under the other -- held by
`a-gradient-in-one-output-group-moves-only-that-group` in
`tests/functional/custom-objective.lisp`, which runs the same fixture on both backends and
would fail if either flattening were transposed.

`:objective` is the only place inside `train`'s loop where code cl-gbdt did not write runs, and
that code can reach the handles the loop is holding: `free-dataset` on the training set, on a
`:valid-sets` entry, or `close-backend` on the backend itself. All three are **caught**, not
crashed on. `train` re-runs its own dataset and backend checks the moment the objective
returns, before the iteration makes another foreign call, and reads fresh pointers from them --
so freeing the training set from inside an objective signals `released-handle-error` naming
that dataset, exactly as freeing it anywhere else in this library does. Without that re-check
the loop hands a pointer into freed memory straight to C: measured, LightGBM died with
`Memory fault at 0x543447170e8a6` and XGBoost with `Signal 7 received`, killing the process
rather than signalling anything a caller could handle.

### LightGBM forces `objective` to `"none"`

`LGBM_BoosterUpdateOneIterCustom` refuses to run at all while the booster holds an objective
function -- `Check failed: objective_function_ == nullptr`, measured against the vendored
library to return non-zero and train nothing -- so a non-`NIL` `:objective` on LightGBM
**overrides** any `objective` entry in `:parameters`, forcing it to `"none"` before
`LGBM_BoosterCreate` ever sees the string. This is not a convenience the caller can opt out of:
the combination it replaces has no working form to preserve. Every other parameter passes
through untouched and in its original order, `num_class` included, which is still what tells
LightGBM how many output groups a multiclass custom objective has.

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified) :silent t)

(defparameter *co-matrix*
  (make-array '(8 1) :element-type 'double-float
              :initial-contents '((0.0d0) (1.0d0) (2.0d0) (3.0d0)
                                   (4.0d0) (5.0d0) (6.0d0) (7.0d0))))
(defparameter *co-label*
  (make-array 8 :element-type 'double-float
              :initial-contents '(0.0d0 1.0d0 4.0d0 9.0d0 16.0d0 25.0d0 36.0d0 49.0d0)))

(defun squared-error (scores)
  "GRAD = prediction - label, HESS = 1 -- squared error's own derivatives."
  (let* ((rows (array-dimension scores 0))
         (grad (make-array (list rows 1) :element-type 'double-float))
         (hess (make-array (list rows 1) :element-type 'double-float :initial-element 1.0d0)))
    (dotimes (row rows (values grad hess))
      (setf (aref grad row 0) (- (aref scores row 0) (aref *co-label* row))))))

;; A caller who explicitly names LightGBM's own "regression" objective in :parameters
;; alongside :objective still gets the identical model a run naming no objective there gets --
;; proof the override happened, not merely documented.
(let ((backend (cl-gbdt:open-backend :lightgbm)))
  (unwind-protect
      (cl-gbdt:with-dataset
          (dataset (cl-gbdt:make-dataset backend *co-matrix* :label *co-label*
                                          :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                                        :verbose -1)))
        (cl-gbdt:with-booster
            (silent (cl-gbdt:train backend dataset :num-rounds 5
                                    :parameters '(:num-leaves 4 :min-data-in-leaf 1
                                                  :min-data-in-bin 1 :verbose -1
                                                  :boost-from-average nil)
                                    :objective #'squared-error))
          (cl-gbdt:with-booster
              (overridden (cl-gbdt:train backend dataset :num-rounds 5
                                          :parameters '(:objective "regression" :num-leaves 4
                                                        :min-data-in-leaf 1 :min-data-in-bin 1
                                                        :verbose -1 :boost-from-average nil)
                                          :objective #'squared-error))
            (format t "no objective named in :parameters, :raw:~%~S~%"
                    (cl-gbdt:predict silent *co-matrix* :kind :raw))
            (format t "\"regression\" named in :parameters too, :raw:~%~S~%"
                    (cl-gbdt:predict overridden *co-matrix* :kind :raw)))))
    (cl-gbdt:close-backend backend)))
```

Output:

```
no objective named in :parameters, :raw:
#2A((0.9006424501538277d0)
    (0.9006424501538277d0)
    (0.9006424501538277d0)
    (3.7816945374011977d0)
    (5.989342808723447d0)
    (11.236049175262446d0)
    (13.940538883209221d0)
    (19.681847476959206d0))
"regression" named in :parameters too, :raw:
#2A((0.9006424501538277d0)
    (0.9006424501538277d0)
    (0.9006424501538277d0)
    (3.7816945374011977d0)
    (5.989342808723447d0)
    (11.236049175262446d0)
    (13.940538883209221d0)
    (19.681847476959206d0))
```

Naming `"regression"` explicitly changes nothing: `train` drops every `objective` entry
`:parameters` holds and appends its own `:objective "none"` last, so the two runs above are the
identical booster. **XGBoost's `:parameters` are never rewritten** --
`XGBoosterTrainOneIter` has no such restriction, measured to accept a custom update with any
objective set -- so there is nothing on that backend to override.

**"Every `objective` entry" means all five spellings LightGBM honours**, not the literal one
alone. That library reads `objective_type`, `app`, `application` and `loss` as aliases for
`objective` -- its own `LGBM_DumpParamAliases` returns
`"objective": ["app", "loss", "application", "objective_type"]`, and each of the four is live
in the vendored 4.7.0, `:app "binary"` training the identical model `:objective "binary"`
trains. All five are dropped, so `:app "regression"` alongside `:objective #'squared-error`
behaves exactly like the `:objective "regression"` run above. The list can only be enumerated,
never prefix-matched: `apps` is *not* an alias, and neither is `objective_seed`, which is a
real LightGBM parameter in its own right -- dropping either would silently delete a caller's
configuration. Only keys that render to one of the five are touched; everything else passes
through in its original order.

Finally, a non-`NIL` `:objective` must be a `function`. A number, a string, or a *symbol*
naming a function signals `unsupported-argument` naming `train's :objective`, before any
foreign call and so before a booster exists -- on both backends. The symbol case is refused
deliberately: `funcall` would have accepted it and resolved it afresh at each iteration
against whatever global definition happened to be in force, rather than against what the
caller passed.

### The remaining divergence: what `:normal` means under a custom objective

Because LightGBM's objective is forced to `"none"` while XGBoost's stays whatever the caller
configured, the two backends disagree about what `predict`'s `:kind :normal` means under a
custom objective. LightGBM applies no transform at all, so `:normal` equals `:raw` there. A
configured XGBoost objective's own prediction transform stays in effect regardless of who
supplied the gradient, so with `binary:logistic` still set, `:normal` returns probabilities of
a margin the caller's own loss produced, while `:raw` returns that margin untouched:

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

(defparameter *co-matrix*
  (make-array '(8 1) :element-type 'double-float
              :initial-contents '((0.0d0) (1.0d0) (2.0d0) (3.0d0)
                                   (4.0d0) (5.0d0) (6.0d0) (7.0d0))))
(defparameter *co-binary-label*
  (make-array 8 :element-type 'single-float
              :initial-contents '(0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0)))

(defun logistic-objective (scores)
  "GRAD/HESS for logistic loss over *CO-BINARY-LABEL*, from the raw margin SCORES."
  (let* ((rows (array-dimension scores 0))
         (grad (make-array (list rows 1) :element-type 'double-float))
         (hess (make-array (list rows 1) :element-type 'double-float)))
    (dotimes (row rows (values grad hess))
      (let ((p (/ 1.0d0 (+ 1.0d0 (exp (- (aref scores row 0)))))))
        (setf (aref grad row 0) (- p (aref *co-binary-label* row)))
        (setf (aref hess row 0) (max 1d-6 (* p (- 1.0d0 p))))))))

;; LightGBM's :normal equals its :raw under a custom objective, since :objective forces
;; "objective":"none". XGBoost's configured objective keeps transforming: :normal differs
;; from :raw there.
(let ((lgbm (cl-gbdt:open-backend :lightgbm))
      (xgb (cl-gbdt:open-backend :xgboost)))
  (unwind-protect
      (progn
        (cl-gbdt:with-dataset
            (dataset (cl-gbdt:make-dataset lgbm *co-matrix* :label *co-binary-label*
                                            :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1
                                                          :verbose -1)))
          (cl-gbdt:with-booster
              (booster (cl-gbdt:train lgbm dataset :num-rounds 5
                                       :parameters '(:num-leaves 4 :min-data-in-leaf 1
                                                     :min-data-in-bin 1 :verbose -1)
                                       :objective #'logistic-objective))
            (format t "LightGBM :normal:~%~S~%" (cl-gbdt:predict booster *co-matrix* :kind :normal))
            (format t "LightGBM :raw:~%~S~%" (cl-gbdt:predict booster *co-matrix* :kind :raw))))
        (cl-gbdt:with-dataset
            (dataset (cl-gbdt:make-dataset xgb *co-matrix* :label *co-binary-label*))
          (cl-gbdt:with-booster
              (booster (cl-gbdt:train xgb dataset :num-rounds 5
                                       :parameters '(:objective "binary:logistic" :max-depth 2
                                                     :eta 0.5d0 :verbosity 0)
                                       :objective #'logistic-objective))
            (format t "XGBoost  :normal:~%~S~%" (cl-gbdt:predict booster *co-matrix* :kind :normal))
            (format t "XGBoost  :raw:~%~S~%" (cl-gbdt:predict booster *co-matrix* :kind :raw)))))
    (cl-gbdt:close-backend lgbm)
    (cl-gbdt:close-backend xgb)))
```

Output:

```
LightGBM :normal:
#2A((-0.857090443203849d0)
    (-0.857090443203849d0)
    (-0.857090443203849d0)
    (-0.857090443203849d0)
    (0.8570904432038492d0)
    (0.8570904432038492d0)
    (0.8570904432038492d0)
    (0.8570904432038492d0))
LightGBM :raw:
#2A((-0.857090443203849d0)
    (-0.857090443203849d0)
    (-0.857090443203849d0)
    (-0.857090443203849d0)
    (0.8570904432038492d0)
    (0.8570904432038492d0)
    (0.8570904432038492d0)
    (0.8570904432038492d0))
XGBoost  :normal:
#2A((0.3775406777858734d0)
    (0.3775406777858734d0)
    (0.3775406777858734d0)
    (0.3775406777858734d0)
    (0.622459352016449d0)
    (0.622459352016449d0)
    (0.622459352016449d0)
    (0.622459352016449d0))
XGBoost  :raw:
#2A((-0.5d0) (-0.5d0) (-0.5d0) (-0.5d0) (0.5d0) (0.5d0) (0.5d0) (0.5d0))
```

LightGBM's `:normal` and `:raw` match to the last digit; XGBoost's do not, and its `:normal`
values sit in `(0, 1)`, a sigmoid of its own `:raw` margin. One custom-objective run, two
meanings for `:normal` -- a caller moving the same `:objective` function between backends gets
a probability from one and a margin from the other unless they account for it.

### What `:objective` sees on XGBoost's training set, and DART

XGBoost has no counterpart to LightGBM's `LGBM_BoosterGetPredict`, which simply hands back
scores the booster already holds; each iteration's scores for `:objective` are instead a fresh
margin prediction over the training `DMatrix`, sent with `"training":true` in that call's
config JSON. The vendored header
(`ffi-spec/xgboost/include/xgboost/c_api.h:1180-1191`) documents that key as distinguishing two
prediction scenarios, obtaining `y_pred` versus "obtain[ing] the prediction for computing
gradients", and says the second "applies when you are defining a custom objective function".
On the default `gbtree` booster the two values were measured to train identical models; the
same header names DART's training-time dropout as the case where they differ, since dropped
trees make "the prediction result... different from the one obtained by normal inference step".
**That DART difference is the header's own statement, not a measurement taken in this
repository** -- no test here exercises `:booster "dart"` together with `:objective`, though
`:booster` reaches XGBoost's parameters untouched, so the combination is reachable and simply
unmeasured.

Two further things this argument does not change. A library metric configured through
`:parameters` relates to the library's own objective, not to the caller's, so what
`:record-history` records and what `:early-stopping` watches -- see [Training
report](training.md#training-report) -- may be meaningless under a custom objective; nothing
here signals
about that, the caller decides. And `:objective` is `funcall`ed inside `train`'s own
floating-point-trap mask, so the caller's Lisp arithmetic runs under the same masked convention
the two C libraries are written against: `(/ 1.0d0 0.0d0)` yields infinity rather than
signalling, on x86-64 as well as on aarch64.

## Custom evaluation

`train` also takes `:evaluation`, a function called once per dataset per iteration, after
that iteration's update, so a run records the caller's own measure of fit beside the
library's own metrics. It needs the `:custom-evaluation` capability, answerable through
`backend-supports-p` and true on both vendored backends; `train` re-checks it itself and
signals `capability-unavailable` for a non-`NIL` `:evaluation` when it reads false, before
any foreign call.

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

(defparameter *ce-matrix*
  (make-array '(8 2) :element-type 'double-float
              :initial-contents '((0.0d0 0.0d0) (0.0d0 1.0d0) (0.0d0 2.0d0)
                                   (0.0d0 3.0d0) (5.0d0 0.0d0) (5.0d0 1.0d0)
                                   (5.0d0 2.0d0) (5.0d0 3.0d0))))
(defparameter *ce-label*
  (make-array 8 :element-type 'single-float
              :initial-contents '(0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0)))

(defun my-logloss (scores index)
  "A caller-written binary log loss over SCORES -- predict :kind :normal's probabilities, not
the margin :objective would see. INDEX is ignored here since both datasets below share
*CE-LABEL*."
  (declare (ignore index))
  (let ((rows (array-dimension scores 0)) (sum 0d0))
    (dotimes (row rows (values "my_logloss" (/ sum rows)))
      (let ((p (min (max (aref scores row 0) 1d-15) (- 1d0 1d-15)))
            (y (coerce (aref *ce-label* row) 'double-float)))
        (incf sum (- (+ (* y (log p)) (* (- 1d0 y) (log (- 1d0 p))))))))))

(defun show-custom-evaluation (name backend dataset-parameters booster-parameters reference-p)
  (cl-gbdt:with-dataset (train-set (apply #'cl-gbdt:make-dataset backend *ce-matrix*
                                          :label *ce-label* dataset-parameters))
    (cl-gbdt:with-dataset (valid-set (apply #'cl-gbdt:make-dataset backend *ce-matrix*
                                            :label *ce-label*
                                            (append (when reference-p
                                                      (list :reference train-set))
                                                    dataset-parameters)))
      (multiple-value-bind (booster report)
          (cl-gbdt:train backend train-set :num-rounds 5
                          :valid-sets (list (cons "valid" valid-set))
                          :parameters booster-parameters
                          :evaluation #'my-logloss)
        (unwind-protect
             (dolist (series (cl-gbdt:training-report-series report))
               (format t "~A series: index=~S metric=~S last=~S~%"
                       name (cl-gbdt:training-series-index series)
                       (cl-gbdt:training-series-metric series)
                       (aref (cl-gbdt:training-series-values series) 4)))
          (cl-gbdt:free-booster booster))))))

(let ((lgbm (cl-gbdt:open-backend :lightgbm))
      (xgb (cl-gbdt:open-backend :xgboost)))
  (show-custom-evaluation "LightGBM" lgbm
        '(:parameters (:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1))
        '(:objective "binary" :num-leaves 2 :min-data-in-leaf 1 :min-data-in-bin 1
          :verbose -1 :metric "binary_logloss")
        t)
  (show-custom-evaluation "XGBoost " xgb '()
        '(:objective "binary:logistic" :max-depth 2 :eta 0.5 :verbosity 0
          :eval-metric "logloss")
        nil)
  (cl-gbdt:close-backend lgbm)
  (cl-gbdt:close-backend xgb))
```

Output:

```
LightGBM series: index=0 metric="binary_logloss" last=0.35374722486733523d0
LightGBM series: index=1 metric="binary_logloss" last=0.35374722486733523d0
LightGBM series: index=0 metric="my_logloss" last=0.35374722486733523d0
LightGBM series: index=1 metric="my_logloss" last=0.35374722486733523d0
XGBoost  series: index=0 metric="logloss" last=0.4740770012140274d0
XGBoost  series: index=1 metric="logloss" last=0.4740770012140274d0
XGBoost  series: index=0 metric="my_logloss" last=0.47407697467999527d0
XGBoost  series: index=1 metric="my_logloss" last=0.47407697467999527d0
```

`MY-LOGLOSS` above reimplements the same binary log loss both libraries already compute, and
its series lands on each backend's own to the last few digits -- not because the two are
forced to agree, but because a caller-written metric over the same probabilities the library
scored really does compute the same number. It also shows the ordering `training-report-series`
holds to: the two `training-series` for `"binary_logloss"`/`"logloss"` -- one per dataset,
library metrics first -- come before either of `"my_logloss"`'s, on both backends. `train`'s
generic docstring states this as a guarantee rather than an accident of this example: the
library's own series are exactly what `evaluation` already reports, in the same order, and
`:evaluation`'s own entries are APPENDED after every one of them, so they form a PREFIX of
`training-report-series` (`src/protocol.lisp`). The append itself happens once per
backend, in `%custom-evaluation-entries`
(`src/lightgbm/protocol.lisp`/`src/xgboost/protocol.lisp`), and
`training-report-from-history` preserves first-seen order rather than sorting anything
(`src/training/history.lisp`), which is what turns "appended last" into "prefix" once
the whole run's history is folded. `evaluation` itself never reports a custom metric -- it
asks the library what the library computed, and the library never computed this one
(`src/protocol.lisp`).

`:evaluation` is called with **two arguments**: SCORES, that dataset's current predictions as
a `(ROWS GROUPS)` `double-float` array, and the dataset's **INDEX** -- `0` for the training
set, `N+1` for the Nth `:valid-sets` entry, the same numbering `:early-stopping`'s `:dataset`
key and `evaluation`'s own `DATASET-INDEX` already use
(`src/protocol.lisp`). It must return **two values**, a metric NAME (a string) and a
VALUE (a real or `NIL`); a NAME that is not a string, or a VALUE that is neither, signals
`unsupported-argument` (`custom-metric-entry` in `src/training/custom-metric.lisp`).
A real VALUE is **recorded as a `double-float`**, coerced where the entry is built rather than
stored as returned: `training-series-values` documents every element of every series as a
`double-float` or `NIL`, and both libraries' own values already are doubles, so a caller
returning `1/3` reads `0.3333333333333333d0` back out of its own series rather than a `ratio`
landing in a slot every other consumer was promised held doubles (same file). A real too large
for a `double-float` to hold records the signed infinity, identically on every platform: the
coercion is wrapped the same way `%rational-json` (`src/config/missing-value.lisp`) wraps its
own, because whether `coerce` *signals* on such a value is a property of the platform's
floating-point traps rather than of the value -- see that function's docstring, which records
the split.
`NIL` means "not computable this iteration" -- a fold whose
denominator was zero, a metric undefined before some minimum number of rows -- and is
recorded in its place in the series rather than dropped, counting as no improvement to an
`:early-stopping` watcher exactly as a value the backend itself could not report does
(`src/protocol.lisp`).

SCORES is what `predict :kind :normal` returns for that dataset, and **not** the margin
`:objective` is handed -- with a classification objective configured, these are the
transformed probabilities. INDEX is not decorative: `predict :kind :normal` on the trained
booster, and the array `:evaluation` was handed for that same dataset during the run, are the
identical array, checked below for both the training set (index 0) and the one `:valid-sets`
entry (index 1):

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified) :silent t)

(defparameter *ce-matrix*
  (make-array '(8 2) :element-type 'double-float
              :initial-contents '((0.0d0 0.0d0) (0.0d0 1.0d0) (0.0d0 2.0d0)
                                   (0.0d0 3.0d0) (5.0d0 0.0d0) (5.0d0 1.0d0)
                                   (5.0d0 2.0d0) (5.0d0 3.0d0))))
(defparameter *ce-label*
  (make-array 8 :element-type 'single-float
              :initial-contents '(0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0)))

(let ((lgbm (cl-gbdt:open-backend :lightgbm)))
  (unwind-protect
      (cl-gbdt:with-dataset
          (train-set (cl-gbdt:make-dataset lgbm *ce-matrix* :label *ce-label*
                       :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1)))
        (cl-gbdt:with-dataset
            (valid-set (cl-gbdt:make-dataset lgbm *ce-matrix* :label *ce-label*
                         :reference train-set
                         :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1)))
          (let ((last-scores (make-array 2 :initial-element nil)))
            (cl-gbdt:with-booster
                (booster (cl-gbdt:train lgbm train-set :num-rounds 5
                           :valid-sets (list valid-set)
                           :parameters '(:objective "binary" :num-leaves 2 :min-data-in-leaf 1
                                         :min-data-in-bin 1 :verbose -1)
                           ;; INDEX is 0 for the training set, 1 for the first (and only)
                           ;; :valid-sets entry -- the same numbering :early-stopping's
                           ;; :dataset key already uses.
                           :evaluation (lambda (scores index)
                                         (setf (aref last-scores index) scores)
                                         (values "captured" 0.0d0))))
              (format t "index 0's SCORES is predict :kind :normal's: ~S~%"
                      (equalp (aref last-scores 0)
                              (cl-gbdt:predict booster *ce-matrix* :kind :normal)))
              (format t "index 1's SCORES is predict :kind :normal's: ~S~%"
                      (equalp (aref last-scores 1)
                              (cl-gbdt:predict booster *ce-matrix* :kind :normal)))
              (format t "index 0's SCORES is NOT predict :kind :raw's: ~S~%"
                      (not (equalp (aref last-scores 0)
                                   (cl-gbdt:predict booster *ce-matrix* :kind :raw))))))))
    (cl-gbdt:close-backend lgbm)))
```

Output:

```
index 0's SCORES is predict :kind :normal's: T
index 1's SCORES is predict :kind :normal's: T
index 0's SCORES is NOT predict :kind :raw's: T
```

This is measured differently on each backend, and the two measurements are not the same
kind of fact: on LightGBM, `%booster-predictions` reads `LGBM_BoosterGetPredict`, a value the
library already holds rather than a fresh prediction, so agreeing with `predict` says two
different C functions agree -- measured on both a 40-row training set and a 17-row
validation set to `0.0`, and `0.706` away from `:raw` under `objective=binary`
(`src/lightgbm/native.lisp`). On XGBoost, `%booster-predictions` runs a fresh
`XGBoosterPredictFromDMatrix` prediction pass over that dataset's own `DMatrix`, the same
call `predict` itself makes, so agreeing says that `DMatrix` and a fresh one built from the
same rows answer alike -- also measured to `0.0` on both datasets, and `0.756` away from
`:raw` under `binary:logistic` after five iterations
(`src/xgboost/native.lisp`). Neither figure stands in for the other's,
and the two are not compared -- policy section 13; what they share is only that both
backends' SCORES equal `predict :kind :normal`'s. Under a custom `:objective` the two then
part company exactly as [Custom objective](#custom-objective) already describes: LightGBM
forces `objective=none`, so `:normal` and `:raw` coincide there and SCORES equals what
`:objective` was handed, while XGBoost rewrites nothing, so a configured `binary:logistic`
keeps transforming and `:evaluation` reads probabilities while `:objective` reads the margin
behind them, in the same run.

A custom metric's values become **series of their own** in the report, one per (INDEX, NAME)
pair, indistinguishable in shape from the library's own -- so `:early-stopping` can watch one
with nothing extra arranged for it, by giving `:metric` the name the function returns and
`:dataset` the index it was returned for:

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified) :silent t)

(defparameter *ce-matrix*
  (make-array '(8 2) :element-type 'double-float
              :initial-contents '((0.0d0 0.0d0) (0.0d0 1.0d0) (0.0d0 2.0d0)
                                   (0.0d0 3.0d0) (5.0d0 0.0d0) (5.0d0 1.0d0)
                                   (5.0d0 2.0d0) (5.0d0 3.0d0))))
(defparameter *ce-label*
  (make-array 8 :element-type 'single-float
              :initial-contents '(0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0)))

(defun stalling-metric ()
  "3.0, 2.0, 1.0, 1.0, 1.0, ... -- ignores SCORES entirely. What is under test is that a
caller's own metric name reaches :early-stopping at all, not that a real model produced it."
  (let ((calls 0))
    (lambda (scores index)
      (declare (ignore scores index))
      (incf calls)
      (values "stalls" (coerce (max 1 (- 4 calls)) 'double-float)))))

(let ((lgbm (cl-gbdt:open-backend :lightgbm)))
  (unwind-protect
      (cl-gbdt:with-dataset
          (train-set (cl-gbdt:make-dataset lgbm *ce-matrix* :label *ce-label*
                       :parameters '(:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1)))
        (multiple-value-bind (booster report)
            (cl-gbdt:train lgbm train-set :num-rounds 10
                            ;; The library's own metric turned off, so the only series in the
                            ;; report -- and the only thing :early-stopping could be watching --
                            ;; is the caller's own.
                            :parameters '(:objective "binary" :num-leaves 2 :min-data-in-leaf 1
                                          :min-data-in-bin 1 :verbose -1 :metric "none")
                            :evaluation (stalling-metric)
                            :early-stopping (list :metric "stalls" :dataset 0
                                                   :direction :lower-is-better :rounds 2))
          (cl-gbdt:free-booster booster)
          (format t "ran ~S of 10 rounds~%" (cl-gbdt:training-report-num-rounds report))
          (format t "early-stopped-p: ~S~%" (cl-gbdt:training-report-early-stopped-p report))
          (format t "best-iteration: ~S~%" (cl-gbdt:training-report-best-iteration report))
          (format t "best-score: ~S~%" (cl-gbdt:training-report-best-score report))))
    (cl-gbdt:close-backend lgbm)))
```

Output:

```
ran 5 of 10 rounds
early-stopped-p: T
best-iteration: 3
best-score: 1.0d0
```

The value improves at iterations 1, 2 and 3 and then holds at `1.0`; improvement is strict, so
a plateau does not count, and two consecutive non-improving iterations (`:rounds 2`) stop the
run at iteration 5 with iteration 3 recorded best -- driven entirely by a metric that never
reads its SCORES argument, which is the point: what reaches the watcher is the (INDEX, NAME)
pair `:evaluation` returned, the same mechanism a library metric reaches it through, not
anything specific to `:evaluation`.

`train` refuses a non-`NIL` `:evaluation` in two more shapes, both checked before any foreign
call. `:record-history nil` together with `:evaluation` signals `unsupported-argument`: a
custom metric's whole result is the per-iteration series `:record-history nil` exists not to
build, so the values would be computed at full cost and then dropped -- the same
contradiction `:early-stopping` and `:record-history nil` already make. And `:evaluation`
must be a `function`; a number, a string, or a **symbol** naming a real function of the right
arity all signal `unsupported-argument` naming `:evaluation` -- the symbol deliberately,
since `funcall` would have accepted it happily and resolved it afresh each iteration against
whatever global definition happened to be in force, rather than against what the caller
passed:

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

(defparameter *ce-matrix*
  (make-array '(8 2) :element-type 'double-float
              :initial-contents '((0.0d0 0.0d0) (0.0d0 1.0d0) (0.0d0 2.0d0)
                                   (0.0d0 3.0d0) (5.0d0 0.0d0) (5.0d0 1.0d0)
                                   (5.0d0 2.0d0) (5.0d0 3.0d0))))
(defparameter *ce-label*
  (make-array 8 :element-type 'single-float
              :initial-contents '(0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0)))

(defparameter *ce-booster-parameters*
  '((:lightgbm :objective "binary" :num-leaves 2 :min-data-in-leaf 1 :min-data-in-bin 1
     :verbose -1)
    (:xgboost :objective "binary:logistic" :max-depth 2 :eta 0.5 :verbosity 0)))
(defparameter *ce-dataset-parameters*
  '((:lightgbm :parameters (:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1))
    (:xgboost)))

(defun a-constant-metric (scores index)
  "A SYMBOL naming a real function of the right arity, so `funcall' would have accepted it
happily -- which is exactly why :evaluation refuses it explicitly rather than leaving the
mistake to surface some other way."
  (declare (ignore scores index))
  (values "constant" 0.5d0))

(dolist (name '(:lightgbm :xgboost))
  (let ((backend (cl-gbdt:open-backend name)))
    (unwind-protect
        (cl-gbdt:with-dataset
            (dataset (apply #'cl-gbdt:make-dataset backend *ce-matrix* :label *ce-label*
                             (cdr (assoc name *ce-dataset-parameters*))))
          (format t "~A :evaluation with :record-history nil:~%" name)
          (handler-case
              (cl-gbdt:free-booster
               (cl-gbdt:train backend dataset :num-rounds 3 :record-history nil
                               :parameters (cdr (assoc name *ce-booster-parameters*))
                               :evaluation (lambda (scores index)
                                             (declare (ignore scores index))
                                             (values "x" 0.0d0))))
            (error (c) (format t "  SIGNALED ~A: ~A~%" (type-of c) c)))
          (dolist (value (list 42 'a-constant-metric))
            (format t "~A :evaluation ~S:~%" name value)
            (handler-case
                (cl-gbdt:free-booster
                 (cl-gbdt:train backend dataset :num-rounds 3
                                 :parameters (cdr (assoc name *ce-booster-parameters*))
                                 :evaluation value))
              (error (c) (format t "  SIGNALED ~A: ~A~%" (type-of c) c)))))
      (cl-gbdt:close-backend backend))))
```

Output:

```
LIGHTGBM :evaluation with :record-history nil:
  SIGNALED UNSUPPORTED-ARGUMENT: train's :evaluation is not supported by LIGHTGBM: a custom metric is recorded per iteration, which :record-history NIL skips; pass :record-history T, or drop :evaluation.
LIGHTGBM :evaluation 42:
  SIGNALED UNSUPPORTED-ARGUMENT: train's :evaluation is not supported by LIGHTGBM: the custom metric must be a function of two arguments, or NIL for the library's own metrics only -- got 42.
LIGHTGBM :evaluation A-CONSTANT-METRIC:
  SIGNALED UNSUPPORTED-ARGUMENT: train's :evaluation is not supported by LIGHTGBM: the custom metric must be a function of two arguments, or NIL for the library's own metrics only -- got A-CONSTANT-METRIC.
XGBOOST :evaluation with :record-history nil:
  SIGNALED UNSUPPORTED-ARGUMENT: train's :evaluation is not supported by XGBOOST: a custom metric is recorded per iteration, which :record-history NIL skips; pass :record-history T, or drop :evaluation.
XGBOOST :evaluation 42:
  SIGNALED UNSUPPORTED-ARGUMENT: train's :evaluation is not supported by XGBOOST: the custom metric must be a function of two arguments, or NIL for the library's own metrics only -- got 42.
XGBOOST :evaluation A-CONSTANT-METRIC:
  SIGNALED UNSUPPORTED-ARGUMENT: train's :evaluation is not supported by XGBOOST: the custom metric must be a function of two arguments, or NIL for the library's own metrics only -- got A-CONSTANT-METRIC.
```

Both checks live in each backend's own `%check-custom-evaluation`
(`src/lightgbm/protocol.lisp`/`src/xgboost/protocol.lisp`), ahead of the capability
check that runs first; identical wording on both backends because both call the same
backend-neutral checks underneath.

A NAME colliding with one the library itself reports for the **same** dataset index signals
`unsupported-argument` too -- checked at the end of the first iteration, the first moment
there is a real evaluation to compare against. The pair (INDEX, NAME) is what a series is
keyed by, so two different quantities under one pair would corrupt the series rather than
produce two; what is compared is what this booster **actually reported**, not a list of
well-known metric names, which is why the identical name is accepted the moment the library
reports no metric at all:

```lisp
(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

(defparameter *ce-matrix*
  (make-array '(8 2) :element-type 'double-float
              :initial-contents '((0.0d0 0.0d0) (0.0d0 1.0d0) (0.0d0 2.0d0)
                                   (0.0d0 3.0d0) (5.0d0 0.0d0) (5.0d0 1.0d0)
                                   (5.0d0 2.0d0) (5.0d0 3.0d0))))
(defparameter *ce-label*
  (make-array 8 :element-type 'single-float
              :initial-contents '(0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0)))

(defparameter *ce-library-metric-names*
  '((:lightgbm . "binary_logloss") (:xgboost . "logloss")))

;; One booster parameter plist per backend WITH the library's metric on, and one WITHOUT --
;; LightGBM's own "metric none", XGBoost's own "disable_default_eval_metric 1".
(defparameter *ce-with-metric*
  '((:lightgbm :objective "binary" :num-leaves 2 :min-data-in-leaf 1 :min-data-in-bin 1
     :verbose -1 :metric "binary_logloss")
    (:xgboost :objective "binary:logistic" :max-depth 2 :eta 0.5 :verbosity 0
     :eval-metric "logloss")))
(defparameter *ce-without-metric*
  '((:lightgbm :objective "binary" :num-leaves 2 :min-data-in-leaf 1 :min-data-in-bin 1
     :verbose -1 :metric "none")
    (:xgboost :objective "binary:logistic" :max-depth 2 :eta 0.5 :verbosity 0
     :disable-default-eval-metric 1)))
(defparameter *ce-dataset-parameters*
  '((:lightgbm :parameters (:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1))
    (:xgboost)))

(dolist (name '(:lightgbm :xgboost))
  (let ((backend (cl-gbdt:open-backend name))
        (library-name (cdr (assoc name *ce-library-metric-names*))))
    (unwind-protect
        (cl-gbdt:with-dataset
            (dataset (apply #'cl-gbdt:make-dataset backend *ce-matrix* :label *ce-label*
                             (cdr (assoc name *ce-dataset-parameters*))))
          (flet ((train-named (parameters)
                   (cl-gbdt:train backend dataset :num-rounds 3 :parameters parameters
                                   :evaluation (lambda (scores index)
                                                 (declare (ignore scores index))
                                                 (values library-name 0.5d0)))))
            (format t "~A :evaluation returns ~S, the library's own name, while it is ~
                       configured:~%" name library-name)
            (handler-case
                (cl-gbdt:free-booster
                 (train-named (cdr (assoc name *ce-with-metric*))))
              (error (c) (format t "  SIGNALED ~A: ~A~%" (type-of c) c)))
            (format t "~A :evaluation returns ~S while the library reports no metric at ~
                       all:~%" name library-name)
            (multiple-value-bind (booster report)
                (train-named (cdr (assoc name *ce-without-metric*)))
              (cl-gbdt:free-booster booster)
              (format t "  accepted; series pairs: ~S~%"
                      (mapcar (lambda (series)
                                (cons (cl-gbdt:training-series-index series)
                                      (cl-gbdt:training-series-metric series)))
                              (cl-gbdt:training-report-series report))))))
      (cl-gbdt:close-backend backend))))
```

Output:

```
LIGHTGBM :evaluation returns "binary_logloss", the library's own name, while it is configured:
  SIGNALED UNSUPPORTED-ARGUMENT: train's :evaluation is not supported by LIGHTGBM: "binary_logloss" already names a metric the library reports for dataset index 0.
LIGHTGBM :evaluation returns "binary_logloss" while the library reports no metric at all:
  accepted; series pairs: ((0 . "binary_logloss"))
XGBOOST :evaluation returns "logloss", the library's own name, while it is configured:
  SIGNALED UNSUPPORTED-ARGUMENT: train's :evaluation is not supported by XGBOOST: "logloss" already names a metric the library reports for dataset index 0.
XGBOOST :evaluation returns "logloss" while the library reports no metric at all:
  accepted; series pairs: ((0 . "logloss"))
```

The check itself is backend-neutral, `check-metric-name-collision`
(`src/training/custom-metric.lisp`), given each iteration's own library entries to
compare against rather than a static list. The same name at a **different** index does not
collide -- not constructible against either vendored library, since both report the same
metric list for every dataset they retain, so this project's own assertion of that half of
the keying is at layer 1, over a written entry list rather than a measured one
(`check-metric-name-collision-allows-a-name-the-library-uses-elsewhere` in
`tests/custom-metric.lisp`).

`:evaluation` must return **the same name for a given index on every iteration** of the run.
Two indices may return two different names; what is refused is one index's name changing
between iterations. The first iteration's name is remembered per index, and a later iteration
returning a different one for that index signals `unsupported-argument` naming `:evaluation`,
mid-run and at the very call that changed it:

```
train's :evaluation is not supported by XGBOOST: the custom metric returned name
"another_metric" for dataset index 0 after returning "my_logloss" for it; one name per dataset
index is required for the whole run.
```

This is a requirement rather than a nicety, and it is what keeps `train`'s promise that
**every series is exactly `training-report-num-rounds` long** true of a caller's own series as
well as the library's. A series holds one value per completed iteration, keyed by the (INDEX,
NAME) pair, so a name that varies asks for something no series can be:

- Varying **without ever colliding** gives one series per name it took, each pushed only on
  the iterations that name appeared in -- several series, all shorter than the run, each
  misaligned with the iterations its values were measured at.
- Varying **into the library's own name** for that index is the case the collision check
  above cannot reach, since that check runs on the first iteration only: a caller returning a
  safe name then and a colliding one afterwards walks straight past it. From the iteration the
  two names meet, one key collects two values per iteration and its series comes out
  `1 + 2(N-1)` values long over N rounds -- *longer* than `training-report-num-rounds` says the
  run was, and silently. It would also break the "at most one entry matches a given (index,
  metric) pair" invariant `:early-stopping` reads under: `find-if` returns the first of the
  two, so a watcher would read one value per iteration and never learn the other existed
  (`%find-watched-entry` in `src/training/early-stopping.lisp`).

Pinning the name closes both, and closes the second without a second collision check: once
each index's name is fixed at the first iteration, the only name that can ever reach the
library's is the one the collision check already compared. Like the collision check it is
backend-neutral -- `make-metric-name-pin` and `pin-metric-name`
(`src/training/custom-metric.lisp`), one pin per `train` call.

Returning **one string object and rewriting it in place** is refused on exactly the same
terms, and it is not something the pin could have caught by itself. A string is mutable, so a
caller keeping one name buffer and refilling it each iteration would have handed the pin an
object that compares `string=` with itself however its characters changed -- and every
recorded entry would have held that same object, to be read once, at the end of the run, under
whatever the name said by then. Measured before this was closed, four rounds on LightGBM with
`metric "binary_logloss"` and a 14-character name rewritten to `"binary_logloss"` from the
second iteration: `train` returned normally, nothing signalled, and the report held a single
**eight-value series for a four-round run**. What closes it is that `custom-metric-entry`
`copy-seq`s the name into the entry it builds, and both `train` methods take the name back
**out of that entry** for the collision check and the pin -- so the history, the pin and the
collision check all hold one snapshot and the caller's own object reaches none of them
(`src/training/custom-metric.lisp`,
`%custom-evaluation-entries` in `src/lightgbm/protocol.lisp`/`src/xgboost/protocol.lisp`).

A NAME that is not a string at all, or a VALUE that is neither a real nor `NIL`, is refused
the same way and at the same point in the run -- `custom-metric-entry`, in that same file --
so `(values :my-metric 0.25d0)` and `(values "my_logloss" "0.25")` each signal
`unsupported-argument` naming `:evaluation` rather than reaching the report.

`:evaluation` runs inside `train`'s own floating-point-trap mask, on the same terms
`:objective` does (see [Custom objective](#custom-objective)): the caller's own arithmetic
does not trap, so `(/ 1.0d0 0.0d0)` yields infinity rather than signalling
`division-by-zero`, on x86-64 as well as on aarch64. A handle it frees, or a backend it
closes, is caught the moment it returns and before the next dataset's predictions are read --
`%custom-evaluation-entries` re-checks between two consecutive datasets rather than once per
iteration, which is what makes freeing a `:valid-sets` entry from inside the FIRST dataset's
call signal `released-handle-error` naming that dataset rather than faulting the process on
the second dataset's read.

### The measured cost

A custom metric adds a `predict :kind :normal`-shaped array read plus a Lisp call, per
dataset per iteration, on top of what `:record-history` already reads. Measured the same way
as [that section](training.md#turning-recording-off-record-history) -- 2000 rows x 20 columns, one
validation set, `:record-history t` in **both** arms so the only difference is whether
`:evaluation` is supplied:

```lisp
;;;; Same shape of fixture as the :record-history measurement above: 2000 rows x 20 columns,
;;;; 1500 rounds, one validation set. RECORD-HISTORY is T in both arms, so the only difference
;;;; between the two is whether :evaluation is supplied.

(ql:quickload '(:cl-gbdt :cl-gbdt/lightgbm/unified :cl-gbdt/xgboost/unified) :silent t)

(defparameter *rows* 2000)
(defparameter *valid-rows* 500)
(defparameter *columns* 20)
(defparameter *rounds* 1500)

(defun make-fixture-matrix (rows columns offset)
  (let ((matrix (make-array (list rows columns) :element-type 'double-float)))
    (dotimes (row rows matrix)
      (dotimes (col columns)
        (setf (aref matrix row col)
              (coerce (mod (+ (* 7 (+ row offset)) (* 13 col)) 97) 'double-float))))))

(defun make-fixture-label (rows offset)
  (let ((label (make-array rows :element-type 'single-float)))
    (dotimes (row rows label)
      (setf (aref label row) (if (evenp (+ row offset)) 1.0 0.0)))))

(defparameter *train-matrix* (make-fixture-matrix *rows* *columns* 0))
(defparameter *train-label* (make-fixture-label *rows* 0))
(defparameter *valid-matrix* (make-fixture-matrix *valid-rows* *columns* 5))
(defparameter *valid-label* (make-fixture-label *valid-rows* 5))

(defun my-metric (scores index)
  "A representative caller-written metric: mean log loss over SCORES against this run's own
label vector for dataset INDEX -- real arithmetic over every row, not a constant, so the
measurement includes a Lisp-side cost proportional to row count and not just the array read."
  (let* ((labels* (if (zerop index) *train-label* *valid-label*))
         (rows (array-dimension scores 0))
         (sum 0d0))
    (dotimes (row rows)
      (let ((p (min (max (aref scores row 0) 1d-15) (- 1d0 1d-15)))
            (y (coerce (aref labels* row) 'double-float)))
        (incf sum (- (+ (* y (log p)) (* (- 1d0 y) (log (- 1d0 p))))))))
    (values "my_logloss" (/ sum rows))))

(defun run-once (backend-name make-dataset-parameters booster-parameters reference-p
                  evaluation-p)
  "Train once and return the wall-clock seconds, :record-history T throughout. EVALUATION-P T
supplies :EVALUATION #'MY-METRIC; NIL supplies none."
  (let ((backend (cl-gbdt:open-backend backend-name)))
    (unwind-protect
         (cl-gbdt:with-dataset
             (train-set (apply #'cl-gbdt:make-dataset backend *train-matrix*
                                :label *train-label* make-dataset-parameters))
           (cl-gbdt:with-dataset
               (valid-set (apply #'cl-gbdt:make-dataset backend *valid-matrix*
                                  :label *valid-label*
                                  (append (when reference-p (list :reference train-set))
                                          make-dataset-parameters)))
             (let ((start (get-internal-real-time)))
               (multiple-value-bind (booster report)
                   (apply #'cl-gbdt:train backend train-set
                          :valid-sets (list valid-set) :num-rounds *rounds*
                          :record-history t :parameters booster-parameters
                          (when evaluation-p (list :evaluation #'my-metric)))
                 (declare (ignore report))
                 (cl-gbdt:free-booster booster))
               (/ (float (- (get-internal-real-time) start) 1d0)
                  internal-time-units-per-second))))
      (cl-gbdt:close-backend backend))))

(defun report-timing (backend-name make-dataset-parameters booster-parameters reference-p)
  ;; One untimed warm-up run per arm first, then 5 timed runs each, interleaved
  ;; WITHOUT/WITH/WITHOUT/... so neither arm is systematically first or last.
  (run-once backend-name make-dataset-parameters booster-parameters reference-p nil)
  (run-once backend-name make-dataset-parameters booster-parameters reference-p t)
  (let ((without '()) (with '()))
    (dotimes (i 5)
      (push (run-once backend-name make-dataset-parameters booster-parameters reference-p nil)
            without)
      (push (run-once backend-name make-dataset-parameters booster-parameters reference-p t)
            with))
    (setf without (nreverse without) with (nreverse with))
    (format t "~A without :evaluation: ~{~,3F~^ ~} seconds (mean ~,3F)~%"
            backend-name without (/ (reduce #'+ without) (length without)))
    (format t "~A with    :evaluation: ~{~,3F~^ ~} seconds (mean ~,3F)~%"
            backend-name with (/ (reduce #'+ with) (length with)))
    (format t "~A ratio (with / without): ~,3F~%"
            backend-name (/ (/ (reduce #'+ with) (length with))
                             (/ (reduce #'+ without) (length without))))))

(report-timing :lightgbm
               '(:parameters (:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1))
               '(:objective "binary" :num-leaves 31 :verbose -1 :metric "binary_logloss,auc"
                 :min-data-in-leaf 1 :min-data-in-bin 1)
               t)

(report-timing :xgboost
               '()
               '(:objective "binary:logistic" :max-depth 6 :eta 0.3d0 :verbosity 0
                 :eval-metric "logloss" :eval-metric "error")
               nil)
```

Output, two independent runs:

```
LIGHTGBM without :evaluation: 2.614 2.654 2.783 2.467 2.668 seconds (mean 2.637)
LIGHTGBM with    :evaluation: 2.946 3.053 2.790 2.946 3.025 seconds (mean 2.952)
LIGHTGBM ratio (with / without): 1.119
XGBOOST without :evaluation: 0.353 0.643 0.445 0.555 0.536 seconds (mean 0.506)
XGBOOST with    :evaluation: 0.703 0.697 0.700 0.818 0.830 seconds (mean 0.750)
XGBOOST ratio (with / without): 1.480
```

```
LIGHTGBM without :evaluation: 2.458 2.542 2.466 2.646 2.552 seconds (mean 2.533)
LIGHTGBM with    :evaluation: 3.062 2.899 2.811 2.983 2.831 seconds (mean 2.917)
LIGHTGBM ratio (with / without): 1.152
XGBOOST without :evaluation: 0.535 0.551 0.492 0.442 0.609 seconds (mean 0.526)
XGBOOST with    :evaluation: 0.720 0.764 0.859 0.875 0.760 seconds (mean 0.796)
XGBOOST ratio (with / without): 1.513
```

`:evaluation` added roughly **12-15%** to LightGBM's wall-clock `train` time here, and
roughly **48-51%** to XGBoost's. The two are each this backend's own ratio and are not
compared with one another -- policy section 13 -- and the gap between them is explained by
the same mechanism the SCORES paragraph above already measured: LightGBM's per-dataset read
is a cached value the booster already holds, so the added cost is close to the Lisp call and
the array copy alone, while XGBoost's is a whole extra `XGBoosterPredictFromDMatrix` pass per
dataset per iteration -- a real prediction, not a cached read, which is the more expensive of
the two operations on either backend. Treat these as orders of magnitude on one machine, not
precise figures -- run-to-run variance on the same code was as wide as 1.12-1.15 for LightGBM and
1.48-1.51 for XGBoost across the two runs above, and an earlier pair of runs at 500 rounds
(a fifth of the round count, and so closer to the noise floor of process startup and dataset
construction) ranged 1.08-1.15 for LightGBM and 1.20-2.27 for XGBoost.

`:custom-evaluation` is answerable through `backend-supports-p` on both backends, but the two
true answers come out of different lists for a reason that is a fact about the two
*libraries* rather than a difference a caller of `:evaluation` can see: LightGBM's
per-dataset read needs three C functions, none of them in that backend's required set, so it
is PROBED like `:custom-objective`; XGBoost's needs one, which IS required there, so it is
DECLARED (`src/backend.lisp`). Every check `:evaluation` is put through, and
every error it can signal, is identical prose on both backends, as the refusal output above
shows -- so this is not a row in
[the differences table](backend-differences.md#where-the-two-backends-genuinely-differ):
there is nothing here a caller's code, as opposed to a reader of `backend-info`'s probed
plist, can tell apart.
