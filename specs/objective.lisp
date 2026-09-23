;;;; objective.lisp --- Executable contracts for `train''s custom-objective helpers.
;;;;
;;;; `objective-single-float' has two outcomes, a `single-float' for a real and
;;;; `unsupported-element-type' for anything else, stated as two exclusive cases over any
;;;; object -- save a real beyond single-float range, whose outcome depends on the ambient
;;;; floating-point traps (see `within-single-range-p').
;;;;
;;;; `objective-parameters' is where the enumerated-not-prefix-matched rule lives: `app' is a
;;;; LightGBM alias for `objective' and `apps' is not. The alias list below is restated from
;;;; that function's `*objective-parameter-names*' docstring -- measured against the vendored
;;;; library -- rather than imported, so that editing the variable is caught here instead of
;;;; silently redefining the contract. The generated keys mix true aliases, in keyword and
;;;; string spellings and cases, with near misses that must survive.

(uiop:define-package #:cl-gbdt/specs/objective
  (:use #:cl)
  (:import-from #:cl-spec/main
                #:defgenerator
                #:defproperty
                #:defspec
                #:defspec-function)
  (:import-from #:cl-gbdt/src/conditions
                #:unsupported-element-type)
  (:import-from #:cl-gbdt/src/config/objective
                #:objective-parameters
                #:objective-single-float)
  (:import-from #:cl-gbdt/specs/values
                #:draw-finite-double
                #:pairs-plist
                #:parameter-value
                #:plist-pairs)
  (:export #:objective-parameters-ends-with-the-one-canonical-objective
           #:objective-parameters-keeps-every-other-entry-in-order
           #:objective-parameters-is-idempotent))

(in-package #:cl-gbdt/specs/objective)

(defparameter *objective-aliases*
  '("objective" "objective_type" "app" "application" "loss")
  "The five spellings LightGBM 4.7.0 honours for `objective', as a parameter string names
them. Restated, not imported -- see this file's header.")

(defun objective-alias-p (key)
  "True when KEY, a keyword or string, names LightGBM's `objective' parameter."
  (member (substitute #\_ #\- (string-downcase (string key))) *objective-aliases*
          :test #'string=))

(defun within-single-rounding-p (result value)
  "True when the single-float RESULT is within one single-float rounding of the real VALUE.

One rounding is a relative error of at most `single-float-epsilon' for a normal result, and at
most the subnormal spacing, `least-positive-single-float', below that -- where a relative
bound alone would fail: 1d-40 becomes 9.999946e-41. Compared as rationals, so the check itself
does no floating-point arithmetic that could trap."
  (<= (abs (- (rational result) (rational value)))
      (max (* (abs (rational value)) (rational single-float-epsilon))
           (rational least-positive-single-float))))

(defun within-single-range-p (value)
  "True when VALUE is not a real, or is a real no larger in magnitude than
`most-positive-single-float'.

That is the portable guarantee `objective-single-float''s own docstring states. A real beyond
it has no single-float to become, and what `coerce' then does depends on the floating-point
traps in force -- `floating-point-overflow' where SBCL enables the trap (x86-64), an infinity
where it does not (aarch64, and inside `train''s own masked foreign call) -- so the contract,
like the docstring, promises nothing there."
  (or (not (realp value))
      (<= (abs (rational value)) (rational most-positive-single-float))))

(defgenerator objective-single-float-arguments ()
  (list (case (random 12)
          (0 (- (random 2000001) 1000000))
          ((1 2) (draw-finite-double))
          (3 (- (random 2000.0) 1000.0))
          (4 (/ (1+ (random 1000)) (+ 2 (random 999))))
          ;; A subnormal single, and the largest single, as doubles: the two ends of the range.
          (5 (* (1+ (random 1000)) 1d-42))
          (6 (if (zerop (random 2))
                 (coerce most-positive-single-float 'double-float)
                 (coerce most-negative-single-float 'double-float)))
          (7 (format nil "~D" (random 100)))
          (8 (zerop (random 2)))
          (9 :gradient)
          (10 #\x)
          (t (list (random 10))))))

(defspec-function objective-single-float
  "A real within single-float range becomes the nearest single-float; anything that is not a
real is refused."
  ;; Any object at all. The generator samples every branch, both ends of the single-float
  ;; range included; that bounds what is tried, not what is promised.
  (:args (value t))
  (:args-generator objective-single-float-arguments)
  (:pre (within-single-range-p value))
  (:cases
   (:real
    (:when (realp value))
    (:returns single-float)
    (:post (within-single-rounding-p result value)))
   (:not-real
    (:when (not (realp value)))
    (:signals (type unsupported-element-type)))))

;;; A generation domain for the Properties below, which only ever see generated keys. `member'
;;; compares with EQL, so as a validator it would refuse a freshly made "objective" string;
;;; it is not used to validate a caller's data anywhere.
(defspec objective-key
  (member :objective :objective-type :app :application :loss
          "objective" "OBJECTIVE" "objective_type" "App"
          :apps :obj :objectives :objective-function :loss-function :app-type
          :applications :losses "apps" "obj"
          :num-class :num-leaves :learning-rate :metric))

;;; `(and (type list) ...)' because a `tuple' alone admits a vector, and `pairs-plist'
;;; destructures lists.
(defspec objective-pairs
  (list-of (and (type list) (tuple objective-key parameter-value)) :max-length 8))

(defproperty objective-parameters-ends-with-the-one-canonical-objective
    ((pairs objective-pairs))
  "Exactly one objective spelling survives, and it is the appended :objective \"none\"."
  (:about objective-parameters)
  (:kind :invariant)
  (:trials (:smoke 20 :normal 200))
  (let ((result (plist-pairs (objective-parameters (pairs-plist pairs)))))
    (and (equal (car (last result)) '(:objective "none"))
         (= 1 (count-if (lambda (pair) (objective-alias-p (first pair))) result)))))

(defproperty objective-parameters-keeps-every-other-entry-in-order
    ((pairs objective-pairs))
  "Every entry that is not an objective spelling -- near misses included -- survives with its
value and in its order."
  (:about objective-parameters)
  (:kind :invariant)
  (:trials (:smoke 20 :normal 200))
  (let ((result (plist-pairs (objective-parameters (pairs-plist pairs)))))
    (equal (butlast result)
           (remove-if (lambda (pair) (objective-alias-p (first pair))) pairs))))

(defproperty objective-parameters-is-idempotent
    ((pairs objective-pairs))
  "Applying it to its own result changes nothing."
  (:about objective-parameters)
  (:kind :idempotence)
  (:trials (:smoke 20 :normal 200))
  (let ((once (objective-parameters (pairs-plist pairs))))
    (equal once (objective-parameters once))))
