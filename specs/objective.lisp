;;;; objective.lisp --- Executable contracts for `train''s custom-objective helpers.
;;;;
;;;; `objective-single-float' has two outcomes, a `single-float' for a real and
;;;; `unsupported-element-type' for anything else, stated as two exclusive cases.
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
                #:defproperty
                #:defspec
                #:defspec-function)
  (:import-from #:cl-gbdt/src/conditions
                #:unsupported-element-type)
  (:import-from #:cl-gbdt/src/config/objective
                #:objective-parameters
                #:objective-single-float)
  (:import-from #:cl-gbdt/specs/values
                #:finite-double
                #:non-integer-ratio
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

Compared as rationals, so the check itself does no floating-point arithmetic that could trap."
  (<= (abs (- (rational result) (rational value)))
      (* (abs (rational value)) (rational single-float-epsilon))))

(defspec element-value
  (or (range integer -1000000 1000000) (range real -1000000 1000000) finite-double
      non-integer-ratio string boolean (member :gradient #\x)))

(defspec-function objective-single-float
  "A real becomes the nearest single-float; anything else is refused."
  (:args (value element-value))
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
