;;;; values.lisp --- Value specs and generators the other specification files share.
;;;;
;;;; Most of this file exists because of what cl-spec's check-it backend cannot generate yet:
;;;; `(range double-float ...)' is refused outright, `double-float', `ratio' and `keyword'
;;;; have no generator, and `real' draws single-floats only. cl-gbdt's parameters and
;;;; gradients are overwhelmingly double-floats and ratios, so those are supplied here as
;;;; custom generators. Custom value generators do not shrink: a counterexample keeps its
;;;; doubles exactly as drawn. docs/cl-spec-dogfooding.md records both facts.
;;;;
;;;; A spec here that a Function Spec names as an argument states what the target ACCEPTS; its
;;;; generator, or a whole-call generator, only decides what is SAMPLED. `parameter-key' is the
;;;; pattern: any keyword or string validates, and a fixed set is drawn. Specs used only as
;;;; Property domains (`parameter-value', `parameter-pair') are sampling domains and say so.

(uiop:define-package #:cl-gbdt/specs/values
  (:use #:cl)
  (:import-from #:cl-spec/main
                #:defgenerator
                #:defspec)
  (:export #:*parameter-keys*
           #:draw-finite-double
           #:finite-double
           #:finite-double-generator
           #:non-integer-ratio
           #:non-integer-ratio-generator
           #:parameter-key
           #:parameter-key-generator
           #:parameter-pair
           #:parameter-value
           #:pairs-plist
           #:plist-pairs))

(in-package #:cl-gbdt/specs/values)

(defun draw-finite-double ()
  "Return a double-float drawn with `random': a signed integer mantissa scaled by 10^-8 ..
10^8, so a magnitude from 1d-8 to 1d14 or exactly zero.

Well inside single-float range too, so coercing one never overflows on a trapping platform.
A plain function rather than only a generator body because a `defgenerator' body cannot draw
from a registered spec, so the whole-call generators elsewhere call this directly."
  (* (float (- (random 2000001) 1000000) 1d0)
     (expt 10d0 (- (random 17) 8))))

(defgenerator finite-double-generator ()
  (draw-finite-double))

;;; Validates any double-float, infinities included; only the generator is finite.
(defspec finite-double (type double-float)
  (:generator finite-double-generator))

(defgenerator non-integer-ratio-generator ()
  ;; NUMERATOR is never a multiple of DENOMINATOR, so the quotient is always a true ratio.
  (let* ((denominator (+ 2 (random 999)))
         (numerator (+ (* denominator (random 50)) 1 (random (1- denominator))))
         (ratio (/ numerator denominator)))
    (if (zerop (random 2)) ratio (- ratio))))

(defspec non-integer-ratio (type ratio)
  (:generator non-integer-ratio-generator))

(defparameter *parameter-keys*
  '(:num-leaves :learning-rate :min-data-in-leaf :objective :is-unbalance :bagging-fraction
    :feature-fraction :max-depth :lambda-l1 :num-class :verbosity :verbose)
  "The keys generated plists draw from: LightGBM and XGBoost spellings, the backend-specific
ones among them. A sampling set, not the domain -- see `parameter-key'.")

(defgenerator parameter-key-generator ()
  (nth (random (length *parameter-keys*)) *parameter-keys*))

;;; Any keyword or string validates: `normalize-parameters' has no per-key allowlist -- a
;;; backend-specific key passes through, README's Features section says so -- and a string
;;; key is the same key to it (`objective-parameters' relies on that). Only the generator
;;; draws from a fixed set, so that generated plists repeat keys.
(defspec parameter-key (or keyword string)
  (:generator parameter-key-generator))

;;; A sampling domain for the Properties' generated values, not a statement of what
;;; `normalize-parameters' accepts (any value -- see its Function Spec). The bounds keep draws
;;; small; the types are the ones a caller writes and the ones the rendering rules differ on.
(defspec parameter-value
  (or (range integer -100000 100000) string boolean (range real -1000 1000)
      finite-double non-integer-ratio (member :gbdt :dart :rf :binary :multiclass)))

;;; One (KEY VALUE) pair of a Property's sampled plist. A `tuple' alone admits a vector as
;;; well as a list; the helpers below destructure lists.
(defspec parameter-pair
  (and (type list) (tuple parameter-key parameter-value)))

(defun pairs-plist (pairs)
  "Return the plist whose key/value pairs are PAIRS, a list of (KEY VALUE) lists, in order."
  (loop :for (key value) :in pairs :append (list key value)))

(defun plist-pairs (plist)
  "Return PLIST as a list of (KEY VALUE) lists, in order."
  (loop :for (key value) :on plist :by #'cddr :collect (list key value)))
