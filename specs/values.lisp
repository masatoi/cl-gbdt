;;;; values.lisp --- Value specs and generators the other specification files share.
;;;;
;;;; Most of this file exists because of what cl-spec's check-it backend cannot generate yet:
;;;; `(range double-float ...)' is refused outright, `double-float', `ratio' and `keyword'
;;;; have no generator, and `real' draws single-floats only. cl-gbdt's parameters and
;;;; gradients are overwhelmingly double-floats and ratios, so those are supplied here as
;;;; custom generators. Custom value generators do not shrink: a counterexample keeps its
;;;; doubles exactly as drawn. docs/cl-spec-dogfooding.md records both facts.

(uiop:define-package #:cl-gbdt/specs/values
  (:use #:cl)
  (:import-from #:cl-spec/main
                #:defgenerator
                #:defspec)
  (:export #:finite-double
           #:finite-double-generator
           #:non-integer-ratio
           #:non-integer-ratio-generator
           #:parameter-key
           #:parameter-value
           #:pairs-plist
           #:plist-pairs))

(in-package #:cl-gbdt/specs/values)

(defgenerator finite-double-generator ()
  ;; A signed integer mantissa scaled by 10^-8 .. 10^8: magnitudes from 1d-8 to 1d14, well
  ;; inside single-float range too, so coercing one never overflows on a trapping platform.
  (* (float (- (random 2000001) 1000000) 1d0)
     (expt 10d0 (- (random 17) 8))))

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

(defspec parameter-key
  (member :num-leaves :learning-rate :min-data-in-leaf :objective :is-unbalance
          :bagging-fraction :max-depth :lambda-l1 :num-class :verbosity))

(defspec parameter-value
  (or (range integer -100000 100000) string boolean (range real -1000 1000)
      finite-double non-integer-ratio))

(defun pairs-plist (pairs)
  "Return the plist whose key/value pairs are PAIRS, a list of (KEY VALUE) lists, in order."
  (loop :for (key value) :in pairs :append (list key value)))

(defun plist-pairs (plist)
  "Return PLIST as a list of (KEY VALUE) lists, in order."
  (loop :for (key value) :on plist :by #'cddr :collect (list key value)))
