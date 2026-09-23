;;;; prediction-shape.lisp --- Executable contract for `contrib-shape'.
;;;;
;;;; `contrib-shape' either derives (ROWS CLASSES WIDTH) from three counts or answers NIL, and
;;;; its docstring says exactly when: the shape exists only when NUM-ROWS is positive,
;;;; NUM-FEATURES is not negative, and ELEMENT-COUNT is a positive multiple of
;;;; NUM-ROWS x (NUM-FEATURES + 1). The contract states that as two exclusive cases.
;;;;
;;;; The generator is not decoration. Drawn independently over these ranges at seed 42, the
;;;; divisible case came up 3 times in 200 trials, and ELEMENT-COUNT 0 with positive rows --
;;;; the zero-class boundary the docstring singles out -- once in 1000, so an implementation
;;;; that accepted a zero-class shape passed the 200-trial budget and was caught only at trial
;;;; 280 (docs/cl-spec-dogfooding.md, G3). `contrib-shape-arguments' draws half its calls
;;;; from constructed shapes, a quarter at ELEMENT-COUNT 0, and the rest uniformly.

(uiop:define-package #:cl-gbdt/specs/prediction-shape
  (:use #:cl)
  (:import-from #:cl-spec/main
                #:defgenerator
                #:defspec-function)
  (:import-from #:cl-gbdt/src/config/prediction-shape
                #:contrib-shape)
  (:export #:contrib-derivable-p))

(in-package #:cl-gbdt/specs/prediction-shape)

(defun contrib-derivable-p (element-count num-rows num-features)
  "True when ELEMENT-COUNT, NUM-ROWS and NUM-FEATURES admit a `:contrib' shape: NUM-ROWS
positive, NUM-FEATURES not negative, and ELEMENT-COUNT a positive multiple of
NUM-ROWS x (NUM-FEATURES + 1).

Stated with `mod' rather than the `truncate' the implementation uses, so the guard and the
function under test do not share the arithmetic they are checked against."
  (and (plusp num-rows)
       (not (minusp num-features))
       (plusp element-count)
       (zerop (mod element-count (* num-rows (1+ num-features))))))

(defgenerator contrib-shape-arguments ()
  (case (random 4)
    ((0 1)
     (let ((rows (1+ (random 20)))
           (features (random 20))
           (classes (1+ (random 5))))
       (list (* rows classes (1+ features)) rows features)))
    (2 (list 0 (- (random 44) 3) (- (random 23) 2)))
    (t (list (random 2001) (- (random 44) 3) (- (random 23) 2)))))

(defspec-function contrib-shape
  "Derive (ROWS CLASSES WIDTH) exactly when the counts admit one; otherwise NIL."
  ;; Any three integers: the NIL cases include negative and zero counts. The generator draws
  ;; small ones; that bounds what is tried, not what is promised.
  (:args (element-count integer)
         (num-rows integer)
         (num-features integer))
  (:args-generator contrib-shape-arguments)
  (:cases
   (:derivable
    "A shape that accounts for every element: ROWS as given, WIDTH one past the features."
    (:when (contrib-derivable-p element-count num-rows num-features))
    (:returns (list-of (range integer 1 *) :min-length 3 :max-length 3))
    (:post (destructuring-bind (rows classes width) result
             (and (= rows num-rows)
                  (= width (1+ num-features))
                  (= element-count (* rows classes width))))))
   (:underivable
    "No shape is stated, never a guessed one."
    (:when (not (contrib-derivable-p element-count num-rows num-features)))
    (:returns null))))
