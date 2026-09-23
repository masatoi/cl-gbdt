;;;; parameters.lisp --- Executable contract and Properties for `normalize-parameters'.
;;;;
;;;; tests/parameters.lisp pins one value per rule (`0.05d0', `1/3', `*print-base*' 16). The
;;;; Properties here state the rules those examples were chosen to illustrate, over generated
;;;; plists: names and order, values that read back as themselves, and independence from the
;;;; caller's printer state.
;;;;
;;;; The printer Property's domain holds symbol values and binds `*print-case*' too. It was
;;;; first written without them, because the implementation `princ'ed a symbol under the
;;;; caller's `*print-case*' -- `:gbdt' rendered as "GBDT" normally and "gbdt" under
;;;; `:downcase'. That was fixed by rendering a symbol as its `symbol-name'; see
;;;; docs/cl-spec-dogfooding.md's "Findings about cl-gbdt".

(uiop:define-package #:cl-gbdt/specs/parameters
  (:use #:cl)
  (:import-from #:cl-spec/main
                #:defgenerator
                #:defproperty
                #:defspec-function)
  (:import-from #:cl-gbdt/src/conditions
                #:data-error)
  (:import-from #:cl-gbdt/src/parameters
                #:normalize-parameters)
  (:import-from #:cl-gbdt/specs/values
                #:pairs-plist
                #:parameter-key
                #:parameter-pair
                #:parameter-value)
  (:export #:normalize-parameters-keeps-order-and-renames-keys
           #:normalize-parameters-values-denote-themselves
           #:normalize-parameters-ignores-the-caller-s-printer))

(in-package #:cl-gbdt/specs/parameters)

(defparameter *keys*
  '(:num-leaves :learning-rate :min-data-in-leaf :objective :is-unbalance)
  "Keys `normalize-parameters-arguments' draws from.")

(defun string-pair-p (object)
  "True when OBJECT is a (NAME . VALUE) cons of two strings."
  (and (consp object) (stringp (car object)) (stringp (cdr object))))

(defun keys-are-keywords-p (plist)
  "True when every key position of PLIST -- position 0, 2, 4 and so on, the last element of
an odd-length PLIST included -- holds a keyword."
  (loop :for key :in plist :by #'cddr :always (keywordp key)))

(defun denotes-p (value text)
  "True when TEXT, `normalize-parameters''s rendering of VALUE, denotes VALUE.

A string passes through as the same object; T and NIL are \"true\" and \"false\"; any other
symbol is its name, exactly; an integer is its decimal digits; a float or ratio carries no
Lisp-only syntax -- no exponent marker other than `e', no `/' -- and reads back as the same
number (a ratio as its `double-float')."
  (typecase value
    (string (eq value text))
    ((eql t) (string= text "true"))
    (null (string= text "false"))
    (symbol (string= text (symbol-name value)))
    (integer (= value (parse-integer text)))
    ((or float ratio)
     (and (notany (lambda (char) (find char "dDfFsSlL/")) text)
          (let ((*read-eval* nil)
                (*read-default-float-format*
                  (if (typep value 'single-float) 'single-float 'double-float)))
            (= (read-from-string text)
               (if (rationalp value) (coerce value 'double-float) value)))))))

(defgenerator normalize-parameters-arguments ()
  (let ((length (random 10)))
    (list (loop :for position :below length
                :collect (if (evenp position)
                             (nth (random (length *keys*)) *keys*)
                             (random 1000))))))

(defspec-function normalize-parameters
  "A plist whose key positions hold keywords: even-length, it becomes one (NAME . VALUE)
string pair per key; odd-length, it is refused with `data-error'."
  (:args (plist (list-of (or parameter-key parameter-value) :max-length 9)))
  (:args-generator normalize-parameters-arguments)
  ;; The element spec alone would admit (42 7), which the implementation cannot name.
  (:pre (keys-are-keywords-p plist))
  (:cases
   (:even-length
    (:when (evenp (length plist)))
    (:returns (list-of (satisfies string-pair-p)))
    (:post (= (length result) (/ (length plist) 2))))
   (:odd-length
    (:when (oddp (length plist)))
    (:signals (type data-error)))))

(defproperty normalize-parameters-keeps-order-and-renames-keys
    ((pairs (list-of parameter-pair :max-length 8)))
  "One pair per key, in plist order; each name is the key lower-cased with `_' for `-'."
  (:about normalize-parameters)
  (:kind :invariant)
  (:trials (:smoke 20 :normal 200))
  (let ((result (normalize-parameters (pairs-plist pairs))))
    (and (= (length result) (length pairs))
         (every (lambda (pair entry)
                  (let ((name (car entry)))
                    (and (stringp name)
                         (notany (lambda (char) (or (char= char #\-) (upper-case-p char)))
                                 name)
                         (string-equal (substitute #\- #\_ name)
                                       (symbol-name (first pair))))))
                pairs result))))

(defproperty normalize-parameters-values-denote-themselves
    ((pairs (list-of parameter-pair :max-length 8)))
  "Every rendered value reads back as the value it came from, with no Lisp-only syntax."
  (:about normalize-parameters)
  (:kind :round-trip)
  (:trials (:smoke 20 :normal 200))
  (every (lambda (pair entry) (denotes-p (second pair) (cdr entry)))
         pairs (normalize-parameters (pairs-plist pairs))))

(defproperty normalize-parameters-ignores-the-caller-s-printer
    ((pairs (list-of parameter-pair :max-length 8))
     (base (range integer 2 36))
     (radix boolean)
     (float-format (member single-float double-float short-float long-float))
     (print-case (member :upcase :downcase :capitalize)))
  "Every value renders the same whatever printer state the caller binds."
  (:about normalize-parameters)
  (:kind :invariant)
  (:trials (:smoke 20 :normal 200))
  (let ((plist (pairs-plist pairs)))
    (equal (with-standard-io-syntax (normalize-parameters plist))
           (let ((*print-base* base)
                 (*print-radix* radix)
                 (*read-default-float-format* float-format)
                 (*print-case* print-case))
             (normalize-parameters plist)))))
