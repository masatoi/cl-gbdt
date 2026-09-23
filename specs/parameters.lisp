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
                #:*parameter-keys*
                #:pairs-plist
                #:parameter-pair)
  (:export #:normalize-parameters-keeps-order-and-renames-keys
           #:normalize-parameters-values-denote-themselves
           #:normalize-parameters-ignores-the-caller-s-printer))

(in-package #:cl-gbdt/specs/parameters)

(defparameter *sampled-values*
  (list 31 0.05 0.05d0 1/3 t nil "binary" :gbdt -7 1.0d-7)
  "Values `normalize-parameters-arguments' draws from, one per rendering rule. What the
contract admits is any value; this is only what it samples.")

(defun string-pair-p (object)
  "True when OBJECT is a (NAME . VALUE) cons of two strings."
  (and (consp object) (stringp (car object)) (stringp (cdr object))))

(defun keys-are-parameter-keys-p (plist)
  "True when every key position of PLIST -- position 0, 2, 4 and so on, the last element of
an odd-length PLIST included -- holds a keyword or a string, as `parameter-key' states."
  (loop :for key :in plist :by #'cddr :always (typep key '(or keyword string))))

(defun parameter-name-string (key)
  "Return KEY's name as a string key would spell it, `:num-leaves' as \"num_leaves\"."
  (substitute #\_ #\- (string-downcase (symbol-name key))))

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
                             (let ((key (nth (random (length *parameter-keys*))
                                             *parameter-keys*)))
                               ;; Now and then the string spelling of the same key.
                               (if (zerop (random 4)) (parameter-name-string key) key))
                             (nth (random (length *sampled-values*)) *sampled-values*))))))

(defspec-function normalize-parameters
  "A plist of any values whose key positions hold keywords or strings: even-length, it
becomes one (NAME . VALUE) string pair per key; odd-length, it is refused with `data-error'."
  ;; Any length, any value: nothing is validated or filtered, as the docstring promises. The
  ;; generator keeps plists short and draws from a sampling set; that bounds what is tried,
  ;; not what is promised.
  (:args (plist (list-of t)))
  (:args-generator normalize-parameters-arguments)
  ;; Without this, (42 7) would be admitted, and the implementation cannot name 42.
  (:pre (keys-are-parameter-keys-p plist))
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
                    ;; Compared with both sides' underscores read as dashes, so a string key
                    ;; already spelled "num_leaves" is held to the same rule as `:num-leaves'.
                    (and (stringp name)
                         (notany (lambda (char) (or (char= char #\-) (upper-case-p char)))
                                 name)
                         (string-equal (substitute #\- #\_ name)
                                       (substitute #\- #\_ (string (first pair)))))))
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
