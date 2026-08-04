;;;; parameters.lisp --- Turning a plist of parameters into backend key/value strings.

(uiop:define-package #:cl-gbdt/src/parameters
  (:use #:cl)
  (:import-from #:cl-gbdt/src/conditions #:data-error)
  (:export #:normalize-parameters))

(in-package #:cl-gbdt/src/parameters)

(defun parameter-name (key)
  "Return the backend spelling of the parameter KEY.

`:num-leaves' becomes `\"num_leaves\"'. Both backends spell parameters in snake_case, so
the keyword's dashes become underscores and its name is downcased."
  (substitute #\_ #\- (string-downcase (string key))))

(defun %rational-to-decimal-string (value)
  "Return the ratio VALUE as a decimal string, via `double-float'.

`princ-to-string' on a ratio prints e.g. \"1/3\"; neither backend's numeric
parser accepts a fraction."
  (princ-to-string (coerce value 'double-float)))

(defun parameter-value (value)
  "Return VALUE as the string a backend expects.

`princ' rather than `prin1', so a string is not re-quoted and a symbol loses its
package. Three printer specials are bound locally for the duration, so the
result cannot depend on bindings already in force in the caller: `*print-base*'
10 and `*print-radix*' nil, so an integer such as `:num-leaves 31' prints as
plain decimal digits regardless of what the caller has bound `*print-base*' to
rather than, say, \"1F\"; and `*read-default-float-format*' `double-float', so a
double-float value -- the natural float type here, since a dataset's matrix
must be `double-float' or `single-float' -- prints as `\"0.05\"' rather than
`\"0.05d0\"', which neither backend's numeric parser accepts. `T' and `NIL' print
as LightGBM's own spelling for a boolean, `\"true\"'/`\"false\"', and a ratio is
rendered as a decimal rather than `\"1/3\"'."
  (let ((*print-base* 10)
        (*print-radix* nil)
        (*read-default-float-format* 'double-float))
    (typecase value
      (string value)
      ((eql t) "true")
      (null "false")
      (ratio (%rational-to-decimal-string value))
      (t (princ-to-string value)))))

(defun normalize-parameters (plist)
  "Return PLIST as a list of (NAME . VALUE) string pairs, in order.

Nothing is validated or filtered: a backend-specific parameter passes through untouched,
which is what `make-dataset' and `train' promise. Signals `data-error' when PLIST has an
odd length, because a silently dropped final key is how a parameter goes missing."
  (when (oddp (length plist))
    (error 'data-error))
  (loop :for (key value) :on plist :by #'cddr
        :collect (cons (parameter-name key) (parameter-value value))))
