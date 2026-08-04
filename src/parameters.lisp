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

(defun parameter-value (value)
  "Return VALUE as the string a backend expects.

`princ' rather than `prin1', so a string is not re-quoted and a symbol loses its package."
  (typecase value
    (string value)
    (t (princ-to-string value))))

(defun normalize-parameters (plist)
  "Return PLIST as a list of (NAME . VALUE) string pairs, in order.

Nothing is validated or filtered: a backend-specific parameter passes through untouched,
which is what `make-dataset' and `train' promise. Signals `data-error' when PLIST has an
odd length, because a silently dropped final key is how a parameter goes missing."
  (when (oddp (length plist))
    (error 'data-error))
  (loop :for (key value) :on plist :by #'cddr
        :collect (cons (parameter-name key) (parameter-value value))))
