(in-package #:cl-user)

;;;; Toy data uses the shared example lock, but has no separate training step.
(load (merge-pathnames "../lifecycle.lisp" *load-truename*))

(defpackage #:cl-gbdt-toy-state
  (:use #:cl)
  (:export #:call-with-preparation #:call-with-analysis #:*staging-directory*))

(in-package #:cl-gbdt-toy-state)

(defvar *staging-directory* nil
  "Private directory containing all three splits until preparation succeeds.")

(defparameter *toy-layout* (cl-gbdt-example-lifecycle:make-layout :name "Toy" :prefix "toy")
  "The shared examples' lock and markers, with toy-named private directories.")

(defun call-with-preparation (root function)
  "Call FUNCTION to write the three splits to *STAGING-DIRECTORY*, under ROOT's lock.
The splits replace those in ROOT's data/ together with a new prepared generation only
if FUNCTION returns; otherwise the earlier splits are restored and analysis is disabled."
  (cl-gbdt-example-lifecycle:call-with-preparation
   root
   (lambda ()
     (let ((*staging-directory* cl-gbdt-example-lifecycle:*preparation-output-directory*))
       (funcall function)))
   :layout *toy-layout* :names '("iris.sexp" "breast_cancer.sexp" "diabetes.sexp")))

(defun call-with-analysis (root function)
  "Call FUNCTION to analyse ROOT's prepared splits under its lock.
What it prints is shown only if the splits are still current when it returns."
  (cl-gbdt-example-lifecycle:call-with-evaluation
   root function :layout *toy-layout* :trained nil))
