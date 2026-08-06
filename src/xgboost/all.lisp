;;;; all.lisp --- The XGBoost backend's public face.
;;;;
;;;; Reassembles `cl-gbdt/src/xgboost/native' (Layer 1: library discovery, the error
;;;; wrapper, every %-function) and `cl-gbdt/src/xgboost/protocol' (Layer 2: the classes
;;;; and the fourteen protocol methods) into one package. This is what `cl-gbdt/xgboost'
;;;; (see cl-gbdt.asd) depends on, following the same shape `src/all.lisp' and
;;;; `src/regen/all.lisp' use for the same reason.
;;;;
;;;; Deliberately does not reexport `cl-gbdt/src/xgboost/c-api', the raw CFFI bindings --
;;;; see policy sections 3 and 11. That package's own docstring already says nothing
;;;; outside the backend system should call it directly; reexporting it here would make
;;;; every one of those raw C symbols part of this system's public surface instead.

(uiop:define-package #:cl-gbdt/src/xgboost/all
  (:use-reexport #:cl-gbdt/src/xgboost/native
                 #:cl-gbdt/src/xgboost/protocol))
