;;;; all.lisp --- The LightGBM backend's public face.
;;;;
;;;; Reassembles `cl-gbdt/src/lightgbm/native' (Layer 1: library discovery, the error
;;;; wrapper, every %-function) and `cl-gbdt/src/lightgbm/protocol' (Layer 2: the classes
;;;; and the fourteen protocol methods) into one package. This is what `cl-gbdt/lightgbm'
;;;; (see cl-gbdt.asd) depends on, following the same shape `src/xgboost/all.lisp' uses
;;;; for the same reason.
;;;;
;;;; Deliberately does not reexport `cl-gbdt/src/lightgbm/c-api', the raw CFFI bindings --
;;;; see policy sections 3 and 11. That package's own docstring already says nothing
;;;; outside the backend system should call it directly; reexporting it here would make
;;;; every one of those raw C symbols part of this system's public surface instead.

(uiop:define-package #:cl-gbdt/src/lightgbm/all
  (:use-reexport #:cl-gbdt/src/lightgbm/native
                 #:cl-gbdt/src/lightgbm/protocol))
