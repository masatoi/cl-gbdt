(in-package #:cl-user)

;; Trains on this directory's data, written by prepare.lisp, with the shared trainer in
;; train-core.lisp; SBCL and CFFI required.

(load (merge-pathnames "../common.lisp" *load-truename*))

(load (merge-pathnames "../ranking-state.lisp" *load-truename*))

(load (merge-pathnames "train-core.lisp" *load-truename*))

;; The root is this file's own directory; nothing outside this LOAD can redirect it.
(let ((root (uiop:pathname-directory-pathname *load-truename*)))
  (cl-gbdt-examples::call-with-ranking-training
   root (lambda (stage) (movielens-ranking-example::train-ranking root stage))
   :name "MovieLens" :prefix "movielens"))
