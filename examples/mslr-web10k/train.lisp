(in-package #:cl-user)

;;;; Trains on this directory's prepared fold with the trainer the MovieLens example shares.

(load (merge-pathnames "../common.lisp" *load-truename*))

(load (merge-pathnames "state.lisp" *load-truename*))

(load (merge-pathnames "../movielens-ranking/train-core.lisp" *load-truename*))

(let ((root (uiop:pathname-directory-pathname *load-truename*)))
  (cl-gbdt-examples::call-with-ranking-training
   root (lambda (stage) (movielens-ranking-example::train-ranking root stage))
   :name "MSLR" :prefix "mslr"))
