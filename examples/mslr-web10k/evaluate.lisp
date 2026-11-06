(in-package #:cl-user)

;;;; Evaluates this directory's predictions with the evaluator the MovieLens example shares.

(load (merge-pathnames "../common.lisp" *load-truename*))

(load (merge-pathnames "state.lisp" *load-truename*))

(load (merge-pathnames "../movielens-ranking/evaluate-core.lisp" *load-truename*))

(let ((root (uiop:pathname-directory-pathname *load-truename*)))
  (cl-gbdt-examples::call-with-ranking-evaluation
   root (lambda () (cl-gbdt-examples::evaluate-ranking root))
   :name "MSLR" :prefix "mslr"))
