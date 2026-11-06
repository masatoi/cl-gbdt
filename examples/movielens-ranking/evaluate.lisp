(in-package #:cl-user)

(load (merge-pathnames "../common.lisp" *load-truename*))

(load (merge-pathnames "../ranking-state.lisp" *load-truename*))

(load (merge-pathnames "evaluate-core.lisp" *load-truename*))

;; The root is this file's own directory; nothing outside this LOAD can redirect it.
(let ((root (uiop:pathname-directory-pathname *load-truename*)))
  (cl-gbdt-examples::call-with-ranking-evaluation
   root (lambda () (cl-gbdt-examples::evaluate-ranking root))
   :name "MovieLens" :prefix "movielens"))
