(:mallet-config
 (:extends :default)
 ;; The non-local-exit tests raise deliberately, inside `ignore-errors', to prove that
 ;; `with-dataset', `with-booster' and `with-backend' still free their handles when the
 ;; body signals. That is the opposite of swallowing an error -- the signal is the
 ;; fixture -- so the rule does not apply to the test tree.
 (:for-paths ("tests/backend.lisp" "tests/functional/with-backend.lisp")
   (:disable :no-ignore-errors)))
