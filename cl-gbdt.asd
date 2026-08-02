(defsystem "cl-gbdt"
  :version "0.0.1"
  :author ""
  :license ""
  :depends-on ()
  :components ((:module "src"
                :components
                ((:file "main"))))
  :description ""
  :in-order-to ((test-op (test-op "cl-gbdt/tests"))))

(defsystem "cl-gbdt/tests"
  :author ""
  :license ""
  :depends-on ("cl-gbdt"
               "rove")
  :components ((:module "tests"
                :components
                ((:file "main"))))
  :description "Test system for cl-gbdt"
  :perform (test-op (op c) (symbol-call :rove :run c)))
