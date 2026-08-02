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

(defsystem "cl-gbdt/regen"
  :description "Binding emitter. Development only; never part of a build."
  :depends-on ("cffi/c2ffi" "com.inuoe.jzon" "alexandria")
  :serial t
  :components ((:module "src/regen"
                :components ((:file "package")
                             (:file "types")
                             (:file "emit")))))

(defsystem "cl-gbdt/tests"
  :author ""
  :license ""
  :depends-on ("cl-gbdt"
               "cl-gbdt/regen"
               "rove")
  :components ((:module "tests"
                :serial t
                :components
                ((:file "package")
                 (:file "main")
                 (:file "regen"))))
  :description "Test system for cl-gbdt"
  :perform (test-op (op c) (symbol-call :rove :run c)))
