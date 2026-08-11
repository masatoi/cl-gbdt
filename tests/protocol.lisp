;;;; protocol.lisp --- Tests for the unified API's own behaviour, independent of any backend.
;;;;
;;;; `cl-gbdt:backend' is instantiable and has no unified-API methods of its own, which is
;;;; exactly the state a program reaches by loading cl-gbdt and cl-gbdt/lightgbm without
;;;; cl-gbdt/lightgbm/unified. So these need no shared library (layer 1). That holds for the
;;;; keyword-parsing test below as well: an unknown keyword is rejected before any method
;;;; body runs, so the `program-error' arrives with no backend open and no library loaded --
;;;; measured in a fresh image with neither backend system present, not assumed.

(uiop:define-package #:cl-gbdt/tests/protocol
  (:use #:cl #:rove)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt))

(in-package #:cl-gbdt/tests/protocol)

(defun bare-backend ()
  "Return a backend whose class has no unified-API methods.

Named `:lightgbm' so the report's advice can be checked verbatim: the condition builds the
system name out of the backend's own name."
  (make-instance 'cl-gbdt:backend :name :lightgbm))

(defun bare-booster ()
  "Return a booster handle on a backend that has no unified-API methods.

Its pointer is never dereferenced: every generic function reached through it signals
before any foreign call, so an arbitrary non-null address is enough. `drop' it."
  (cl-gbdt:make-handle 'cl-gbdt:booster (cffi:make-pointer 1) (bare-backend) :booster))

(defun drop (handle)
  "Release HANDLE with a free function that does nothing, cancelling its finalizer."
  (cl-gbdt:release-handle handle (lambda (pointer) (declare (ignore pointer)) nil)))

(defun rejects-unknown-keyword-p (function &rest arguments)
  "True when calling FUNCTION on ARGUMENTS plus one unknown keyword signals `program-error'.

The bogus keyword is applied rather than written into the call site, so compiling this file
stays quiet. SBCL detects a misspelled keyword at COMPILE time too -- that warning is the
other half of what the widened fallbacks silently switched off -- and a literal call here
would emit it on every build, including the `(asdf:compile-system :cl-gbdt :force t)' run
whose whole purpose is to surface warnings worth reading. What this asserts is the runtime
rejection.

The portable supertype is what is checked, not SBCL's own
`sb-ext:unknown-keyword-argument' and not its report string. `handler-case' rather than
rove's `signals', per this project's REPL prompt."
  (handler-case (progn (apply function (append arguments (list :totally-bogus 1))) nil)
    (program-error () t)))

(defun reaches-the-fallback-p (function &rest arguments)
  "True when calling FUNCTION on ARGUMENTS signals `backend-methods-not-loaded'.

That condition comes from a fallback method's own BODY, which runs only after every
keyword it was handed has been accepted -- so this is the half that stops
`rejects-unknown-keyword-p' being satisfiable by a fallback that rejects everything."
  (handler-case (progn (apply function arguments) nil)
    (cl-gbdt:backend-methods-not-loaded () t)))

(deftest backend-methods-not-loaded-is-a-backend-error
  (testing "it sits under backend-error, and so under gbdt-error"
    (ok (subtypep 'cl-gbdt:backend-methods-not-loaded 'cl-gbdt:backend-error)
        "a caller handling backend-error catches it")
    (ok (subtypep 'cl-gbdt:backend-methods-not-loaded 'cl-gbdt:gbdt-error)
        "and so does one handling gbdt-error")))

(deftest a-backend-dispatching-generic-says-which-system-to-load
  (testing "train names the backend, the generic function, and the system"
    (let ((caught (handler-case (progn (cl-gbdt:train (bare-backend) nil) nil)
                    (cl-gbdt:backend-methods-not-loaded (condition) condition))))
      (ok caught "the condition was signalled")
      (ok (eq :lightgbm (cl-gbdt:backend-error-backend caught)) "it names the backend")
      (ok (eq 'cl-gbdt:train
              (cl-gbdt:backend-methods-not-loaded-generic-function caught))
          "it names the generic function that was called")
      (ok (search "cl-gbdt/lightgbm/unified" (princ-to-string caught))
          "its report names the system to load"))))

(deftest make-dataset-and-load-model-signal-it-too
  (testing "every generic that dispatches on the backend is covered"
    (ok (handler-case (progn (cl-gbdt:make-dataset (bare-backend) #2A((1d0)) ) nil)
          (cl-gbdt:backend-methods-not-loaded () t))
        "make-dataset signals it")
    (ok (handler-case (progn (cl-gbdt:load-model (bare-backend) #p"/nonexistent") nil)
          (cl-gbdt:backend-methods-not-loaded () t))
        "load-model signals it")))

(deftest a-fallback-never-widens-its-generic-to-accept-any-keyword
  ;; The regression this pins is quiet, and it happened. A fallback written
  ;; `&key &allow-other-keys' is APPLICABLE alongside every concrete backend method -- the
  ;; base class it specializes on is a superclass of the concrete one, so both are
  ;; applicable and only the ORDER differs -- and CLHS 7.6.5 makes a generic function accept
  ;; every keyword as soon as ANY applicable method has `&allow-other-keys'. Six of the
  ;; thirteen generics were written that way, and with a real backend loaded
  ;; `(cl-gbdt:make-dataset backend matrix :totally-bogus 1)' stopped signalling and started
  ;; returning a dataset. A typo'd `:parameters' or `:num-rounds' then trained a differently
  ;; configured model and returned different numbers, on both backends, with nothing raised
  ;; and no test anywhere failing. Each fallback now spells out its generic's own keyword
  ;; list, and this is what makes that loud if one is ever widened again -- including one
  ;; widened alone, since every generic that takes keywords is checked separately.
  (let ((backend (bare-backend))
        (booster (bare-booster)))
    (testing "an unknown keyword is a program-error, not a silently accepted argument"
      (ok (rejects-unknown-keyword-p #'cl-gbdt:make-dataset backend #2A((1d0)))
          "make-dataset rejects it")
      (ok (rejects-unknown-keyword-p #'cl-gbdt:train backend nil)
          "train rejects it")
      (ok (rejects-unknown-keyword-p #'cl-gbdt:predict booster #2A((1d0)))
          "predict rejects it")
      (ok (rejects-unknown-keyword-p #'cl-gbdt:save-model booster #p"/nonexistent")
          "save-model rejects it")
      (ok (rejects-unknown-keyword-p #'cl-gbdt:model-to-string booster)
          "model-to-string rejects it")
      (ok (rejects-unknown-keyword-p #'cl-gbdt:feature-importance booster)
          "feature-importance rejects it"))
    (testing "every keyword the generic function does declare is still accepted"
      (ok (reaches-the-fallback-p #'cl-gbdt:make-dataset backend #2A((1d0))
                                  :label #(1d0) :weight #(1d0) :group '(1)
                                  :feature-names '("a") :parameters '(:max-bin 2)
                                  :reference nil :missing -1d0 :categorical-features '(0))
          "make-dataset takes all eight of its keywords")
      (ok (reaches-the-fallback-p #'cl-gbdt:train backend nil
                                  :valid-sets nil :num-rounds 1 :parameters nil
                                  :record-history t :early-stopping nil :objective nil
                                  :evaluation nil)
          "train takes all seven of its keywords")
      (ok (reaches-the-fallback-p #'cl-gbdt:predict booster #2A((1d0))
                                  :kind :normal :num-iteration 1 :missing -1d0)
          "predict takes all three of its keywords")
      (ok (reaches-the-fallback-p #'cl-gbdt:save-model booster #p"/nonexistent"
                                  :num-iteration 1)
          "save-model takes :num-iteration")
      (ok (reaches-the-fallback-p #'cl-gbdt:model-to-string booster :num-iteration 1)
          "model-to-string takes :num-iteration")
      (ok (reaches-the-fallback-p #'cl-gbdt:feature-importance booster
                                  :kind :split :num-iteration 1)
          "feature-importance takes both of its keywords"))
    (drop booster)))

(deftest a-handle-dispatching-generic-names-the-handles-backend
  (testing "dataset-num-rows reaches the backend through the handle"
    (let* ((handle (cl-gbdt:make-handle 'cl-gbdt:dataset (cffi:make-pointer 1)
                                        (bare-backend) :dataset))
           (caught (handler-case (progn (cl-gbdt:dataset-num-rows handle) nil)
                     (cl-gbdt:backend-methods-not-loaded (condition) condition))))
      (ok caught "the condition was signalled")
      (ok (eq :lightgbm (cl-gbdt:backend-error-backend caught))
          "it names the backend the handle belongs to")
      (drop handle))))
