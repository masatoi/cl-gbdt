;;;; threads.lisp --- What cl-gbdt does under concurrency, and what these tests do not prove.
;;;;
;;;; WHAT A GREEN RUN OF THIS FILE DOES NOT MEAN. Read this before citing it.
;;;;
;;;; `threads-report-their-own-foreign-errors' is the only falsifiable claim here. It
;;;; establishes that both libraries' last-error buffers are per-thread, and it fails if that
;;;; is false: with a process-global buffer, threads would routinely read each other's
;;;; messages.
;;;;
;;;; That claim is itself a SAMPLING test, not a proof, and it is the strength of the claim
;;;; that changes, not the claim itself. `check-foreign-call' (`src/foreign.lisp') reads the
;;;; last-error buffer immediately after the failing call returns, so a process-global buffer
;;;; is only OBSERVED when another thread's failing call lands inside that sub-millisecond
;;;; window. `*error-rounds*' below exists to give that window enough chances to be hit; more
;;;; rounds buy this test more probability of catching a process-global buffer in the act, not
;;;; certainty of catching one -- see its own docstring for what raising it costs.
;;;;
;;;; `concurrent-training-agrees-with-serial-training' proves nothing about safety. It samples
;;;; one interleaving out of a nondeterministic space; a race that fires once in a thousand
;;;; runs passes it almost every time. Its value is as a regression net for gross breakage,
;;;; and as a baseline to compare against if a lock or a CAS is ever added.
;;;;
;;;; A concurrency test can find a bug; it cannot show the absence of one. Nothing here makes
;;;; any of `docs/user-guide/threads.md''s "unsafe" cases safe, and nothing here should be
;;;; cited as though it had. The two worst of those cases -- freeing one handle from two
;;;; threads, and closing a backend under a call in flight -- are deliberately NOT tested:
;;;; both end the process rather than failing an assertion, so a test of them would take the
;;;; suite down instead of reporting.
;;;;
;;;; `sb-thread' directly, not `bordeaux-threads': this system is already declared SBCL-only
;;;; in cl-gbdt.asd and its files already call `sb-sys' directly, so a portability dependency
;;;; would buy nothing the surrounding suite has.

(uiop:define-package #:cl-gbdt/tests/functional/threads
  (:use #:cl #:rove)
  (:import-from #:sb-thread)
  (:import-from #:cl-gbdt)
  (:import-from #:cl-gbdt/tests/functional/support
                #:with-backend-library
                #:make-separable-dataset
                #:predictions-agree-p))

(in-package #:cl-gbdt/tests/functional/threads)

(defparameter *thread-count* 4
  "Threads each concurrent test runs. Four rather than one per core: this file is about
whether cl-gbdt holds shared state, not about saturating the machine, and every thread here
trains a model inside a functional suite that already runs about three minutes.")

(defparameter *error-rounds* 500
  "Times each thread provokes a foreign error in `threads-report-their-own-foreign-errors'.

Each round is one failing `train' call on an eight-row fixture, so raising this is cheap:
measured at roughly 1 second total, both backends and all four threads together, against
roughly 0.1 second for the 8 rounds this file used before that number was found to be too
small to say anything. `check-foreign-call' (`src/foreign.lisp') reads the last-error buffer
immediately after the failing call returns, so a process-global buffer is only OBSERVED when
another thread's failing call lands inside that sub-millisecond window -- at a low round count
a genuinely global buffer could easily pass by chance alone. More rounds buy this test more
opportunities for that window to be hit; they buy probability, not certainty. This test can
still pass against a process-global buffer it never happened to catch in the act.")

(defun %thread-token (index)
  "Return the objective name thread INDEX provokes its foreign error with.

Lower case deliberately: LightGBM downcases the objective name it echoes into its error
message (measured against the vendored 4.7.0), so an upper-case token would fail the search on
that backend for a reason unrelated to threads.

The trailing `z' is not decoration. Without it `t1' is a prefix of `t10', and a thread would
accept another thread's message as its own once the count reached ten -- silently weakening
the test rather than failing it."
  (format nil "bogus-obj-t~Dz" index))

(defun %provoke-foreign-error (backend token matrix labels)
  "Train on BACKEND with TOKEN as the objective, and return the resulting condition's
message. Returns NIL if the call unexpectedly succeeds.

BACKEND is an ALREADY-OPEN backend shared with the other threads, not a name to open here.
Opening one per thread would have every thread closing the shared library while the others
were calling into it -- `shutdown-backend' calls `cffi:close-foreign-library', whose own
docstring says it may unmap the library -- and that is one of the very hazards
docs/user-guide/threads.md documents as unsafe. A test must not demonstrate the thing the
contract forbids. Sharing one open backend is instead exactly the contract's SAFE tier:
distinct handles, distinct threads, the backend open throughout. Do not \"simplify\" this
back to opening per call.

The `handler-case' clause below is deliberately `cl-gbdt:foreign-call-error', not bare
`error', and returns `foreign-call-error-message', not `princ-to-string' of the whole
condition. A bare `error' clause would also catch a Lisp-side validation of :OBJECTIVE that
never reached the foreign call at all, and `princ-to-string' includes this wrapper's own
\"FUNCTION-NAME returned CODE: \" prefix rather than only the library's own text. Either
laxity would let a future Lisp-side check that happened to echo the caller's token satisfy
every assertion below with no foreign last-error buffer involved at all -- exactly what this
test exists to rule out."
  (handler-case
      (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset backend matrix :label labels))
        (cl-gbdt:train backend dataset :num-rounds 1
                                       :parameters (list :objective token))
        nil)
    (cl-gbdt:foreign-call-error (condition) (cl-gbdt:foreign-call-error-message condition))))

(deftest threads-report-their-own-foreign-errors
  (dolist (backend-name '(:lightgbm :xgboost))
    (with-backend-library (backend-name)
      (testing (format nil "~A: each thread reads its own last-error message" backend-name)
        (multiple-value-bind (matrix labels) (make-separable-dataset)
          ;; ONE backend, opened here on the calling thread and closed only after every
          ;; thread has been joined -- including one that aborted, since `:default :aborted'
          ;; below makes JOIN-THREAD return a sentinel rather than signal and unwind past
          ;; this LET*. Letting an abort unwind here would close the backend while another
          ;; thread was still inside a foreign call -- exactly the hazard this file's header
          ;; says is deliberately not tested. See `%provoke-foreign-error''s docstring for
          ;; why the backend itself is not opened per thread.
          (cl-gbdt:with-backend (backend (cl-gbdt:open-backend backend-name))
           (let* ((tokens (loop :for index :below *thread-count*
                                :collect (%thread-token index)))
                  (threads
                    (loop :for index :below *thread-count*
                          ;; A fresh LET per iteration, deliberately. LOOP's `:for x := ...'
                          ;; steps ONE binding, so every closure below would capture the same
                          ;; variable and read the last token assigned to it -- every thread
                          ;; would then use one token and the "never saw another thread's
                          ;; token" assertion would be vacuously true.
                          :collect (let ((token (nth index tokens)))
                                     (sb-thread:make-thread
                                      (lambda ()
                                        (loop :repeat *error-rounds*
                                              :collect (%provoke-foreign-error
                                                        backend token matrix labels)))
                                      :name (format nil "cl-gbdt-error-~D" index)))))
                  ;; :DEFAULT :ABORTED makes JOIN-THREAD return a sentinel instead of
                  ;; signalling when a thread aborted, so every thread here is actually
                  ;; joined -- and the backend above stays open until all of them have
                  ;; finished, cleanly or not -- before this LET* can unwind at all.
                  (results (mapcar (lambda (thread)
                                      (sb-thread:join-thread thread :default :aborted))
                                    threads)))
            (loop :for index :below *thread-count*
                  :for token := (nth index tokens)
                  :for messages := (nth index results)
                  :do (ok (not (eq messages :aborted))
                          (format nil "thread ~D did not abort" index))
                      (unless (eq messages :aborted)
                        (ok (every (lambda (message)
                                     (and message (search token message)))
                                   messages)
                            (format nil "thread ~D saw its own token in every message"
                                    index))
                        (ok (notany (lambda (message)
                                      (some (lambda (other)
                                              (and (not (string= other token))
                                                   message
                                                   (search other message)))
                                            tokens))
                                    messages)
                            (format nil "thread ~D never saw another thread's token"
                                    index)))))))))))

(defparameter *single-threaded-parameters*
  '((:lightgbm :num-threads 1) (:xgboost :nthread 1))
  "Booster parameters pinning each library to one internal thread, for
`concurrent-training-agrees-with-serial-training'.

Both libraries parallelise internally with OpenMP, and a floating-point reduction's order can
depend on how many threads perform it -- so N Lisp threads each running a multi-threaded
library could differ from a serial run in the last bits for a reason that has nothing to do
with cl-gbdt. Pinning both to one thread removes that variable, and removes the
oversubscription of running *THREAD-COUNT* trainings at once besides.

The key differs by backend on purpose: `num_threads' is LightGBM's spelling and `nthread' is
XGBoost's. Passing one library the other's key is not something this file relies on being
tolerated.")

(defun %train-and-predict (backend matrix labels)
  "Train on BACKEND over MATRIX/LABELS, predict on MATRIX, and return the predictions.

BACKEND is an ALREADY-OPEN backend shared with the other threads, for the reason
`%provoke-foreign-error''s docstring gives. Every DATASET and BOOSTER is this call's own, so
two concurrent calls share only the backend and the input arrays, and the arrays are read and
never written -- which is precisely the arrangement docs/user-guide/threads.md calls safe."
  (let ((parameters (cdr (assoc (cl-gbdt:backend-name backend)
                                *single-threaded-parameters*))))
    (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset backend matrix :label labels))
      (cl-gbdt:with-booster (booster (cl-gbdt:train backend dataset :num-rounds 5
                                                                   :parameters parameters))
        (cl-gbdt:predict booster matrix)))))

(deftest concurrent-training-agrees-with-serial-training
  (dolist (backend-name '(:lightgbm :xgboost))
    (with-backend-library (backend-name)
      (testing (format nil "~A: concurrent results equal serial ones" backend-name)
        (multiple-value-bind (matrix labels) (make-separable-dataset)
          (cl-gbdt:with-backend (backend (cl-gbdt:open-backend backend-name))
            ;; See `threads-report-their-own-foreign-errors' for why JOIN-THREAD is called
            ;; with :DEFAULT :ABORTED below: it keeps every thread joined, and the backend
            ;; above open, until all of them have finished -- cleanly or not -- rather than
            ;; letting an aborted join signal and unwind past this LET* while another thread
            ;; is still inside a foreign call.
            (let* ((serial (%train-and-predict backend matrix labels))
                   (threads (loop :for index :below *thread-count*
                                  :collect (sb-thread:make-thread
                                            (lambda ()
                                              (%train-and-predict backend matrix labels))
                                            :name (format nil "cl-gbdt-train-~D" index))))
                   (concurrent (mapcar (lambda (thread)
                                          (sb-thread:join-thread thread :default :aborted))
                                        threads)))
              (loop :for index :below *thread-count*
                    :for predictions :in concurrent
                    :do (ok (not (eq predictions :aborted))
                            (format nil "thread ~D did not abort" index))
                        (unless (eq predictions :aborted)
                          (ok (predictions-agree-p predictions serial)
                              (format nil "thread ~D's predictions match the serial run"
                                      index)))))))))))
