;;;; classes.lisp --- XGBoost's CLOS types, the shared library's lifetime, and `slice-model'.
;;;;
;;;; Layer 1, not Layer 2. `initialize-backend' and `shutdown-backend' are generic functions
;;;; declared in `src/backend.lisp' -- the shared basis every layer is written against -- not
;;;; in `src/protocol.lisp', so implementing them is backend work that owes nothing to the
;;;; unified API. Keeping them here is what lets `cl-gbdt/xgboost' open and close the library
;;;; without loading Layer 2 at all.
;;;;
;;;; `slice-model' is here for a reason of its own, and was Layer 1 already while it sat at
;;;; the end of `protocol.lisp': it is a public XGBoost-specific entry point, never a protocol
;;;; method, and it builds a booster handle, so it must name the concrete `xgboost-booster'
;;;; class -- which is defined right here. That is also why it cannot live in `native.lisp'
;;;; beside its siblings `evaluate-one-iteration' and `booster-boosted-rounds': that file must
;;;; not depend on this one. See its own Model slicing section below.
;;;;
;;;; Loads after `native.lisp' and cannot precede it: `initialize-backend' reads that file's
;;;; `*required-symbols*', `*optional-symbols*', `*provided-capabilities*', `*library-env-var*',
;;;; `*vendor-library-directory*', `*vendor-library-pattern*' and `*default-library-name*' and
;;;; calls its `%read-version', and `slice-model' calls its `%check-xgboost-booster',
;;;; `%check-unsupported', `%slice' and `%free-booster-unchecked'.
;;;;
;;;; Every form here that reaches the shared library wraps its whole body in
;;;; `with-foreign-float-traps-masked', for the reason `cl-gbdt/src/xgboost/protocol''s
;;;; "Floating-point trap safety" comment gives. Moving a form between files never removes
;;;; that wrap -- but it can move a form out from under the check that verifies it, and it did
;;;; here: `tools/ci/check-float-traps.lisp''s +BACKEND-FILE-PATTERNS+ names `src/*/native.lisp'
;;;; and `src/*/protocol.lisp' only, so `slice-model''s wrap, which that scan counted while the
;;;; function lived in `protocol.lisp', is held by reading alone until this path is added there.

(uiop:define-package #:cl-gbdt/src/xgboost/classes
  (:use #:cl)
  (:import-from #:cffi)
  (:import-from #:cl-gbdt/src/xgboost/native
                #:*library-env-var*
                #:*vendor-library-directory*
                #:*vendor-library-pattern*
                #:*default-library-name*
                #:*required-symbols*
                #:*optional-symbols*
                #:*provided-capabilities*
                #:%read-version
                #:%check-unsupported
                #:%check-xgboost-booster
                #:%free-booster-unchecked
                #:%slice)
  (:import-from #:cl-gbdt/src/backend
                #:backend
                #:backend-name
                #:backend-library-path
                #:backend-version
                #:backend-capabilities
                #:backend-supports-p
                #:probe-foreign-symbols
                #:probe-capabilities
                #:register-backend
                #:initialize-backend
                #:shutdown-backend)
  (:import-from #:cl-gbdt/src/handle
                #:dataset
                #:booster
                #:with-pointer-ownership
                #:handle-backend)
  (:import-from #:cl-gbdt/src/conditions
                #:missing-foreign-symbols
                #:capability-unavailable)
  (:import-from #:cl-gbdt/src/library
                #:resolve-and-load-library)
  (:import-from #:cl-gbdt/src/foreign
                #:with-foreign-float-traps-masked)
  (:import-from #:cl-gbdt/src/version
                #:check-backend-version
                #:*xgboost-version-range*)
  (:export #:xgboost-backend
           #:xgboost-dataset
           #:xgboost-booster
           #:%xgboost-foreign-library
           #:slice-model))

(in-package #:cl-gbdt/src/xgboost/classes)

;;; ---------------------------------------------------------------------------
;;; The backend class

(defclass xgboost-backend (backend)
  ((foreign-library :initform nil
                     :accessor %xgboost-foreign-library
                     :documentation "The `cffi:foreign-library' `initialize-backend'
loaded, kept so `shutdown-backend' can close exactly this one."))
  (:documentation "A connection to the XGBoost shared library, implementing cl-gbdt's
unified backend protocol."))

(register-backend :xgboost 'xgboost-backend)

;;; Handles must be subclassed per backend, not shared -- see
;;; `cl-gbdt/src/lightgbm/classes''s identical commentary above `lightgbm-dataset'.
;;; `dataset-num-rows', `predict', `free-dataset' and `free-booster' dispatch on the
;;; HANDLE, so a method on the core `dataset' or `booster' class would be replaced, not
;;; specialized, the moment the other backend defined its own -- an XGBoost dataset would
;;; silently answer through LightGBM's C API, or vice versa.

(defclass xgboost-dataset (dataset) ()
  (:documentation "A dataset (a DMatrix) held by the XGBoost library."))

(defclass xgboost-booster (booster) ()
  (:documentation "A booster held by the XGBoost library."))

(defmethod initialize-backend ((backend xgboost-backend) &key path)
  "Load XGBoost's shared library and record its capabilities on BACKEND.

Discovery order: PATH, then *library-env-var*, then the vendored directory under
*vendor-library-directory*, then CFFI's system library search for
*default-library-name* -- see `resolve-and-load-library' for the exact rules and the
conditions each failure mode signals.

Once a library is loaded, every name in *required-symbols* must resolve via
`probe-foreign-symbols', passed the `cffi:foreign-library' just loaded as :LIBRARY -- see
that function's docstring for the SBCL caveat: it validates the library argument but,
on this platform, cannot actually scope the symbol search to it -- or this signals
`missing-foreign-symbols'.

Only once that required check has passed does this probe *optional-symbols* via
`probe-capabilities' and record the result on `backend-capabilities' -- unlike a missing
required symbol, a missing optional one never signals; it only makes `backend-supports-p'
answer NIL for the capability that symbol backs. *provided-capabilities* goes to the same
call as :PROVIDED, recording the capabilities this backend provides unconditionally --
nothing is probed for them, because the C functions they need are in *required-symbols* and
the probe above has already passed.

Once `backend-version' is read, `check-backend-version' compares it against
`*xgboost-version-range*' and signals `untested-backend-version' -- a warning, not an
error -- when it falls outside that range's recorded bounds. This is the only backend
this runs for: LightGBM's C API has no version entry point, so `backend-version' stays
NIL there and there is nothing to compare -- see
`cl-gbdt/src/lightgbm/classes''s `initialize-backend'.

`open-backend' only marks a backend open -- and so only calls `close-backend' on it --
once this method returns normally. So if the symbol probe (or anything else after the
library loads) signals, the library is closed right here before the condition propagates;
otherwise it would stay mapped into the process with BACKEND dropped and nothing left able
to close it."
  (with-foreign-float-traps-masked
    (multiple-value-bind (library library-path)
        (resolve-and-load-library backend :path path
                                           :env-var *library-env-var*
                                           :directory *vendor-library-directory*
                                           :pattern *vendor-library-pattern*
                                           :default-name *default-library-name*)
      (let ((succeeded nil))
        (unwind-protect
             (progn
               (setf (%xgboost-foreign-library backend) library)
               (setf (backend-library-path backend) library-path)
               (let ((missing (probe-foreign-symbols *required-symbols* :library library)))
                 (when missing
                   (error 'missing-foreign-symbols
                          :backend (backend-name backend) :names missing)))
               (setf (backend-capabilities backend)
                     (probe-capabilities *optional-symbols*
                                         :provided *provided-capabilities*
                                         :library library))
               (setf (backend-version backend) (%read-version))
               (check-backend-version :xgboost (backend-version backend)
                                       *xgboost-version-range*)
               (setf succeeded t))
          (unless succeeded
            (handler-case (cffi:close-foreign-library library)
              (error () nil))
            (setf (%xgboost-foreign-library backend) nil)))
        backend))))

(defmethod shutdown-backend ((backend xgboost-backend))
  "Close XGBoost's shared library.

`cffi:close-foreign-library' drops cl-gbdt's own reference and, on platforms where the C
loader honors `dlclose' reference counting, may unmap the library; POSIX does not
guarantee an actual unload, so this cannot promise the library's code and data are gone
from the process afterward -- only that cl-gbdt no longer holds it open."
  (with-foreign-float-traps-masked
    (let ((library (%xgboost-foreign-library backend)))
      (when library
        (cffi:close-foreign-library library)
        (setf (%xgboost-foreign-library backend) nil)))
    backend))

;;; ---------------------------------------------------------------------------
;;; Model slicing
;;;
;;; `slice-model' is the one function in this file that is not part of the backend's own
;;; library lifetime, and the only Layer 1 entry point that does not live in
;;; `cl-gbdt/src/xgboost/native' beside its siblings `evaluate-one-iteration' and
;;; `booster-boosted-rounds'. It is here because it returns a NEW booster: `make-handle'
;;; needs the concrete class `xgboost-booster', which is defined in this file, and
;;; `native.lisp' must not depend on this one (policy section 11). Every other `make-handle'
;;; call in this project is in a `protocol.lisp' for exactly that reason --
;;; `cl-gbdt/src/xgboost/protocol''s `load-model' is the closest sibling, and this follows its
;;; shape: the guards and the handle construction here, the foreign call delegated to a
;;; `%'-function in `native.lisp' (`%slice'). See that file's own Model slicing section for
;;; the other half.
;;;
;;; Deliberately NOT a generic function in `cl-gbdt/src/protocol'. Section 4's criterion for
;;; the unified API is that both backends can mean the same thing by an operation; LightGBM
;;; has no counterpart to `XGBoosterSlice', so a portable `slice-model' would either signal
;;; on one backend for every caller or emulate -- and emulation is the silent fallback
;;; section 7 forbids. It is published from `cl-gbdt/xgboost' instead, which is what section
;;; 3's Layer 1 and section 11 are for.

(defun slice-model (booster &key (begin 0) end (step 1))
  "Return a new booster holding BOOSTER's layers from BEGIN to END, taken STEP at a time.

The interval is HALF-OPEN, `[BEGIN, END)': END names the first layer left out, so slicing a
ten-round booster with `:begin 0 :end 5' gives five rounds, not six. Measured against the
vendored libxgboost, whose header documents no interval semantics at all; XGBoost's own
rejection of `:begin 5 :end 5' as \"Empty slice is not allowed\" is the same reading from
the other side.

END defaults to NIL, meaning through the last layer, and is passed to `XGBoosterSlice' as
its own 0. NIL rather than 0 in Lisp because a caller writing `:END 0' means \"nothing\",
and silently reading that as \"everything\" is the kind of translation policy section 5
exists to prevent -- so an explicit `:END 0' signals `unsupported-argument' rather than
being forwarded to a C 0 that would mean the opposite. Every other out-of-range request is
XGBoost's own to refuse, and it does, with `foreign-call-error': END past the last layer,
BEGIN below zero, STEP below one, and a STEP that does not divide the interval evenly.

The returned booster belongs to the caller, who frees it with `free-booster'. It is
INDEPENDENT of BOOSTER: `XGBoosterSlice' copies the layers it selects, so freeing BOOSTER
first is legitimate, and the slice keeps predicting the same values afterward -- verified
against the vendored library with BOOSTER and the DMatrix it was trained on both freed. It
therefore retains no parent, exactly as a `load-model' booster retains no training set;
retaining one anyway would make freeing BOOSTER signal `released-handle-error' on correct
code. For the same reason the slice has no training set of its own, so `evaluation' and
`update-one-iteration' on it behave as they do for a `load-model' booster.

Signals `wrong-backend-reference' when BOOSTER was not built by the XGBoost backend,
`released-handle-error' when it has been freed, `backend-not-open' when its backend has
been closed, `capability-unavailable' when the loaded library has no `XGBoosterSlice',
`unsupported-argument' for an explicit `:END 0', and `foreign-call-error' when the slice
itself fails.

The capability is re-checked here rather than assumed: policy section 7 requires the
operation to signal for itself, so a caller who never asked `backend-supports-p' gets a
typed condition instead of a missing-symbol crash. The handle check runs first, before the
capability check, so handing this a LightGBM booster reports the wrong handle rather than
the true-but-irrelevant news that the backend it came from cannot slice."
  (with-foreign-float-traps-masked
    (let ((pointer (%check-xgboost-booster booster "slice-model's booster argument"))
          (backend (handle-backend booster)))
      (unless (backend-supports-p backend :model-slicing)
        (error 'capability-unavailable
               :backend (backend-name backend) :capability :model-slicing))
      (%check-unsupported
       backend "slice-model's :END of 0" (eql end 0)
       (format nil "0 would be an empty slice, which XGBoost rejects outright, but ~
                    XGBoosterSlice's own end_layer 0 means the last layer -- pass NIL for ~
                    that rather than letting the two readings collide"))
      ;; Unlike `cl-gbdt/src/xgboost/protocol''s `load-model' and `train', no further foreign
      ;; call runs between the handle appearing in C and `make-handle' taking ownership of it
      ;; -- `%slice' returns a booster that is already complete, and `make-handle' is the very
      ;; next thing that runs.
      ;; The `owned' unwind-protect dance is still needed, though: `make-handle' itself --
      ;; `make-instance' or finalizer attachment -- can signal, e.g. on `storage-condition',
      ;; and a signal there must not orphan the foreign booster `%slice' already returned.
      (let ((slice-pointer (%slice pointer begin (or end 0) step)))
        (with-pointer-ownership (slice-pointer #'%free-booster-unchecked take-ownership)
          (take-ownership 'xgboost-booster backend :booster))))))
