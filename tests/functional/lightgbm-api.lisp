;;;; lightgbm-api.lisp --- Round trip through cl-gbdt's unified API, LightGBM backend.
;;;;
;;;; tests/functional/lightgbm.lisp calls the raw FFI bindings directly, proving the
;;;; generated bindings match the LightGBM ABI. This file never touches
;;;; cl-gbdt/src/lightgbm/c-api or the FFI package's `lgbm::' symbols; it goes through
;;;; `open-backend', `with-dataset' + `make-dataset', `with-booster' + `train',
;;;; `predict' and `close-backend' exactly as any caller of cl-gbdt would, and stays
;;;; green only because the twelve LightGBM protocol methods actually do what the
;;;; unified API promises.

(uiop:define-package #:cl-gbdt/tests/functional/lightgbm-api
  (:use #:cl #:rove)
  (:import-from #:cl-gbdt)
  ;; Zero symbols: nothing below refers to this package by name. Its only job is to
  ;; run at load time and register :lightgbm with `open-backend' -- see
  ;; `register-backend' at the bottom of src/lightgbm/backend.lisp. Without this
  ;; clause, package-inferred-system has no edge to that file at all, and
  ;; `(cl-gbdt:open-backend :lightgbm)' below would signal `unknown-backend'.
  (:import-from #:cl-gbdt/src/lightgbm/backend)
  (:import-from #:cl-gbdt/tests/functional/support
                #:with-backend-library
                #:make-separable-dataset
                #:predictions-separate-p))

(in-package #:cl-gbdt/tests/functional/lightgbm-api)

(defparameter *dataset-parameters*
  '(:min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1)
  "LightGBM dataset parameters for the eight-row fixture, as the plist `make-dataset'
accepts. The two minima are required: the defaults refuse to bin or split so few
rows, which would leave every prediction identical and fail the separation
assertion for a reason that has nothing to do with the unified API. `verbose -1'
keeps the library off standard output during the suite. See
tests/functional/lightgbm.lisp's *dataset-parameters* for the same rule stated
against the raw FFI.")

(defparameter *booster-parameters*
  '(:objective "binary" :num-leaves 2 :min-data-in-leaf 1 :min-data-in-bin 1 :verbose -1)
  "LightGBM booster parameters for the eight-row fixture. See *dataset-parameters*.")

(defun %column (matrix column)
  "Return COLUMN of the 2D array MATRIX as a fresh `(simple-array double-float (*))'.

`predictions-separate-p' takes a 1D sequence; `predict' returns a 2D array, one row
per input row and one column per class. The fixture's objective is binary, so
COLUMN is always 0, but the shape still has to be unpacked by hand."
  (let* ((rows (array-dimension matrix 0))
         (result (make-array rows :element-type 'double-float)))
    (dotimes (row rows result)
      (setf (aref result row) (aref matrix row column)))))

(deftest lightgbm-api-round-trip
  (with-backend-library (:lightgbm)
    (multiple-value-bind (matrix label-vector) (make-separable-dataset)
      (let ((rows (array-dimension matrix 0))
            (cols (array-dimension matrix 1))
            (backend (cl-gbdt:open-backend :lightgbm)))
        (testing "open-backend marks the backend open"
          (ok (cl-gbdt:backend-open-p backend)))
        (unwind-protect
             (cl-gbdt:with-dataset (dataset (cl-gbdt:make-dataset
                                              backend matrix :label label-vector
                                              :parameters *dataset-parameters*))
               (testing "dataset-num-rows and dataset-num-features match the fixture"
                 (ok (= rows (cl-gbdt:dataset-num-rows dataset))
                     (format nil "dataset-num-rows is ~D, expected ~D"
                             (cl-gbdt:dataset-num-rows dataset) rows))
                 (ok (= cols (cl-gbdt:dataset-num-features dataset))
                     (format nil "dataset-num-features is ~D, expected ~D"
                             (cl-gbdt:dataset-num-features dataset) cols)))

               (cl-gbdt:with-booster (booster (cl-gbdt:train
                                                backend dataset :num-rounds 5
                                                :parameters *booster-parameters*))
                 (let ((predictions (cl-gbdt:predict booster matrix)))
                   (testing "predict returns a 2D double-float array with the fixture's rows"
                     (ok (typep predictions '(simple-array double-float (* *)))
                         (format nil "predict's element type is ~A"
                                 (array-element-type predictions)))
                     (ok (= rows (array-dimension predictions 0))
                         (format nil "predict's row count is ~D, expected ~D"
                                 (array-dimension predictions 0) rows)))

                   (testing "predictions separate along the label boundary"
                     (ok (predictions-separate-p (%column predictions 0) label-vector)
                         (format nil "predictions: ~S" predictions)))

                   (testing "feature-importance has one entry per feature"
                     (let ((importance (cl-gbdt:feature-importance booster)))
                       (ok (= cols (length importance))
                           (format nil "feature-importance length is ~D, expected ~D"
                                   (length importance) cols))))

                   (testing "save-model then load-model reproduces the original predictions"
                     (uiop:with-temporary-file (:pathname path :type "txt")
                       (cl-gbdt:save-model booster path)
                       (cl-gbdt:with-booster (loaded (cl-gbdt:load-model backend path))
                         (ok (equalp predictions (cl-gbdt:predict loaded matrix))
                             "reloaded predictions equal the original's, elementwise")))))))
          (cl-gbdt:close-backend backend))
        (testing "close-backend marks the backend closed"
          (ng (cl-gbdt:backend-open-p backend)))))))
