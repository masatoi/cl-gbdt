(in-package #:cl-user)

(load (merge-pathnames "../common.lisp" *load-truename*))

(load (merge-pathnames "../lifecycle.lisp" *load-truename*))

(in-package #:cl-gbdt-examples)

(defvar *retail-root* (uiop:pathname-directory-pathname *load-truename*))

(defstruct retail-transaction customer invoice stock quantity price date country)
(defstruct retail-snapshot customer last invoices products spend quantity price-sum lines
           recent-spend country target)

(defun retail-number (value)
  (typecase value
    (real value)
    (string (parse-number value))
    (t (error "Invalid spreadsheet number: ~S" value))))

(defun retail-day (year month day)
  (+ 2 (/ (encode-universal-time 0 0 0 day month year 0) 86400)))

(defun read-retail-transactions (path)
  (let ((transactions (make-array 0 :adjustable t :fill-pointer 0))
        (sheet (lisp-xl:read-sheet path 1 :silent t))
        (rows 0) (*read-eval* nil))
    (unwind-protect
         (lisp-xl:process-sheet
          sheet :initial-row 2 :silent t
          :row-function
          (lambda (fields)
            (incf rows)
            (destructuring-bind (invoice stock description quantity date price customer country
                                  &rest rest)
                fields
              (declare (ignore description rest))
              (when (and invoice stock quantity date price customer country
                         (plusp (length (princ-to-string invoice)))
                         (not (char-equal (char (princ-to-string invoice) 0) #\C)))
                (let ((units (retail-number quantity)) (unit-price (retail-number price)))
                  (when (and (> units 0) (> unit-price 0))
                    (vector-push-extend
                     (make-retail-transaction
                      :customer (round (retail-number customer))
                      :invoice (princ-to-string invoice) :stock (princ-to-string stock)
                      :quantity units :price unit-price
                      :date (round (* 1440 (retail-number date)))
                      :country (princ-to-string country))
                     transactions)))))))
      (lisp-xl:close-sheet sheet))
    (format t "spreadsheet rows=~D qualifying transactions=~D~%" rows (length transactions))
    (when (zerop (length transactions)) (error "No qualifying transactions"))
    (stable-sort transactions #'< :key #'retail-transaction-date)))

(defun retail-snapshot-at (transactions when-day)
  (let ((when-minute (* 1440 when-day))
        (customers (make-hash-table))
        (future (make-hash-table)))
    (unless (<= (+ when-minute (* 30 1440))
                (retail-transaction-date (aref transactions (1- (length transactions)))))
      (error "Insufficient observation horizon"))
    (loop for txn across transactions
          for date = (retail-transaction-date txn)
          for customer = (retail-transaction-customer txn)
          do (cond
               ((and (<= (- when-minute (* 90 1440)) date) (< date when-minute))
                (let ((row (or (gethash customer customers)
                               (setf (gethash customer customers)
                                     (make-retail-snapshot
                                      :customer customer :invoices (make-hash-table :test #'equal)
                                      :products (make-hash-table :test #'equal)
                                      :spend 0d0 :quantity 0 :price-sum 0d0 :lines 0
                                      :recent-spend 0d0)))))
                  (setf (retail-snapshot-last row) date
                        (retail-snapshot-country row) (retail-transaction-country txn)
                        (gethash (retail-transaction-invoice txn)
                                 (retail-snapshot-invoices row)) t
                        (gethash (retail-transaction-stock txn)
                                 (retail-snapshot-products row)) t)
                  (let ((amount (* (retail-transaction-quantity txn)
                                   (retail-transaction-price txn))))
                    (incf (retail-snapshot-spend row) amount)
                    (when (>= date (- when-minute (* 30 1440)))
                      (incf (retail-snapshot-recent-spend row) amount)))
                  (incf (retail-snapshot-quantity row) (retail-transaction-quantity txn))
                  (incf (retail-snapshot-price-sum row) (retail-transaction-price txn))
                  (incf (retail-snapshot-lines row))))
               ((and (<= when-minute date) (< date (+ when-minute (* 30 1440))))
                (setf (gethash customer future) t))))
    (let ((rows (sort (loop for row being the hash-values in customers collect row)
                      #'< :key #'retail-snapshot-customer)))
      (dolist (row rows)
        (setf (retail-snapshot-target row)
              (if (gethash (retail-snapshot-customer row) future) 1 0)
              (retail-snapshot-last row) (/ (- when-minute (retail-snapshot-last row)) 1440d0)))
      rows)))

(defun retail-country-map (train)
  (let ((countries (make-hash-table :test #'equal)) (map (make-hash-table :test #'equal)))
    (dolist (row train) (setf (gethash (retail-snapshot-country row) countries) t))
    (loop for name in (sort (loop for key being the hash-keys in countries collect key) #'string<)
          for code from 1 do (setf (gethash name map) code))
    map))

(defun write-retail-part (name rows countries &optional weights)
  (let ((data (merge-pathnames "data/" *retail-root*)))
    (with-open-file (x (merge-pathnames (format nil "~A-x.f32" name) data)
                       :direction :output :if-exists :supersede :element-type '(unsigned-byte 8))
      (with-open-file (y (merge-pathnames (format nil "~A-y.f32" name) data)
                         :direction :output :if-exists :supersede :element-type '(unsigned-byte 8))
        (dolist (row rows)
          (dolist (value (list (retail-snapshot-last row)
                               (hash-table-count (retail-snapshot-invoices row))
                               (retail-snapshot-spend row)
                               (retail-snapshot-quantity row)
                               (hash-table-count (retail-snapshot-products row))
                               (/ (retail-snapshot-price-sum row) (retail-snapshot-lines row))
                               (retail-snapshot-recent-spend row)
                               (gethash (retail-snapshot-country row) countries 0)))
            (write-f32 x value))
          (write-f32 y (retail-snapshot-target row)))))
    (when weights
      (with-open-file (w (merge-pathnames "train-w.f32" data)
                         :direction :output :if-exists :supersede :element-type '(unsigned-byte 8))
        (loop for weight across weights do (write-f32 w weight))))
    (format t "~A: rows=~D positives=~D~%" name (length rows)
            (count 1 rows :key #'retail-snapshot-target))))

(defun prepare-retail ()
  (cl-gbdt-example-lifecycle:with-preparation (*retail-root*)
    (ensure-example-directories *retail-root*)
    (let* ((data (merge-pathnames "data/" *retail-root*))
           (transactions (read-retail-transactions (merge-pathnames "Online Retail.xlsx" data)))
           (dates (mapcar (lambda (month) (retail-day 2011 month 1)) '(4 5 6 7 8 9 11)))
           (snapshots (mapcar (lambda (date) (retail-snapshot-at transactions date)) dates))
           (train (apply #'append (subseq snapshots 0 5)))
           (valid (nth 5 snapshots)) (test (nth 6 snapshots))
           (countries (retail-country-map train))
           (frequency (make-hash-table))
           (weights (make-array (length train) :element-type 'double-float))
           (total 0d0))
      (unless (and (<= (+ (nth 4 dates) 30) (nth 5 dates))
                   (<= (+ (nth 5 dates) 30) (nth 6 dates)))
        (error "Label horizons overlap the next split"))
      (dolist (row train) (incf (gethash (retail-snapshot-customer row) frequency 0)))
      (loop for row in train for i from 0
            do (setf (aref weights i) (/ 1d0 (gethash (retail-snapshot-customer row) frequency)))
               (incf total (aref weights i)))
      (loop for i below (length weights) do (setf (aref weights i) (* (aref weights i)
                                                                      (/ (length weights) total))))
      (write-retail-part "train" train countries weights)
      (write-retail-part "valid" valid countries)
      (write-retail-part "test" test countries)
      (write-sizes (merge-pathnames "sizes.lisp" data)
                   :train (length train) :valid (length valid) :test (length test)
                   :features 8 :categorical '(7)))))

(let ((*retail-root* (uiop:pathname-directory-pathname *load-truename*)))
  (prepare-retail))
