(in-package #:cl-user)

(load (merge-pathnames "../common.lisp" *load-truename*))
(in-package #:cl-gbdt-examples)

(defparameter *fraud-root* (uiop:pathname-directory-pathname *load-truename*))

(defstruct fraud-row time values label hash)

(defun read-fraud-data (path)
  (let ((rows (make-array 0 :adjustable t :fill-pointer 0))
        (frauds 0))
    (with-open-file (in path)
      (let ((expected (append '("Time")
                              (loop for i from 1 to 28 collect (format nil "V~D" i))
                              '("Amount" "Class"))))
        (unless (equal (read-csv-row in) expected) (error "Unexpected credit card columns")))
      (loop for fields = (read-csv-row in) while fields
            do (unless (= (length fields) 31) (error "Unexpected credit card row width"))
               (let ((time (parse-number (first fields)))
                     (values (make-array 29 :element-type 'single-float))
                     (hash #xcbf29ce484222325)
                     (label (parse-integer (nth 30 fields))))
                 (unless (and (finite-real-p time) (member label '(0 1)))
                   (error "Invalid transaction time or label"))
                 (loop for field in (subseq fields 1 30) for j from 0
                       for value = (parse-number field)
                       do (unless (finite-real-p value) (error "Non-finite transaction"))
                          (setf (aref values j) (coerce value 'single-float)
                                hash (fnv1a field hash)
                                hash (fnv1a "|" hash)))
                 (incf frauds label)
                 (vector-push-extend (make-fraud-row :time time :values values
                                                      :label label :hash hash)
                                     rows))))
    (unless (and (= (length rows) 284807) (= frauds 492))
      (error "Expected 284807 transactions and 492 frauds, got ~D and ~D" (length rows) frauds))
    (stable-sort rows #'< :key #'fraud-row-time)))

(defun fraud-boundary (rows fraction)
  (let* ((n (length rows))
         (index (floor (* n fraction)))
         (time (fraud-row-time (aref rows index))))
    (loop while (and (< index n) (= (fraud-row-time (aref rows index)) time))
          do (incf index))
    (unless (< 0 index n) (error "Empty partition after moving the timestamp boundary"))
    index))

(defun fraud-duplicate-count (rows)
  "Count full-row hash matches; the 64-bit feature hash may collide."
  (let ((seen (make-hash-table :test #'equal)) (duplicates 0))
    (loop for row across rows
          for key = (list (fraud-row-time row) (fraud-row-hash row) (fraud-row-label row))
          do (if (gethash key seen)
                 (incf duplicates)
                 (setf (gethash key seen) t)))
    duplicates))

(defun write-fraud-partition (name rows prior)
  (let* ((data (merge-pathnames "data/" *fraud-root*))
         (y-path (merge-pathnames (format nil "~A-y.f32" name) data))
         (x-path (merge-pathnames (format nil "~A-x.f32" name) data))
         (seen-path (merge-pathnames (format nil "~A-seen.u8" name) data))
         (seen-count 0) (seen-frauds 0) (frauds 0))
    (with-open-file (x x-path :direction :output :if-exists :supersede
                              :element-type '(unsigned-byte 8))
      (with-open-file (y y-path :direction :output :if-exists :supersede
                                :element-type '(unsigned-byte 8))
        (with-open-file (seen seen-path :direction :output :if-exists :supersede
                                        :element-type '(unsigned-byte 8))
          (loop for row across rows
                for matched = (gethash (fraud-row-hash row) prior)
                do (loop for value across (fraud-row-values row) do (write-f32 x value))
                   (write-f32 y (fraud-row-label row))
                   (write-byte (if matched 1 0) seen)
                   (incf frauds (fraud-row-label row))
                   (when matched
                     (incf seen-count)
                     (incf seen-frauds (fraud-row-label row)))))))
    (loop for row across rows do (setf (gethash (fraud-row-hash row) prior) t))
    (format t "~A: rows=~D frauds=~D time=~A..~A prior_feature_matches=~D (~D frauds)~%"
            name (length rows) frauds (fraud-row-time (aref rows 0))
            (fraud-row-time (aref rows (1- (length rows)))) seen-count seen-frauds)
    (list :rows (length rows) :frauds frauds :min-time (fraud-row-time (aref rows 0))
          :max-time (fraud-row-time (aref rows (1- (length rows))))
          :prior-feature-matches seen-count :prior-feature-match-frauds seen-frauds)))

(defun prepare-fraud ()
  (ensure-example-directories *fraud-root*)
  (let* ((data (merge-pathnames "data/" *fraud-root*))
         (source (or (uiop:getenv "CL_GBDT_CREDIT_CSV")
                     (namestring (merge-pathnames "creditcard.csv" data))))
         (rows (read-fraud-data source))
         (duplicates (fraud-duplicate-count rows))
         (i (fraud-boundary rows 0.6d0))
         (j (fraud-boundary rows 0.8d0))
         (train (subseq rows 0 i)) (valid (subseq rows i j)) (test (subseq rows j))
         (prior (make-hash-table :test #'eql))
         (train-info (write-fraud-partition "train" train prior))
         (valid-info (write-fraud-partition "valid" valid prior))
         (test-info (write-fraud-partition "test" test prior)))
    (write-sizes (merge-pathnames "sizes.lisp" data)
                 :train (length train) :valid (length valid) :test (length test) :features 29)
    (with-open-file (out (merge-pathnames "metadata.sexp" data)
                         :direction :output :if-exists :supersede :if-does-not-exist :create)
      (let ((*print-readably* t) (*print-case* :downcase))
        (write (list :source source :rows (length rows) :frauds 492 :features 29
                     :duplicate-full-row-hashes duplicates
                     :train train-info :valid valid-info :test test-info)
               :stream out)
        (terpri out)))
    (format t "full-row hash matches: ~D~%" duplicates)))

(prepare-fraud)
