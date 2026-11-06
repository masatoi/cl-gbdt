(in-package #:cl-user)

(load (merge-pathnames "../common.lisp" *load-truename*))

(load (merge-pathnames "../lifecycle.lisp" *load-truename*))

(in-package #:cl-gbdt-examples)

(defvar *adult-root* (uiop:pathname-directory-pathname *load-truename*))

(defparameter *adult-categorical* '(1 3 5 6 7 8 9 13))

(defun read-adult (path)
  (with-open-file (in path :external-format :utf-8)
    (let ((rows (make-array 0 :adjustable t :fill-pointer 0)))
      (loop for row = (read-csv-row in) while row
            unless (or (null row) (zerop (length (first row)))
                       (char= (char (first row) 0) #\|))
              do (unless (= (length row) 15) (error "Expected 15 Adult columns"))
                 (let ((fields (map 'vector
                                    (lambda (value)
                                      (string-trim '(#\Space #\Tab #\Return) value))
                                    row)))
                   (setf (aref fields 14) (string-right-trim "." (aref fields 14)))
                   (unless (member (aref fields 14) '("<=50K" ">50K") :test #'string=)
                     (error "Unknown income label: ~S" (aref fields 14)))
                   (vector-push-extend fields rows)))
      rows)))

(defun adult-label (row)
  (if (string= (aref row 14) ">50K") 1 0))

(defun adult-split (rows)
  (let ((train nil) (valid nil))
    (dolist (label '(0 1))
      (let* ((indices (coerce (loop for row across rows for i from 0
                                    when (= (adult-label row) label) collect i)
                              'vector))
             (shuffled (shuffle-vector indices (+ 42 label)))
             (nvalid (round (* 0.2d0 (length shuffled)))))
        (loop for index across shuffled for i from 0
              do (if (< i nvalid)
                     (push (aref rows index) valid)
                     (push (aref rows index) train)))))
    (values (shuffle-vector (coerce train 'vector) 44)
            (shuffle-vector (coerce valid 'vector) 45))))

(defun adult-category-maps (train)
  (loop for column in *adult-categorical*
        collect (let ((seen (make-hash-table :test #'equal))
                      (map (make-hash-table :test #'equal)))
                  (loop for row across train for value = (aref row column)
                        unless (string= value "?") do (setf (gethash value seen) t))
                  (loop for value in (sort (loop for key being the hash-keys in seen collect key)
                                          #'string<)
                        for code from 2 do (setf (gethash value map) code))
                  (cons column map))))

(defun write-adult-categories (maps data)
  "Save the training-only category dictionaries for inspecting later predictions."
  (with-open-file (out (merge-pathnames "categories.sexp" data)
                       :direction :output :if-exists :supersede :if-does-not-exist :create)
    (let ((*print-readably* t) (*print-case* :downcase))
      (write (loop for (column . table) in maps
                   collect (cons column
                                 (sort (loop for name being the hash-keys in table
                                             using (hash-value code)
                                             collect (cons name code))
                                       #'< :key #'cdr)))
             :stream out)
      (terpri out))))

(defun write-adult-split (name rows maps)
  (let ((data (merge-pathnames "data/" *adult-root*)))
    (with-open-file (x (merge-pathnames (format nil "~A-x.f32" name) data)
                       :direction :output :if-exists :supersede
                       :element-type '(unsigned-byte 8))
      (with-open-file (y (merge-pathnames (format nil "~A-y.f32" name) data)
                         :direction :output :if-exists :supersede
                         :element-type '(unsigned-byte 8))
        (loop for row across rows
              do (dotimes (column 14)
                   (let* ((value (aref row column))
                          (map (cdr (assoc column maps)))
                          (number (cond
                                    ((and map (string= value "?")) 0)
                                    (map (gethash value map 1))
                                    (t (parse-number value)))))
                     (write-f32 x number)))
                 (write-f32 y (adult-label row)))))
    (when (string= name "test")
      (with-open-file (out (merge-pathnames "test-meta.csv" data)
                           :direction :output :if-exists :supersede)
        (format out "income,sex,race~%")
        (loop for row across rows
              do (format out "~D," (adult-label row))
                 (write-csv-field out (aref row 9))
                 (write-char #\, out)
                 (write-csv-field out (aref row 8))
                 (terpri out))))
    (format t "~A: rows=~D positives=~D~%" name (length rows)
            (count 1 rows :key #'adult-label))))

(defun prepare-adult ()
  (cl-gbdt-example-lifecycle:with-preparation (*adult-root*)
    (ensure-example-directories *adult-root*)
    (let* ((data (merge-pathnames "data/" *adult-root*))
           (source (read-adult (merge-pathnames "adult.data" data)))
           (test (read-adult (merge-pathnames "adult.test" data))))
      (multiple-value-bind (train valid) (adult-split source)
        (let ((maps (adult-category-maps train)))
          (write-adult-categories maps data)
          (write-adult-split "train" train maps)
          (write-adult-split "valid" valid maps)
          (write-adult-split "test" test maps)
          (write-sizes (merge-pathnames "sizes.lisp" data)
                       :train (length train) :valid (length valid) :test (length test)
                       :features 14 :categorical *adult-categorical*))))))

(let ((*adult-root* (uiop:pathname-directory-pathname *load-truename*)))
  (prepare-adult))
