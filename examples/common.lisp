(in-package #:cl-user)

(defpackage #:cl-gbdt-examples
  (:use #:cl)
  (:export #:parse-number #:split-fields #:read-csv-row #:read-scores
           #:write-f32 #:write-i32 #:read-f32 #:write-sizes #:write-groups
           #:shuffle-vector #:split-indexes #:binary-metrics #:best-threshold
           #:ranking-ndcg #:read-label-csv #:finite-real-p #:fnv1a
           #:ensure-example-directories #:write-csv-field
           #:fetch-example-data #:ensure-retail-reader))

(in-package #:cl-gbdt-examples)

(defun parse-number (text)
  "Read a complete decimal number without allowing reader evaluation."
  (let ((*read-eval* nil)
        (*read-default-float-format* 'double-float))
    (multiple-value-bind (value end) (read-from-string text nil nil)
      (unless (and (realp value)
                   (every (lambda (ch) (find ch '(#\Space #\Tab #\Return)))
                          (subseq text end)))
        (error "Not a number: ~S" text))
      value)))

(defun finite-real-p (number)
  (and (realp number)
       (or (not (floatp number))
           (and (not (sb-ext:float-nan-p number))
                (not (sb-ext:float-infinity-p number))))))

(defun split-fields (line delimiter)
  "Split an unquoted delimited LINE, retaining empty fields."
  (loop with start = 0
        for end = (position delimiter line :start start)
        collect (subseq line start end)
        while end do (setf start (1+ end))))

(defun read-csv-row (stream &optional (delimiter #\,))
  "Read a CSV record, including quoted delimiters, quotes, and embedded newlines."
  (let ((fields nil)
        (field (make-string-output-stream))
        (quoted nil)
        (started nil))
    (labels ((finish-field ()
               (push (get-output-stream-string field) fields))
             (finish-row ()
               (finish-field)
               (nreverse fields)))
      (loop for ch = (read-char stream nil nil)
            do (cond
                 ((null ch)
                  (when quoted (error "Unterminated quoted CSV field"))
                  (return (when (or started fields) (finish-row))))
                 (quoted
                  (if (char= ch #\")
                      (if (eql (peek-char nil stream nil nil) #\")
                          (progn (read-char stream) (write-char #\" field))
                          (setf quoted nil))
                      (write-char ch field)))
                 ((char= ch #\") (setf quoted t started t))
                 ((char= ch delimiter) (finish-field) (setf started t))
                 ((char= ch #\Newline) (return (finish-row)))
                 ((char= ch #\Return)
                  (when (eql (peek-char nil stream nil nil) #\Newline) (read-char stream))
                  (return (finish-row)))
                 (t (write-char ch field) (setf started t)))))))

(defun write-csv-field (stream value)
  (let ((text (princ-to-string value)))
    (write-char #\" stream)
    (loop for ch across text
          do (when (char= ch #\") (write-char #\" stream))
             (write-char ch stream))
    (write-char #\" stream)))

(defun read-scores (path)
  (with-open-file (in path :direction :input)
    (let ((values (make-array 0 :element-type 'double-float :adjustable t :fill-pointer 0)))
      (loop for line = (read-line in nil nil)
            while line
            for value = (parse-number line)
            do (unless (finite-real-p value) (error "Non-finite score in ~A" path))
               (vector-push-extend (coerce value 'double-float) values))
      values)))

(defun write-i32 (stream value)
  "Write an integer in little-endian two's-complement form."
  (unless (typep value '(signed-byte 32)) (error "Not int32: ~S" value))
  (let ((bits (ldb (byte 32 0) value)))
    (dotimes (i 4)
      (write-byte (ldb (byte 8 (* i 8)) bits) stream))))

(defun write-f32 (stream value)
  "Write an IEEE-754 single float in little-endian order on SBCL."
  (unless (finite-real-p value) (error "Non-finite input: ~S" value))
  (write-i32 stream (sb-kernel:single-float-bits (coerce value 'single-float))))

(defun read-f32 (path)
  "Read a little-endian float32 file into a fresh vector."
  (with-open-file (in path :element-type '(unsigned-byte 8))
    (let ((size (file-length in)))
      (unless (zerop (mod size 4)) (error "Truncated float32 file: ~A" path))
      (let ((values (make-array (/ size 4) :element-type 'single-float)))
        (dotimes (i (length values) values)
          (let ((bits (loop for shift from 0 below 32 by 8
                            sum (ash (or (read-byte in nil nil)
                                         (error "Truncated float32 file: ~A" path))
                                     shift))))
            (setf (aref values i)
                  (sb-kernel:make-single-float
                   (if (logbitp 31 bits) (- bits (ash 1 32)) bits)))))))))

(defun write-sizes (path &rest plist)
  (with-open-file (out path :direction :output :if-exists :supersede
                            :if-does-not-exist :create)
    (let ((*print-case* :downcase) (*print-readably* t))
      (write plist :stream out)
      (terpri out))))

(defun write-groups (path sizes)
  (with-open-file (out path :direction :output :if-exists :supersede
                            :if-does-not-exist :create)
    (let ((*print-readably* t))
      (write (coerce sizes 'list) :stream out)
      (terpri out))))

(defun shuffle-vector (vector seed)
  "Return a reproducibly shuffled copy using a specified 32-bit xorshift generator."
  (let ((result (copy-seq vector))
        (state (if (zerop seed) 1 (logand seed #xffffffff))))
    (labels ((next-value ()
               (setf state (logand #xffffffff (logxor state (ash state 13)))
                     state (logand #xffffffff (logxor state (ash state -17)))
                     state (logand #xffffffff (logxor state (ash state 5))))))
      (loop for i downfrom (1- (length result)) above 0
            for j = (mod (next-value) (1+ i))
            do (rotatef (aref result i) (aref result j))))
    result))

(defun split-indexes (count seed &optional (train-fraction 0.6d0) (valid-fraction 0.2d0))
  (let* ((shuffled (shuffle-vector (coerce (loop for i below count collect i) 'vector) seed))
         (i (floor (* count train-fraction)))
         (j (+ i (floor (* count valid-fraction)))))
    (values (subseq shuffled 0 i) (subseq shuffled i j) (subseq shuffled j))))

(defun binary-metrics (labels scores &optional (threshold 0.5d0))
  "Compute tie-aware average precision, ROC-AUC and threshold metrics."
  (let* ((n (length labels))
         (positives (count 1 labels))
         (negatives (- n positives))
         (order (stable-sort (coerce (loop for i below n collect i) 'vector)
                             #'> :key (lambda (i) (aref scores i))))
         (tp 0) (fp 0) (area 0d0) (ap 0d0) (loss 0d0))
    (unless (and (= n (length scores)) (plusp positives) (plusp negatives)
                 (every (lambda (x) (member x '(0 1))) labels))
      (error "Binary metrics require equal lengths and both classes"))
    (loop for i below n for p = (aref scores i) for y = (aref labels i)
          do (unless (and (finite-real-p p) (<= 0 p 1))
               (error "Invalid probability ~S" p))
             (let ((q (max 1d-15 (min (- 1d0 1d-15) p))))
               (decf loss (+ (* y (log q)) (* (- 1 y) (log (- 1d0 q)))))))
    (loop for start = 0 then end
          for end = (let ((score (aref scores (aref order start))))
                      (or (position-if (lambda (i) (/= (aref scores i) score))
                                       order :start start)
                          n))
          while (< start n)
          do (let ((old-tp tp) (old-fp fp))
               (loop for j from start below end
                     do (if (= 1 (aref labels (aref order j))) (incf tp) (incf fp)))
               (incf area (* (- fp old-fp) (/ (+ old-tp tp) 2d0)))
               (incf ap (* (/ (- tp old-tp) positives) (/ tp (+ tp fp)))))
          when (= end n) do (return))
    (let ((pred-tp 0) (pred-fp 0))
      (dotimes (i n)
        (when (>= (aref scores i) threshold)
          (if (= (aref labels i) 1) (incf pred-tp) (incf pred-fp))))
      (let* ((precision (if (plusp (+ pred-tp pred-fp))
                            (/ pred-tp (+ pred-tp pred-fp) 1d0) 0d0))
             (recall (/ pred-tp positives 1d0))
             (f1 (if (plusp (+ precision recall))
                     (/ (* 2d0 precision recall) (+ precision recall)) 0d0)))
        (list :rows n :positives positives :prevalence (/ positives n 1d0)
              :average-precision ap :roc-auc (/ area (* positives negatives))
              :log-loss (/ loss n) :threshold threshold :tp pred-tp :fp pred-fp
              :fn (- positives pred-tp) :tn (- negatives pred-fp)
              :precision precision :recall recall :f1 f1)))))

(defun best-threshold (labels scores)
  "Select the smallest score attaining maximal validation F1."
  (unless (= (length labels) (length scores)) (error "Mismatched label and score counts"))
  (let* ((order (stable-sort (coerce (loop for i below (length labels) collect i) 'vector)
                             #'> :key (lambda (i) (aref scores i))))
         (positives (count 1 labels))
         (tp 0) (seen 0) (best-f1 -1d0) (best nil))
    (unless (plusp positives) (error "No positive labels in validation data"))
    (loop for start = 0 then end
          for score = (aref scores (aref order start))
          for end = (or (position-if (lambda (i) (/= (aref scores i) score))
                                     order :start start)
                        (length order))
          do (loop for j from start below end
                   do (incf seen)
                      (incf tp (aref labels (aref order j))))
             (let ((f1 (/ (* 2d0 tp) (+ positives seen))))
               (when (>= f1 best-f1) (setf best-f1 f1 best score)))
          when (= end (length order)) do (return))
    best))

(defun ranking-ndcg (labels scores groups &optional (k 10))
  "Return the mean NDCG@K of contiguous query groups."
  (unless (= (length labels) (length scores) (reduce #'+ groups :initial-value 0))
    (error "Scores, labels, and groups disagree"))
  (when (zerop (length groups)) (error "No ranking groups"))
  (loop with start = 0
        for size across (coerce groups 'vector)
        for end = (+ start size)
        for indices = (coerce (loop for i from start below end collect i) 'vector)
        for predicted = (stable-sort (copy-seq indices) #'> :key (lambda (i) (aref scores i)))
        for ideal = (stable-sort indices #'> :key (lambda (i) (aref labels i)))
        for dcg = (loop for j below (min k size)
                        sum (/ (1- (expt 2d0 (aref labels (aref predicted j))))
                               (log (+ j 2) 2d0)))
        for idcg = (loop for j below (min k size)
                         sum (/ (1- (expt 2d0 (aref labels (aref ideal j))))
                                (log (+ j 2) 2d0)))
        sum (if (plusp idcg) (/ dcg idcg) 0d0) into total
        do (setf start end)
        finally (return (/ total (length groups)))))

(defun read-label-csv (path column)
  (with-open-file (in path)
    (let* ((header (read-csv-row in))
           (index (position column header :test #'string=))
           (values (make-array 0 :adjustable t :fill-pointer 0)))
      (unless index (error "No ~A column in ~A" column path))
      (loop for row = (read-csv-row in) while row
            do (vector-push-extend (parse-number (nth index row)) values))
      values)))

(defun fnv1a (text &optional (seed #xcbf29ce484222325))
  "Hash characters deterministically into a 64-bit FNV-1a value."
  (loop with hash = seed for ch across text
        do (setf hash (ldb (byte 64 0) (* (logxor hash (char-code ch))
                                         #x100000001b3)))
        finally (return hash)))

(defun ensure-example-directories (root)
  (ensure-directories-exist (merge-pathnames "data/.keep" root))
  (ensure-directories-exist (merge-pathnames "output/.keep" root)))
