(in-package #:cl-user)

(unless (and (find-package "CL-GBDT-EXAMPLE-TEST-PROCESS")
             (symbol-value (find-symbol "*CHILD*" "CL-GBDT-EXAMPLE-TEST-PROCESS")))
  (error "Run examples/tests.lisp to execute this test in an isolated process."))

(load (merge-pathnames "common.lisp" *load-truename*))
(in-package #:cl-gbdt-examples)

(defun signals-error-p (thunk)
  "True when THUNK signals an ERROR that is not an ARITHMETIC-ERROR trap."
  (handler-case (progn (funcall thunk) nil)
    (arithmetic-error () nil)
    (error () t)))

(defun test-parse-number ()
  ;; Plain integer syntax stays an integer; MSLR relevance labels depend on it.
  (assert (eql (parse-number "123") 123))
  (assert (eql (parse-number "+7") 7))
  (assert (eql (parse-number "-7") -7))
  (assert (eql (parse-number "007") 7))
  (assert (eql (parse-number "123456789012345678901234567890")
               123456789012345678901234567890))
  ;; Everything else is the double-float nearest the exact decimal value.
  (assert (eql (parse-number "0.1") 0.1d0))
  (assert (eql (parse-number "-2.5e3") -2500d0))
  (assert (eql (parse-number "1.5D2") 150d0))
  (assert (eql (parse-number "1.5E-2") 0.015d0))
  (assert (eql (parse-number "2.5e+1") 25d0))
  (assert (eql (parse-number "1.") 1d0))
  (assert (eql (parse-number ".5") 0.5d0))
  (assert (eql (parse-number "0.200000000000000") 0.2d0))
  (assert (eql (parse-number "1.7976931348623157e308") most-positive-double-float))
  (assert (eql (parse-number "4.9406564584124654e-324") least-positive-double-float))
  (assert (zerop (parse-number "-0.0")))
  ;; The caller's reader settings must not change the value.
  (let ((*read-base* 16) (*read-default-float-format* 'single-float))
    (assert (eql (parse-number "10") 10))
    (assert (eql (parse-number "1.5") 1.5d0)))
  ;; Blanks around a number, as left by READ-LINE on CRLF files, are accepted.
  (assert (eql (parse-number (format nil " 12~C~C" #\Tab #\Return)) 12))
  ;; Nothing outside the decimal grammar is a number.
  (dolist (text (list "" " " "+" "-" "." "+." "e5" ".e5" "1e" "1e+" "1.5x" "1 2" "1,5"
                      "nan" "NaN" "inf" "-inf" "infinity" "1/3" "#x1F" "#+sbcl 2"
                      "#.(error \"unsafe\")" "0x10" "1_0" "1.5f0" "1.5s0" "1.5l0"
                      "١٢" (format nil "1~%2") "--1" "+-1" "1e5.5"))
    (assert (signals-error-p (lambda () (parse-number text))) () "Accepted ~S" text))
  ;; Values a double-float cannot hold are a clear error, not a float trap.
  (dolist (text '("1e309" "-1e309" "1.8e308" "1e-400" "1e999999999" "1e-999999999"))
    (assert (signals-error-p (lambda () (parse-number text))) () "Accepted ~S" text))
  ;; Rejecting a token must not intern it.
  (assert (signals-error-p (lambda () (parse-number "zqxparseprobe"))))
  (assert (null (find-symbol "ZQXPARSEPROBE" *package*)))
  (assert (null (find-symbol "ZQXPARSEPROBE" "CL-GBDT-EXAMPLES"))))

(defun run-example-tests ()
  (let* ((labels #(0 1 0 1))
         (scores #(0.5d0 0.5d0 0.25d0 0.75d0))
         (metrics (binary-metrics labels scores 0.5d0)))
    (assert (< (abs (- (getf metrics :roc-auc) 0.875d0)) 1d-12))
    (assert (< (abs (- (getf metrics :average-precision) (/ 5d0 6d0))) 1d-12))
    (assert (= (getf metrics :tp) 2))
    (assert (= (getf metrics :fp) 1))
    (assert (= (best-threshold labels scores) 0.5d0))
    (assert (= (ranking-ndcg #(3 2 0) #(3d0 2d0 0d0) '(3)) 1d0))
    (assert (= (ranking-ndcg #(0 0) #(1d0 2d0) '(2)) 0d0))
    (dolist (groups '((0 3) (-1 4) (1.5 1.5) ()))
      (assert (handler-case
                  (progn (ranking-ndcg #(2f0 1f0 0f0) #(3d0 2d0 1d0) groups) nil)
                (error () t))))
    (dolist (cutoff '(0 -1 0.5))
      (assert (handler-case
                  (progn (ranking-ndcg #(2f0 1f0 0f0) #(3d0 2d0 1d0) '(3) cutoff) nil)
                (error () t))))
    (assert (handler-case
                (progn (ranking-ndcg #(1f0 -1f0) #(2d0 1d0) '(2)) nil)
              (error () t)))
    (dolist (bad (list (sb-kernel:make-single-float #x7fc00000)
                       sb-ext:single-float-positive-infinity))
      (assert (handler-case
                  (progn (ranking-ndcg (vector 1f0 bad) #(2d0 1d0) '(2)) nil)
                (error () t)))
      (assert (handler-case
                  (progn (ranking-ndcg #(1f0 0f0) (vector 2d0 bad) '(2)) nil)
                (error () t)))))
  (multiple-value-bind (train valid test) (split-indexes 100 42)
    (assert (= (length train) 60))
    (assert (= (length valid) 20))
    (assert (= (length test) 20))
    (assert (= (length (remove-duplicates
                        (concatenate 'list train valid test))) 100))
    (multiple-value-bind (again) (split-indexes 100 42)
      (assert (equalp train again))))
  (with-input-from-string (stream (format nil "a,~Cb,~%c~C,~Cd~C~Ce~C~%1,2,3~%"
                                          #\" #\" #\" #\" #\" #\"))
    (assert (equal (read-csv-row stream)
                   (list "a" (format nil "b,~%c") "d\"e")))
    (assert (equal (read-csv-row stream) '("1" "2" "3")))
    (assert (null (read-csv-row stream))))
  (test-parse-number)
  ;; Every exported function common.lisp defines is documented; unused ones are gone.
  (do-external-symbols (symbol "CL-GBDT-EXAMPLES")
    (when (fboundp symbol)
      (assert (documentation symbol 'function) () "~S has no docstring" symbol)))
  (assert (null (find-symbol "READ-LABEL-CSV" "CL-GBDT-EXAMPLES")))
  (uiop:with-temporary-file (:pathname path :prefix "cl-gbdt-f32-"
                             :element-type '(unsigned-byte 8))
    (let ((wanted #(1.5f0 -2.25f0 0f0)))
      (with-open-file (out path :direction :output :if-exists :supersede
                                :element-type '(unsigned-byte 8))
        (loop for value across wanted do (write-f32 out value)))
      (assert (equalp wanted (read-f32 path)))
      (with-open-file (out path :direction :output :if-exists :supersede
                                :element-type '(unsigned-byte 8))
        (write-byte 0 out))
      (assert (handler-case (progn (read-f32 path) nil) (error () t)))))
  (format t "Example utility tests passed.~%"))

(run-example-tests)
