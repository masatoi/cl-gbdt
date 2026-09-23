;;;; checks.lisp --- Run every executable specification at a fixed seed.
;;;;
;;;; The bundle in specs/ declares contracts and Properties; nothing runs them until something
;;;; asks. This file is that something for CI: every Function Spec `cl-gbdt/specs/all' lists is
;;;; checked with `cl-spec:check-function', and every Property with `cl-spec:run-property',
;;;; both at +SEED+, so a red run replays exactly.
;;;;
;;;; A pass here means "not falsified over the trials that were generated" -- not a proof. It
;;;; is also stricter than `status :passed': a contract that rejected any generated input, or
;;;; whose `:cases' left a case uncalled, fails too, because a pass that never reached a branch
;;;; says nothing about that branch.
;;;;
;;;; Needs no shared library: `tools/ci/run-tests.lisp' fails this system if one is opened.

(uiop:define-package #:cl-gbdt/tests/specs/checks
  (:use #:cl #:rove)
  ;; Bare: the bundle plus the check-it backend, which is what makes generation possible.
  (:import-from #:cl-gbdt/specs/check-it)
  (:import-from #:cl-spec/main)
  (:import-from #:cl-gbdt/specs/all
                #:contract-names
                #:property-names))

(in-package #:cl-gbdt/tests/specs/checks)

(defparameter +seed+ 42
  "The seed every check here runs at. Fixed so that a failure in CI replays locally with
`spec-check' and the same seed.")

(defparameter +contract-trials+ 200
  "Trials per Function Spec. Properties take their count from their own `:normal' profile.")

(defparameter +property-trials+ 200
  "The `:normal' budget every Property in the bundle declares.")

(defun home-package-prefix-p (symbol prefix)
  "True when SYMBOL's home package name starts with PREFIX."
  (let ((package (symbol-package symbol)))
    (and package
         (let ((name (package-name package)))
           (and (>= (length name) (length prefix))
                (string= prefix name :end2 (length prefix)))))))

(defun contract-verdict (name)
  "Run NAME's Function Spec at +SEED+ and return a plist of what CI judges it by.

`cl-spec:result-data' returns a plist even for a contract with no `:cases' clause, its
:CASES and :DECLARED-CASES both NIL then -- confirmed at seed 42 for a case-less contract
(`cl-gbdt/src/training-report:make-training-series') and a `:cases' one
(`cl-gbdt/src/config/prediction-shape:contrib-shape') before this was written. :CASE-REPORT,
:DECLARED-CASES and :CASE-CALLS (a (name . called) alist built straight from :CASE-REPORT's
own :CASES entries) let a caller re-derive the never-called check from the declared case
names and each case's own :CALLED count, rather than trusting :CASE-REPORT's :NEVER-CALLED
key alone -- a renamed or dropped :CASE-REPORT key would otherwise leave CASE-REPORT NIL,
:NEVER-CALLED vacuously NIL too, and the case check passing on nothing."
  (let* ((result (cl-spec:check-function name :trials +contract-trials+ :seed +seed+))
         (case-report (getf (cl-spec:result-data result) :case-report)))
    (list :status (cl-spec:property-result-status result)
          :rejected (cl-spec:function-check-result-rejected result)
          :never-called (and (consp case-report) (getf case-report :never-called))
          :case-report case-report
          :declared-cases (and (consp case-report) (getf case-report :declared-cases))
          :case-calls (and (consp case-report)
                            (mapcar (lambda (case)
                                      (cons (getf case :name) (getf case :called)))
                                    (getf case-report :cases))))))

(deftest every-registered-definition-is-listed
  (testing "the registry holds nothing from this project that the lists leave out"
    ;; The lists are what the two tests below run. A definition added to specs/ and not to
    ;; them would load, register, and never be checked -- a green run of one check fewer.
    (let ((contracts (remove-if-not (lambda (name) (home-package-prefix-p name "CL-GBDT/"))
                                    (cl-spec:list-function-specs)))
          (properties (remove-if-not (lambda (name)
                                       (home-package-prefix-p name "CL-GBDT/SPECS/"))
                                     (cl-spec:list-properties))))
      (ok (null (set-difference contracts (contract-names)))
          (format nil "unlisted contracts: ~S" (set-difference contracts (contract-names))))
      (ok (null (set-difference properties (property-names)))
          (format nil "unlisted properties: ~S"
                  (set-difference properties (property-names))))
      (ok (null (set-difference (contract-names) contracts))
          (format nil "listed but unregistered contracts: ~S"
                  (set-difference (contract-names) contracts)))
      (ok (null (set-difference (property-names) properties))
          (format nil "listed but unregistered properties: ~S"
                  (set-difference (property-names) properties)))
      (ok (plusp (length (contract-names))) "the bundle lists at least one contract")
      (ok (plusp (length (property-names))) "the bundle lists at least one property"))))

(deftest every-contract-holds-at-the-fixed-seed
  (let ((verdicts nil))
    (dolist (name (contract-names))
      (testing (format nil "~S" name)
        (let ((verdict (contract-verdict name)))
          (push (cons name verdict) verdicts)
          (ok (eq :passed (getf verdict :status))
              (format nil "~S at seed ~D: ~S" name +seed+ (getf verdict :status)))
          (ok (eql 0 (getf verdict :rejected))
              (format nil "~S rejected ~S generated inputs" name (getf verdict :rejected)))
          (ok (null (getf verdict :never-called))
              (format nil "~S never called case(s) ~S" name (getf verdict :never-called)))
          ;; Non-vacuous even if a future cl-spec renames or drops :CASE-REPORT: that would
          ;; leave CASE-REPORT NIL and this assertion, not just :NEVER-CALLED above, red.
          (ok (consp (getf verdict :case-report))
              (format nil "~S: no case report (expected a plist, even for a case-less
contract)" name))
          (ok (= (length (getf verdict :declared-cases)) (length (getf verdict :case-calls)))
              (format nil "~S: ~D declared case(s) but ~D case report(s): ~S"
                      name (length (getf verdict :declared-cases))
                      (length (getf verdict :case-calls)) (getf verdict :case-calls)))
          (ok (every (lambda (call) (plusp (cdr call))) (getf verdict :case-calls))
              (format nil "~S: some declared case was called zero times: ~S"
                      name (getf verdict :case-calls))))))
    (testing "at least one listed contract declares cases"
      ;; Otherwise the two case assertions above hold on every contract only because none
      ;; of them ever has a case to check -- true of an empty set, and not what "every
      ;; declared case was called" is meant to prove.
      (ok (some (lambda (entry) (getf (cdr entry) :declared-cases)) verdicts)
          "no contract in the bundle declared any :cases -- the case checks above never ran
on a real case"))))

(deftest every-property-holds-at-the-fixed-seed
  (dolist (name (property-names))
    (testing (format nil "~S" name)
      (let ((result (cl-spec:run-property name :profile :normal :seed +seed+)))
        (ok (eq :passed (cl-spec:property-result-status result))
            (format nil "~S at seed ~D: ~S" name +seed+
                    (cl-spec:property-result-status result)))
        (ok (eql +property-trials+ (cl-spec:property-result-trials result))
            (format nil "~S ran ~S trials" name (cl-spec:property-result-trials result)))))))
