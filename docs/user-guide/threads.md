# Thread safety

Until this document, cl-gbdt said nothing anywhere about what is safe to do from more than one
thread. A caller could tell a safe operation from one that ends the process only by reading the
implementation. This document writes that contract down, in three tiers -- what is safe, what
is unsafe, and what cl-gbdt does not do about either -- and it adds no synchronisation of its
own: cl-gbdt takes no locks anywhere, before this document or after it, and nothing below
changes that.

## Safe

**Distinct handles, from distinct threads, while the backend stays open.** cl-gbdt keeps no
shared mutable state on the path from a public call to the C library: a `dataset` or `booster`
is a CLOS object wrapping one foreign pointer, and every operation on it reads that pointer and
calls into the shared library. Nothing is interned, cached or pooled between calls. Several
threads each building their own dataset and booster from the same already-open backend, and
reading -- never writing -- the same input arrays, share only the backend itself and those
arrays. That is precisely the arrangement `tests/functional/threads.lisp`'s
`concurrent-training-agrees-with-serial-training` exercises, and its own docstring says so.

**Floating-point trap masking needs no setup from the caller.**
`with-foreign-float-traps-masked` acts on the calling thread alone and restores whatever trap
set was in effect once its body returns or unwinds, and every entry point wraps its whole body
in it -- enforced by `tools/ci/check-float-traps.lisp`. A freshly made thread therefore needs no
setup before calling into cl-gbdt, on either platform, even though SBCL's default
floating-point trap set differs between x86-64 (three traps enabled) and aarch64 (none).

**Error reporting is safe by construction, not by luck.** Both libraries report a failing call
through a last-error buffer that is the calling thread's own: XGBoost's `c_api.h` documents
`XGBGetLastError` outright as "thread safe", and LightGBM's `LGBM_SetLastError` "writes the
*caller's own* thread-local `LastErrorMsg()` buffer" (`ffi-spec/BINDING-COVERAGE.md`). cl-gbdt
reads that buffer synchronously, on the failing call's own thread, the moment the call returns:
`check-foreign-call` (`src/foreign.lisp`) is handed the status code and reads the last-error
message right there, never later and never from a different thread.

`tests/functional/threads.lisp`'s `threads-report-their-own-foreign-errors` exercises that
claim under load -- four threads per backend, each provoking 500 failing calls with a token
unique to that thread, asserting every message it reads back carries its own token and no other
thread's. **Read that test's own header before citing it further: this is a sampling test, not
a proof.** `check-foreign-call` reads the last-error buffer immediately after the failing call
returns, so a process-global buffer would only be *observed* here if another thread's failure
happened to land inside that same sub-millisecond window; 500 rounds buy this test more chances
to catch that in the act, not certainty of catching it. A green run adds confidence on top of
what the two libraries already document and measure about their own buffers -- it is not, by
itself, the reason the claim above is made, and a passing run is not proof the buffers are
per-thread.

## Unsafe, and documentation will not make it safe

**Freeing the same handle from two threads at once.** `release-handle`'s free-once promise is a
check-then-act on a cons cell: both threads can read the cell as not-yet-released before either
one writes to it, and both can go on to call the C free function. **A double free is not a
catchable Lisp condition -- it ends the process outright, skipping every `unwind-protect` on the
way.** This is the single most important sentence in this document.

**Closing a backend while another thread is inside a call on it.** The openness check
`handle-live-pointer` makes and the C call that follows it are not held together. A
`close-backend` landing between them calls `cffi:close-foreign-library` under a call already in
flight on another thread, reaching into a library that may by then be unmapped.

**Opening a backend concurrently with registering one.** `*backend-classes*` is a plain,
unsynchronised hash table, written by `register-backend` and read by `open-backend`. In
practice `register-backend` runs once, when a backend system is loaded, so this is a hazard
about loading systems from several threads rather than about ordinary use -- named here because
the contract should be complete, not because it is likely to matter.

Neither of the first two signals a typed `cl-gbdt` condition: both can end the process before
any handler runs. The third leaves `*backend-classes*` in a state nothing here promises to
describe.

## What cl-gbdt does not do

It takes no locks, anywhere. Synchronising against the hazards above -- two threads that might
free the same handle, a close racing a call still in flight, backend systems loading
concurrently -- is entirely the caller's responsibility. "The library probably handles it" is
not an assumption this document allows.

## `with-backend`

```lisp
(with-backend (backend (open-backend :xgboost))
  (with-dataset (train (make-dataset backend matrix :label labels))
    (with-booster (model (train backend train :num-rounds 10))
      (predict model matrix))))
```

[`with-backend`](../API-REFERENCE.md#cl-gbdt-with-backend) closes its backend on every exit from
its body, normal or not -- the same guarantee `with-dataset` and `with-booster` already give
their own handles. **It must nest outside both of them, never inside.** `close-backend` calls
`cffi:close-foreign-library`, and once it has, `free-dataset` and `free-booster` on that backend
warn and leak rather than calling the C free -- deliberately, because the older behaviour called
into a possibly-unmapped library. A `with-backend` nested inside a handle's macro would close
the library first and leak the handle second.

`with-backend` takes no lock of its own, and using it does nothing about the "Unsafe" tier
above: it guarantees the backend is closed once its own body returns, on the thread running it
-- not that no other thread is still inside a call on that backend at the moment it does.
Coordinating that is, again, the caller's job.

## What a green run of the test file does not mean

`tests/functional/threads.lisp` is not evidence of thread safety in general, and its own header
says so at some length. `threads-report-their-own-foreign-errors` is the one falsifiable claim
in it, and is the sampling test described above under Safe.
`concurrent-training-agrees-with-serial-training` proves nothing about safety by itself: it
samples one interleaving out of a nondeterministic space, and a race that only fires
occasionally can pass it almost every time regardless. Its value is a regression net for gross
breakage and a baseline to compare against if a lock is ever added -- not evidence.

The two worst hazards in the "Unsafe" tier above -- freeing one handle from two threads, and
closing a backend under a call in flight -- are deliberately not tested at all: both end the
process rather than failing an assertion, so a test of them would take the whole suite down
with it instead of reporting a result. [File input](file-input.md) takes the same position
about XGBoost's file-format SIGSEGV, for the same reason.

A concurrency test can find a bug; it cannot show the absence of one. Nothing in this document
or in that test file makes any of the "Unsafe" tier above safe, and nothing here should be
cited as though it had.
