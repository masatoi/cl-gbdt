# Thread safety

Until this document, cl-gbdt said nothing anywhere about what is safe to do from more than one
thread. A caller could tell a safe operation from one that ends the process only by reading the
implementation. This document writes that contract down, in three tiers -- what is safe, what
is unsafe, and what cl-gbdt does not do about either -- and it adds no synchronisation of its
own: cl-gbdt takes no locks anywhere, before this document or after it, and nothing below
changes that.

## Safe

**Disjoint native lifetimes, from distinct threads, while the backend stays open -- for what
cl-gbdt itself holds.** "Distinct handles" is not the right predicate: two handles can be
distinct Lisp objects and still not be independent, because one's native memory is what the
other reaches into. A `booster` is exactly that case. It holds its training set and every
validation set attached to it, and a call on the booster reaches into their native pointers --
`update-one-iteration` dereferences the training set's pointer on both backends, and
`%check-booster-datasets-live` (see Unsafe, below) exists only because that dependency is real.
A booster and its own training or validation datasets are therefore never disjoint, whatever
their Lisp identities are, and this tier does not cover a thread calling on a booster while
another thread frees a dataset that booster depends on. Put positively: a booster's training
and validation datasets must stay unfreed for as long as any call on that booster is in flight,
on any thread.

Within that constraint, cl-gbdt keeps no shared mutable state on the path from a public call to
the C library: a `dataset` or `booster` is a CLOS object wrapping one foreign pointer, and every
operation on it reads that pointer and calls into the shared library. Nothing is interned,
cached or pooled between calls. Several threads each building their OWN dataset and booster
from the same already-open backend, and reading -- never writing -- the same input arrays, are
safe under the corrected predicate above precisely because each thread's handles are its own:
no thread's booster depends on any dataset another thread might free, so the lifetimes stay
disjoint -- not merely because the handles happen to be distinct objects, which was never
enough by itself. Those threads share only the backend itself, those arrays, and SBCL's global
finalizer registry -- every `make-handle` (`src/handle.lisp`) call writes to it, and SBCL
documents `sb-ext:finalize` itself as thread-safe, so that third piece of shared state is a
guarantee this project is borrowing, not one it built. That is the arrangement
`tests/functional/threads.lisp`'s `concurrent-training-agrees-with-serial-training` exercises,
and `%train-and-predict`'s docstring -- the function each of its threads calls -- says so.

**What this does not establish: that LightGBM or XGBoost is itself safe for two independent
boosters trained at once.** Everything above is a claim about cl-gbdt's own state, not about
either library's internals, and this project has not tested either library's behaviour under
concurrent training with its default OpenMP threading left on -- that the libraries themselves
tolerate concurrent independent use is an assumption, not a measured result. The one test that
trains concurrently, `concurrent-training-agrees-with-serial-training`, first pins both
libraries to a single internal thread -- `*single-threaded-parameters*` sets `:num-threads 1`
for LightGBM and `:nthread 1` for XGBoost -- precisely to take each library's own internal
parallelism out of the picture. A green run of it is therefore silent on the case a caller is
actually likely to run: several Lisp threads training against a library left at its default
internal thread count. Treat that combination as untested, not as covered by this tier.

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
thread's. **Read this file's own header before citing it further: this is a sampling test, not
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

**Freeing a dataset while a call is in flight on a booster that depends on it.** A `booster`
holds its training set and every validation set attached to it -- see the corrected predicate
under Safe, above -- and a call on that booster reaches into their native memory while it
runs. `%check-booster-datasets-live` guards this single-threaded: it reads `handle-released-p`
on the training set and every validation set before the call runs. That is a check-then-act,
not a lock -- it is not a concurrency guard and was never meant to be one, only to turn the
single-threaded case into a clean `released-handle-error` instead of a segfault. How wide the
resulting window is differs by backend. On XGBoost, `update-one-iteration`
(`src/xgboost/api.lisp`) reads the training set's live pointer via `handle-live-pointer` only
after `%check-booster-datasets-live` returns, and hands that pointer to
`%update-one-iteration` as a separate, later step, so a `free-dataset` landing in the gap
between the check and that call is a use-after-free inside XGBoost. On LightGBM
(`src/lightgbm/api.lisp`) there is no such gap to land in: `LGBM_BoosterUpdateOneIter` takes no
dataset argument at all, because the booster already holds the training set's pointer
internally from `LGBM_BoosterCreate` and dereferences that stored pointer itself -- so the
window is not the space between a check and a call, it is the entire duration of the call. A
booster call racing a `free-dataset` on a dataset it depends on ends the process the same way
the two hazards above do.

**Opening a backend concurrently with registering one.** `*backend-classes*` is a plain,
unsynchronised hash table, written by `register-backend` and read by `open-backend`. In
practice `register-backend` runs once, when a backend system is loaded, so this is a hazard
about loading systems from several threads rather than about ordinary use -- named here because
the contract should be complete, not because it is likely to matter.

None of the first three signals a typed `cl-gbdt` condition: all three can end the process
before any handler runs. The fourth leaves `*backend-classes*` in a state nothing here promises
to describe.

## What cl-gbdt does not do

It takes no locks, anywhere. Synchronising against the hazards above -- two threads that might
free the same handle, a close racing a call still in flight, a booster call racing a free of a
dataset it depends on, backend systems loading concurrently -- is entirely the caller's
responsibility. "The library probably handles it" is not an assumption this document allows.

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

The three worst hazards in the "Unsafe" tier above -- freeing one handle from two threads,
closing a backend under a call in flight, and freeing a dataset out from under a booster with a
call in flight on it -- are deliberately not tested at all: all three end the process rather
than failing an assertion, so a test of them would take the whole suite down with it instead of
reporting a result. [File input](file-input.md) takes the same position about XGBoost's
file-format SIGSEGV, for the same reason.

A concurrency test can find a bug; it cannot show the absence of one. Nothing in this document
or in that test file makes any of the "Unsafe" tier above safe, and nothing here should be
cited as though it had.
