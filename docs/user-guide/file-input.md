# File input

How `create-dataset-from-file` builds a dataset directly from a file, on each backend.

`cl-gbdt/lightgbm:create-dataset-from-file` and `cl-gbdt/xgboost:create-dataset-from-file` build
a dataset by having the library read a file directly, rather than from a matrix the caller
already holds in memory. **There is no unified form of this** -- `cl-gbdt:make-dataset` takes no
pathname, on either backend, and no `:file-input` capability exists to ask about. Both operations
are Layer 1 only. `docs/cl-gbdt-layered-api-implementation-policy.md` section 12's "Why file
input was not put on the unified API" carries the measurement that decided that -- the short
version is that XGBoost's file-format argument, declared wrong on a unified API, could end the
process outright, in a thread no Lisp handler can reach.

The two signatures differ because Layer 1 mirrors each library rather than harmonising them, the
same rule every other backend-specific difference in [Backend
differences](backend-differences.md) follows:

```lisp
(create-dataset-from-file backend path &key parameters reference)   ; cl-gbdt/lightgbm
(create-dataset-from-file backend path format &key uri-parameters)  ; cl-gbdt/xgboost
```

**LightGBM** takes no format argument at all: `LGBM_DatasetCreateFromFile` infers CSV, TSV or
libsvm from the file's own content -- and reads LightGBM's own binary dataset through the same
call, with no parameter announcing it either. `PARAMETERS` is a plist in LightGBM's own
vocabulary, rendered exactly as `create-dataset`'s `:parameters` already is and handed to the
file reader verbatim; `header=true` consumes the file's first line as a header rather than a data
row, and `label_column=N`/`label=N` (aliases of each other) pick a 0-based column as the label,
defaulting to 0. `REFERENCE` is another `lightgbm-dataset` whose bin mapper this one should align
to, or `NIL` to build its own -- the same meaning `create-dataset`'s own `:reference` already
has.

**XGBoost** takes `FORMAT` as a required, positional third argument -- one of `:libsvm`, `:csv`
or `:binary` -- because `XGDMatrixCreateFromURI`'s `format` query parameter is mandatory for any
text file, and a default would invite a caller not to think about the case below that matters.
Measured against the vendored 3.3.0: **XGBoost does not check `FORMAT` against the file's own
contents before it starts parsing, and getting it wrong is not merely wrong.** A real CSV file
declared `:libsvm`, or a binary DMatrix declared `:libsvm`, both **segfault** inside a thread
dmlc creates for the parse -- outside any Lisp stack, so no `handler-case` anywhere can catch it.
The reverse direction does not crash, but is silently wrong instead: a real libsvm file declared
`:csv` returns a 4-row, 1-column dataset with no label and a success code. Measurement went on to
find that dmlc's own URI syntax is richer than a short list of reserved characters can enumerate
ahead of it: a directory is a file list to dmlc, not an error; `;` splits one URI into several
paths, each read; a glob character expands. `create-dataset-from-file` closes all of it the same
way rather than adding one more guard per shape found: `PATH` is resolved to a single `truename`
exactly once, and that one resolution -- never the caller's own designator a second time -- is
both what gets classified and what the URI is composed from, so the file dmlc opens is provably
the file this wrapper looked at, not merely probably. Every verdict but an exact match with the
declared `FORMAT` is refused with `file-format-mismatch` -- naming the path, the declared format
and the detected one -- including a `PATH` this wrapper could not resolve to one existing file at
all (missing, wild, or a symlink to nowhere), or that resolved to a directory: none of that is
XGBoost's own to report any longer, since dmlc's response to several of those shapes turned out
not to be an error either.

**The classification is of `PATH`'s first non-blank line, not of `PATH`.** A file whose first
line is genuinely libsvm-shaped but whose later rows are not is still classified `:LIBSVM` and
still passes this gate; the mismatched later content reaches `XGDMatrixCreateFromURI` regardless.
Measured (PR #36 re-review, ten runs against five mixed-content shapes and both libraries; the
full record is `cl-gbdt/xgboost:create-dataset-from-file`'s own docstring, reproduced in
[`docs/API-REFERENCE.md`](../API-REFERENCE.md)):
none of the ten crashed, but that is what ten runs showed and not a guarantee about every input.
**A file truncated mid-row is silently accepted by both libraries as though it were complete --
no error, a dataset built from whatever whole rows came before the cut.** A caller who hands
either `create-dataset-from-file` a file another process is still writing can get a dataset back
with no indication anything was missing. A corrupted tail is also read silently by both, but not
identically: the same file gave XGBoost 3 rows and LightGBM 4 for the fixture measured. Reading
the rest of the file to catch either case would mean this wrapper reads what it exists to let the
library read once, so neither library's wrapper does it.

**A FIFO or other blocking special file is not detected, and `PATH` is expected to name a data
file.** ANSI Common Lisp has no portable way to ask whether a resolved path names a named pipe or
a device rather than an ordinary regular file, so nothing above catches one -- unlike an earlier
version, which used SBCL's `sb-posix:stat` to check this too and was reverted because the project
owner declined to add a further SBCL-specific dependency for this one check, not because reverting
it made this backend any more portable: `file-uri`'s own `sb-ext:native-namestring` call and
`src/xgboost/native.lisp`'s array pinning with `sb-sys` primitives (below) already made it
SBCL-only before this decision, and still do after it. The trade bought back no portability at
all, only the loss of a check that would have caught a FIFO or a device file. A FIFO with nothing
on the other end of it blocks indefinitely inside the read that classifies it, with no error and
no diagnostic; an unbounded device such as `/dev/zero` cannot hang this wrapper forever (that same
read is capped), but reads as whatever bytes it produces rather than being refused outright.
`URI-PARAMETERS` is a plist of
further dmlc query keys,
`(:label_column 0)` among them, appended to the URI after `FORMAT`'s own `format=` key; a
`format` key inside `URI-PARAMETERS` itself signals `unsupported-argument`, since `FORMAT` is
this function's own argument to give, not a second, unchecked route to the same key. `:binary`
carries no `format=` key in the URI at all -- there is no such spelling; the way to load an
XGBoost binary DMatrix is a URI with no `format=` key whatsoever, which `create-dataset-from-file`
already knows.

**A libsvm RANKING file is accepted on XGBoost and refused outright on LightGBM.** Measured
(PR #36 review): a row carrying a `qid:<group>` tag between the label and its feature pairs
(`1 qid:1 1:0.5 2:0.3`) reads cleanly through `cl-gbdt/xgboost:create-dataset-from-file`
declared `:libsvm` -- the same shape as the identical rows with `qid` removed, group
boundaries correctly recovered -- but `cl-gbdt/lightgbm:create-dataset-from-file` on the
identical file signals `foreign-call-error` with LightGBM's own `"Input format error when
parsing as LibSVM"`, a limitation of that library's own parser rather than of either wrapper.

**XGBoost's text-file path is deprecated upstream.** Every text-file attempt, including one this
wrapper's own gate refuses, prints once per process, to stderr: `WARNING: .../data.cc:963: Text
file input has been deprecated since 3.1`. It is published anyway, because
`ffi-spec/BINDING-COVERAGE.md` already named `XGDMatrixCreateFromURI` as `XGDMatrixCreateFromFile`'s
own replacement -- wrapping it is what makes that recommendation reachable rather than a dead
pointer, not a bet against the deprecation.

Neither function takes a `LABEL`, `WEIGHT`, `GROUP` or `FEATURE-NAMES` argument at all: the file
already carries whatever `create-dataset`'s caller would otherwise pass separately, and by the
time the foreign call returns a finished handle there is nothing left in Lisp to attach one to.
