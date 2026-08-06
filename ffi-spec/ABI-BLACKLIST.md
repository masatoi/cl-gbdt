# ABI Blacklist

Some upstream functions changed meaning between the reference implementations'
vendored headers and the versions cl-gbdt targets, while keeping the same symbol
name and, in some cases, the same argument count. A call through the generated
CFFI bindings to one of these succeeds -- it never surfaces as an undefined-symbol
or link error -- so `cffi:foreign-symbol-pointer` probing (`src/backend.lisp`)
cannot detect the problem. The only defence is to never call them.

This file is the first-class record of that defence: every function cl-gbdt must
not call, why, and what to call instead. It is referenced from
`src/regen/types.lisp` and `src/backend.lisp`.

## Still present in the generated bindings -- must not be called

These functions are emitted into `src/lightgbm/c-api.lisp` or
`src/xgboost/c-api.lisp` (the symbol exists and is callable), but calling them is
unsafe or points at the wrong signature.

| Function | Reason | Replacement |
|---|---|---|
| `LGBM_DatasetCreateFromMats` | Silent ABI break: the vendored reference implementations' header declared `int is_row_major`; current LightGBM declares `int* is_row_major`. Same name, same argument count, different pointee-ness -- a call compiled against the old signature corrupts memory instead of failing. | `LGBM_DatasetCreateFromMat`, or the streaming API (`LGBM_DatasetInitStreaming` + `LGBM_DatasetPushRows` + `LGBM_DatasetMarkFinished`) |
| `XGDMatrixCreateFromDataIter` | Gained a `float missing` argument upstream since the reference implementations were written. A call built against the old arity passes the wrong values to whatever parameter now occupies that slot. | `XGDMatrixCreateFromCallback` |
| `XGDMatrixCreateFromFile` | Already removed from XGBoost's upstream `master` branch as of design doc section 2.1's measurement, one release after the v3.3.0 tag this project currently vendors (`ffi-spec/VERSIONS`) -- so it is still emitted today, but relying on it means relying on something already gone in the version this project will track next. | `XGDMatrixCreateFromURI` |

## Moot -- removed upstream, never emitted

Design doc section 2.1 found that the reference implementation
(`gitlab.common-lisp.net/cungil/xgboost`) calls five functions that upstream
`master` has since removed; `XGDMatrixCreateFromFile` above is one of them. These
four are the rest, and unlike it, they are simply absent from XGBoost v3.3.0
itself (the version `ffi-spec/VERSIONS` pins), so they were never emitted into
`src/xgboost/c-api.lisp` in the first place: `wanted-p` in `src/regen/emit.lisp`
only emits what the vendored header declares, and the vendored header does not
declare these. Listed here so a future contributor who sees them mentioned in the
reference implementation's source does not go looking for them in this project's
bindings.

| Function | Reason | Replacement |
|---|---|---|
| `XGDMatrixCreateFromCSREx` | Removed upstream | `XGDMatrixCreateFromCSR` |
| `XGDMatrixCreateFromCSCEx` | Removed upstream | `XGDMatrixCreateFromCSC` |
| `XGBoosterGetModelRaw` | Removed upstream | `XGBoosterSaveModelToBuffer` |
| `XGDMatrixSetGroup` | Removed upstream | `XGDMatrixSetInfoFromInterface` |

## Maintaining this list

`tools/check-upstream.lisp` is the upstream-drift checker design doc section 5
specifies. It compares each backend's *imported* functions (not the whole header --
see that file's own header for why) between the vendored spec and a release tag,
after normalising away whitespace, asterisk placement, parameter names, and `const`
so only ABI-meaningful differences are reported.

Run it:

```bash
ros run -- --non-interactive --load tools/check-upstream.lisp
```

With no environment variables, it checks both backends against the tags
`ffi-spec/VERSIONS` pins -- the same tags the vendored headers and generated
bindings were produced from -- and should report clean. To check a specific tag
instead (a prospective upgrade, or to confirm a known break), override the
relevant backend's tag:

```bash
CHECK_UPSTREAM_LIGHTGBM_TAG=v5.0.0 ros run -- --non-interactive \
  --load tools/check-upstream.lisp
CHECK_UPSTREAM_XGBOOST_TAG=v1.5.0 ros run -- --non-interactive \
  --load tools/check-upstream.lisp
```

It needs network access to fetch the upstream header at the tag being checked, and
exits non-zero -- reported as `FAIL`, never folded into a clean result -- if that
fetch fails, so a red exit status always means either "could not check" or "found
drift," never "nothing to report." Both are worth investigating: an offline or
otherwise failed run should simply be re-run once network is available, and a
`FAIL: this tool's own parser disagrees with a vendored header` result means the
vendored spec (or this tool's parser) is stale, not upstream, and should be
resolved by re-running `tools/fetch-headers.sh` before trusting the rest of the
report.

What to do with a real drift finding, per function reported:

- **`ABSENT`** (removed upstream, at or before the checked tag): move the entry
  from "still present" above to "moot," or add it fresh if it is new -- and update
  `src/backend.lisp`/the backend's call sites to stop relying on it, the same way
  `XGBoosterGetModelRaw` and its neighbours already were.
- **`CHANGED`** (present but the normalised signature no longer matches): add a row
  to "still present" naming what changed and, if one exists, a safe replacement --
  the same shape `LGBM_DatasetCreateFromMats`' entry already has.

Either way, `tools/ci/check-abi-blacklist.lisp` picks up the updated table on its
next run with no further wiring: it re-parses this file's tables from scratch every
time.
