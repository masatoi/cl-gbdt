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

Design doc section 11 specifies `tools/check-upstream.lisp`, an upstream-drift
checker not yet built on this branch, to compare the vendored headers against the
latest release tag and drive updates here: a function that disappears from the
diff should move from "still present" to "moot," and a newly detected signature
change should be added to "still present." Until that tool exists, updates happen
manually when `tools/fetch-headers.sh` is re-run against a newer tag.
