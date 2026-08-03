# Third-Party Licenses

cl-gbdt itself is MIT licensed; see `LICENSE`. This file records the third-party material
redistributed in this repository, which remains under its own terms — being included here
does not place it under cl-gbdt's licence.

## Vendored C API headers

`tools/fetch-headers.sh` vendors the upstream headers each backend's `c_api.h` reaches, at
the tags recorded in `ffi-spec/VERSIONS`. They are committed so that the bindings can be
regenerated reproducibly. Each file retains its upstream copyright notice.

| Path | Upstream | Licence |
|---|---|---|
| `ffi-spec/lightgbm/include/LightGBM/c_api.h` | [microsoft/LightGBM](https://github.com/microsoft/LightGBM) | MIT |
| `ffi-spec/lightgbm/include/LightGBM/arrow.h` | microsoft/LightGBM | MIT |
| `ffi-spec/lightgbm/include/LightGBM/export.h` | microsoft/LightGBM | MIT |
| `ffi-spec/xgboost/include/xgboost/c_api.h` | [dmlc/xgboost](https://github.com/dmlc/xgboost) | Apache License 2.0 |

LightGBM: Copyright (c) Microsoft Corporation and the LightGBM developers.
XGBoost: Copyright XGBoost Contributors.

## Generated bindings

`src/lightgbm/c-api.lisp` and `src/xgboost/c-api.lisp` are produced mechanically from the
c2ffi specs of the headers above by `tools/regen.lisp`. They contain function signatures and
integer constants, not upstream implementation code, and are distributed under cl-gbdt's MIT
licence along with the rest of the project.

## Shared libraries

cl-gbdt does not bundle either backend's compiled library. `tools/fetch-libs.sh` downloads
them into the git-ignored `vendor/` directory for development and testing only; nothing under
`vendor/` is redistributed here. Both projects are permissively licensed, so linking against
them imposes no additional obligation on cl-gbdt or on its users.
