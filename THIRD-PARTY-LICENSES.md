# Third-Party Licenses

cl-gbdt itself is MIT licensed; see `LICENSE`. This file records the third-party material
redistributed in this repository, which remains under its own terms — being included here
does not place it under cl-gbdt's licence.

## Vendored C API headers

`tools/fetch-headers.sh` vendors the upstream headers each backend's `c_api.h` reaches, at
the tags recorded in `ffi-spec/VERSIONS`. They are committed so that the bindings can be
regenerated reproducibly. Each file retains its upstream copyright notice.

| Path | Upstream | Licence | Full text |
|---|---|---|---|
| `ffi-spec/lightgbm/include/LightGBM/c_api.h` | microsoft/LightGBM | MIT | `LICENSES/LightGBM-MIT.txt` |
| `ffi-spec/lightgbm/include/LightGBM/arrow.h` | microsoft/LightGBM | MIT | `LICENSES/LightGBM-MIT.txt` |
| `ffi-spec/lightgbm/include/LightGBM/export.h` | microsoft/LightGBM | MIT | `LICENSES/LightGBM-MIT.txt` |
| `ffi-spec/xgboost/include/xgboost/c_api.h` | dmlc/xgboost | Apache 2.0 | `LICENSES/XGBoost-Apache-2.0.txt` |

LightGBM: Copyright (c) Microsoft Corporation and the LightGBM developers.
XGBoost: Copyright XGBoost Contributors.

**Both licence texts are included in this repository, not merely referenced.** Naming a
licence and linking upstream is not enough: Apache 2.0 section 4(a) requires that recipients
of the work receive a copy of the licence, and MIT requires its permission notice to travel
with substantial portions of the software. The header files carry their copyright lines but
point at a `LICENSE` in their own project root, which is not distributed here — so the texts
are vendored verbatim under `LICENSES/` at the same tags recorded in `ffi-spec/VERSIONS`.

XGBoost has no `NOTICE` file at v3.4.1, so Apache 2.0 section 4(d) adds no further
attribution requirement.

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
