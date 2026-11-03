# KASANDR: sparse click prediction

Analyze a reproducible sample of [UCI KASANDR](https://archive.ics.uci.edu/dataset/385/kasandr),
an e-commerce advertisement dataset, with `cl-gbdt:make-csr-matrix`. UCI distributes
the data under CC BY 4.0; credit its creators when using it. No data or generated
matrices are committed here.

After the [common setup](../README.md#setup), run:

```lisp
(cl-gbdt-examples:fetch-example-data :kasandr-sparse)
(load "examples/kasandr-sparse/prepare.lisp")
(load "examples/kasandr-sparse/train.lisp")
(load "examples/kasandr-sparse/evaluate.lisp")
```

Allow at least 12 GB of free disk for the ZIP, compressed TAR, temporary uncompressed
TAR, extracted inputs and prepared data. The source ZIP SHA-256 was
`1851d21603d24c4baa060d4e73ce98a2393effe3d0af68b59e63fb0bd86db509`.
Decompression uses `semz.decompress` and may take several minutes. Preparation scans
both complete raw files and expects 15,844,717 training and 1,919,561 test rows,
excluding headers. This check rejects truncated files. An earlier run accidentally
used a partial test file of 638,478 rows; its results are not reported here. If a
cached source file is incomplete, remove it before fetching again.

The default sample contains 500,000 training-file rows and 100,000 test-file rows,
selected without replacement by reservoir sampling with a deterministic xorshift32
generator. Set `CL_GBDT_KASANDR_TRAIN_SAMPLE` and `CL_GBDT_KASANDR_TEST_SAMPLE`
to smaller sizes for a quicker experiment (at least 5 train and 1 test row); both source files are
still scanned in full. The original files are not globally sorted: each sample is
sorted by timestamp, the source sample is split 80/20 without splitting identical
timestamps, and the test sample is from the separately provided later period.

Each of four ID fields is mapped by deterministic FNV-1a into a separate set of
2,048 buckets. Each row has four explicitly stored ones and 8,192 possible columns.
Hash collisions lose information. The new Lisp sampler and hash differ from the
previous Python implementation and therefore produce a different sample and scores.

Absent CSR entries require care. With default LightGBM settings, an absent entry
is zero and is declared `:implicit-value 0d0`. XGBoost treats absence as missing:
the example checks that it refuses a matrix declaring zero, then intentionally
trains a separate `:implicit-value :missing` representation. **The two experiments
do not use the same missing-value semantics** and are not a controlled backend
comparison. See the [sparse-input guide](../../docs/user-guide/data-and-prediction.md#sparse-input-csr-matrices).

## Recorded run on the complete source files

| Split | Sampled rows | Click rate |
|---|---:|---:|
| Training | 400,000 | 0.0354 |
| Validation | 100,000 | 0.0803 |
| Test | 100,000 | 0.3196 |

| Test metric | LightGBM | XGBoost |
|---|---:|---:|
| Average precision | 0.46946 | 0.46646 |
| ROC-AUC | 0.66254 | 0.66672 |
| Log loss | 1.03208 | 1.03977 |

Parameters and hash-space size are illustrative, not tuned. The original data show
a large shift in click rates between periods. Probabilities are not calibrated;
the results should not be treated as a production click-rate or revenue estimate.
