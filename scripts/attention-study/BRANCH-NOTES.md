# Study branch provenance

This branch starts at origin/main ddf7526a. The study was developed alongside
PR #523; its tested application/export code is recorded in the evidence.
The separate PR benchmark/documentation commit is 315c1b7e.

`mlpf-onnx-attention.py` is a snapshot of the scoped export wrapper from
PR #523 (source commit 32a49388), retained so the wrapper experiment can run
without merging the application changes into this branch. It is experimental
evidence, not an installed application module.

The article and wrapper experiment use the included snapshot independently
of the application changes. Legacy NGT benchmark helpers require the PR
benchmark commit, as documented in ngt/README.md. The generated archive is
retained locally and ignored by Git; publication sources and curated evidence
are tracked directly.

NGT credentials, downloaded tools, results and the nested NGT documentation
checkout were not copied. Raw ignored onnx_validation evidence remains in
the original particleflow checkout; the portable evidence is under
docs/attention-study/data and in the local archive.
