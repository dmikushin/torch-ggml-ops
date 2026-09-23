# llama.cpp provenance

Upstream revisions are recorded outside compiled headers so provenance-only changes do not invalidate MMQ translation units.

## Generated files

The following section is maintained by `tools/generate_vendor.py`.

<!-- generate_vendor:begin -->
Canonical repository: <https://github.com/ggml-org/llama.cpp>

Closest ancestor on `master`: `0cea36222fe9bac5ebfc45716c9eef11f37046c4`

Generated files:

- `iq2_s_grid.cuh`
- `iq2_xxs_grid.cuh`
- `mma.cuh`
- `mmq-load-targets.cuh`
- `mmq-vec-dot-q2-k-rolled.cuh`
- `mmq-vec-dot-targets.cuh`
<!-- generate_vendor:end -->

## Compatibility surface

`common.cuh` was manually derived from `ggml-common.h`, `ggml-cuda/common.cuh`, and `ggml-cuda/vendors/hip.h`. The source revision's closest ancestor on canonical `master` is `00fa7cb284cbf133fc426733bd64238a3588a33e`.

## IQ3_S grid

`iq3_s_grid.cuh` was produced with the extraction that `tools/generate_vendor.py`
now performs, run by hand against a llama.cpp checkout at
`60130d18f9ac7f42cb4d7f6060b088a45d8f242e` (not a canonical `master` tracking
ref, so the generator itself could not be run). The 512 entries were checked to
be identical to the independently encoded `IQ3_S` grid in `gguf-py` 0.19.0. The
next full `generate_vendor.py` run regenerates this file with the others.
