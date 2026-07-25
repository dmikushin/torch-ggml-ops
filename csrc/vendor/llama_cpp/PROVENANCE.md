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
