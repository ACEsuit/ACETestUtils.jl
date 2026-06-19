# ACETestUtils.jl

[![Build Status](https://github.com/ACEsuit/ACETestUtils.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ACEsuit/ACETestUtils.jl/actions/workflows/CI.yml?query=branch%3Amain)

Shared **test-only** utilities for the [ACEsuit](https://github.com/ACEsuit)
packages.

This package collects helpers that several ACEsuit test suites would otherwise
copy-paste between repos. It is meant to be used from the `test/` environment of
a consuming package (via `[extras]` + `[targets].test`), **not** as a runtime
dependency.

Current contents:

- [`gpu_test_backend()`](#gpu-backend-discovery) — discover and set up the GPU
  backend for a test suite.

## Installation

The package is intended to live in a consumer's test environment. Add it to the
`test` target of the consuming package's `Project.toml`:

```toml
[extras]
ACETestUtils = "b973b150-f408-4aa5-b6a2-f0e33df46af3"
Test         = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["ACETestUtils", "Test"]
```

Until the package is registered in the General registry, reference it via a
[`[sources]`](https://pkgdocs.julialang.org/v1/toml-files/#The-[sources]-section)
entry (Julia ≥ 1.11) or `Pkg.develop`.

## GPU backend discovery

`gpu_test_backend()` discovers the GPU backend available for a test suite and
returns the tools needed to write device-agnostic tests. It

- **probes the system** for a GPU *without loading* any GPU package (so a plain
  CI runner stays on CPU and pulls in no heavy GPU dependency),
- **installs the matching backend on demand** into the (sandboxed) test
  environment when a GPU is found,
- **degrades gracefully to CPU** with a warning if a detected backend turns out
  not to be usable, and
- **memoizes** its result, so the probe and install happen at most once per
  session.

```julia
gpu_test_backend() -> (; dev, gpu, gpu_backend, gpu_supports_f64)
```

| field | type | meaning |
|-------|------|---------|
| `dev` | function | host→device transfer function; `identity` on CPU |
| `gpu` | function | alias of `dev` |
| `gpu_backend` | `String` | `"CUDA"`, `"AMDGPU"`, `"Metal"`, `"oneAPI"`, or `"CPU"` |
| `gpu_supports_f64` | `Bool` | `false` on F32-only backends (Metal) |

Set the `TEST_BACKEND` environment variable (`"CPU"`, `"CUDA"`, `"AMDGPU"`,
`"Metal"`, `"oneAPI"`) to force a choice and bypass the system probe — useful for
exercising the CPU path on a GPU machine, or vice versa.

### Detection logic

The backend is chosen by probing the host (first match wins):

1. `TEST_BACKEND` environment variable, if set;
2. Apple silicon (`Sys.isapple() && Sys.ARCH == :aarch64`) → `Metal`;
3. a working `nvidia-smi` → `CUDA`;
4. `rocm-smi` present or `/dev/kfd` exists → `AMDGPU`;
5. `sycl-ls` present → `oneAPI`;
6. otherwise → `CPU`.

### Usage

In a consumer's `test/runtests.jl`:

```julia
using ACETestUtils: gpu_test_backend
(; dev, gpu_backend, gpu_supports_f64) = gpu_test_backend()

Ag = dev(A)
@testset "my kernel ($gpu_backend)" begin
   Out = my_kernel(Ag)
   # residency assertion stays backend-agnostic:
   @test dev === identity || !(Out isa Array)

   # run F64 only where the backend supports it:
   if gpu_supports_f64
      @test my_kernel(dev(Float64.(A))) ≈ reference
   end
end
```

### Caveats

- `gpu_test_backend()` calls `Pkg.add` for the detected backend. This is safe
  **only** under `Pkg.test()`, whose sandboxed environment is throwaway. Do not
  call it against a package's real project environment, or it will permanently
  add the GPU backend as a dependency there.
- The on-demand backend is loaded with `Base.require` and called via
  `Base.invokelatest`, so the function is safe to call from inside a module (it
  does not rely on top-level `using` / world-age behaviour).

## Contributing

Contributions are welcome. This package is deliberately small; please open an
issue to discuss new shared test helpers before adding them, so the scope stays
focused on utilities that are genuinely reused across ACEsuit test suites.
