# Plan: ACETestUtils.jl

Shared **test-only** utilities for the ACEsuit packages. First (and currently
only) member: **GPU backend discovery for test suites**. Designed so it can grow
into a general home for cross-package test helpers later.

---

## 1. Why this package exists

A GPU backend-detection helper (`test/utils_gpu.jl`) is currently copy-pasted
**verbatim** across several ACEsuit repos (EquivariantTensors.jl, AtomicOrbitalKernels.jl,
SpheriCart). Ecosystem research (2026-06-19) confirmed **no off-the-shelf package**
provides the full behaviour:

- **MLDataDevices.jl** (`gpu_device()`) only auto-selects among backends whose
  package is **already loaded** (via extensions). It does not probe the system
  without loading, does not install on demand, has no `TEST_BACKEND` override, no
  f64 flag, no test-time graceful degrade. (~2/6 of what we need.)
- **LuxTestUtils.jl** has `has_cuda()`/`has_amdgpu()` system probes and a
  `packages_to_install()` spec list, but does not integrate them into one flow
  and does not install. (~2/6.)

The novel combination is **system-probe-without-loading + install-on-demand +
`TEST_BACKEND` override + f64 flag + graceful degrade-to-CPU**. Hence a small
dedicated package rather than a dependency.

---

## 2. Public API

A single memoized function (replacing the old include-script that set `Main`
globals):

```julia
gpu_test_backend() -> (; dev, gpu, gpu_backend, gpu_supports_f64)
```

- `dev` / `gpu` — host→device transfer **function** (recursive / Adapt-aware via
  the chosen backend); `identity` on CPU. (`gpu` is an alias of `dev`.)
- `gpu_backend::String` — `"CUDA"`, `"AMDGPU"`, `"Metal"`, `"oneAPI"`, or `"CPU"`.
- `gpu_supports_f64::Bool` — `false` on F32-only backends (Metal); tests should
  run F32 always and F64 only when this is `true`.

Memoize the result in a package-internal `Ref`/`const` so repeated calls don't
re-probe or re-`Pkg.add` (this is what the old `Main` include-guard did).

**Consumer usage** (in their `test/runtests.jl`):
```julia
using ACETestUtils: gpu_test_backend
(; dev, gpu_backend, gpu_supports_f64) = gpu_test_backend()
# then: Ag = dev(A); @testset "... ($gpu_backend)" ...
# residency assertion stays backend-agnostic:  @test dev === identity || !(Out isa Array)
```

---

## 3. Detection logic (port the proven `restructure` version)

The canonical reference is EquivariantTensors.jl on the **`restructure`** branch:
`test/test_utils/utils_gpu.jl`. Port its logic verbatim into the function body
(the only change is: function + return-NamedTuple + memoization instead of
top-level `global`s and the `if !isdefined(Main, ...)` guard).

Reference implementation to port:

```julia
using MLDataDevices
import Pkg

function detect_gpu_backend()
    haskey(ENV, "TEST_BACKEND") && return ENV["TEST_BACKEND"]   # manual override
    if Sys.isapple() && Sys.ARCH == :aarch64
        return "Metal"
    elseif !isnothing(Sys.which("nvidia-smi")) && success(`nvidia-smi`)
        return "CUDA"
    elseif !isnothing(Sys.which("rocm-smi")) || isdir("/dev/kfd")
        return "AMDGPU"
    elseif !isnothing(Sys.which("sycl-ls"))   # crude oneAPI probe
        return "oneAPI"
    else
        return "CPU"
    end
end

# then, once per session:
gpu_backend = detect_gpu_backend()
gpu = dev = identity
gpu_supports_f64 = true
if gpu_backend != "CPU"
    try
        Pkg.add(gpu_backend)                 # into the sandboxed test env only
        @eval using $(Symbol(gpu_backend))
        if gpu_backend == "CUDA"
            @assert CUDA.functional();   gpu = dev = CUDA.cu
        elseif gpu_backend == "Metal"
            @assert Metal.functional();  gpu = dev = Metal.mtl
            gpu_supports_f64 = false                 # Metal is F32-only
        elseif gpu_backend == "AMDGPU"
            @assert AMDGPU.functional(); gpu = dev = MLDataDevices.gpu_device()
        elseif gpu_backend == "oneAPI"
            @assert oneAPI.functional(); gpu = dev = oneAPI.oneArray
        else
            error("unknown TEST_BACKEND = $(gpu_backend)")
        end
        @info "GPU test backend: $(gpu_backend) (F64 supported: $(gpu_supports_f64))"
    catch e
        @warn "GPU backend '$(gpu_backend)' detected but not usable; using CPU." exception=(e, catch_backtrace())
        gpu_backend = "CPU"; gpu = dev = identity; gpu_supports_f64 = true
    end
end
gpu_backend == "CPU" && @info "GPU test backend: CPU (dev = identity)."
```

Notes:
- `Pkg.add` at test time works because consumers run under `Pkg.test()`'s
  sandboxed env. Keep it inside the memoized function so it runs at most once and
  only when called (never at package load).
- **World-age gotcha (IMPORTANT — do not port `@eval using` verbatim).** The
  reference `utils_gpu.jl` runs at *top level* in `Main`, where
  `@eval using CUDA` and the subsequent `CUDA.cu` / `CUDA.functional()` are
  separate top-level statements, each in the latest world age. Moving that same
  code *into a function* breaks two ways: (1) bare `CUDA.functional()` resolves
  `CUDA` against the package module at compile time, where it was never imported
  → `UndefVarError`; (2) code loaded via `eval` mid-function is invisible to the
  already-running function (world age). The fix used in the implementation:
  load the backend with `Backend = Base.require(Main, Symbol(gpu_backend))`
  (returns the module object — no compile-time binding needed) and call into the
  freshly-loaded code with `Base.invokelatest(Backend.functional)`. The transfer
  fn (`Backend.cu`, etc.) is only stored, not called in-function, so it needs no
  special handling.
- **Sandbox-only caveat.** Because the function calls `Pkg.add`, it must only
  ever be invoked under `Pkg.test()`'s throwaway sandbox. Calling it against a
  real project env (e.g. `julia --project=. -e 'gpu_test_backend()'`) will
  permanently add the GPU backend to that project's `Project.toml`. Note this in
  each consumer migration.

---

## 4. Package layout

```
ACETestUtils.jl/
  Project.toml      # name, fresh UUID, version = "0.1.0-DEV"
  LICENSE           # MIT (match ACEsuit convention)
  README.md
  src/ACETestUtils.jl
  test/runtests.jl
  .github/workflows/CI.yml   # CPU-only matrix (Julia 1.11, 1.12, pre)
```

- **Deps**: `MLDataDevices` (for the AMDGPU transfer fn + Adapt-aware recursion)
  and `Pkg` (stdlib). The GPU backend packages (CUDA/Metal/AMDGPU/oneAPI) are
  **not** declared — they are `Pkg.add`-ed on demand. Add `[compat]` for
  MLDataDevices and `julia = "1.10"` (or 1.11 if `[sources]`-based consumption is
  assumed).
- **Module**: export `gpu_test_backend`. Could also live under a `Testing`
  submodule if more helpers are added later — start flat.

---

## 5. Tests

`test/runtests.jl`: call `gpu_test_backend()` and assert the contract on whatever
backend CI resolves to (CI is CPU-only → `"CPU"`):
```julia
using ACETestUtils, Test
nt = gpu_test_backend()
@test nt.gpu_backend isa String
@test nt.dev === identity || nt.gpu_backend != "CPU"
@test nt.dev([1.0, 2.0]) isa AbstractVector     # transfer fn callable
@test nt.gpu_supports_f64 isa Bool
# idempotent / memoized:
@test gpu_test_backend().gpu_backend == nt.gpu_backend
```
Optionally a `TEST_BACKEND=CPU` path test. Real GPU paths are exercised by the
*consumer* suites on GPU machines.

---

## 6. Registration & rollout

- **Visibility**: public (nothing sensitive); register in **General** so private
  consumers (e.g. AtomicOrbitalKernels, private until 0.1) can depend on it
  normally.
- **Interim** (before registration): consumers reference it via a `[sources]`
  entry (Julia ≥ 1.11) or `Pkg.develop`.
- **Consumer migration** (separate per-repo PRs, in order):
  1. **AtomicOrbitalKernels.jl** — dogfood. Add `ACETestUtils` to
     `[extras]`/`[targets].test`, replace `include("test/utils_gpu.jl")` +
     bare `dev`/`gpu_backend` with `using ACETestUtils: gpu_test_backend`, delete
     `test/utils_gpu.jl`. This also lets `MLDataDevices` drop out of AOK's test
     `[extras]` (it becomes transitive via ACETestUtils).
  2. **EquivariantTensors.jl** (`restructure`) — replace
     `test/test_utils/utils_gpu.jl`.
  3. **SpheriCart** — replace its (older) `test/utils_gpu.jl` variant.

---

## 7. References

- Canonical source to port: `EquivariantTensors.jl` (branch `restructure`)
  `test/test_utils/utils_gpu.jl`.
- Current in-tree copies for comparison:
  `AtomicOrbitalKernels.jl` (branch `workstream-ab-atomic-orbitals`)
  `test/utils_gpu.jl`; `SpheriCart` `test/utils_gpu.jl` (older variant).
- The consuming test files that show usage patterns:
  `AtomicOrbitalKernels.jl` `test/runtests.jl`, `test/test_gpu.jl`,
  `test/orbitals/test_gpu.jl`.
