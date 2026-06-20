# AGENTS.md — ACETestUtils.jl

Shared **test-only** utilities for the ACEsuit packages. Use it from a consuming
package's `test/` environment (via `[extras]` + `[targets].test`), **not** as a
runtime dependency.

UUID: `b973b150-f408-4aa5-b6a2-f0e33df46af3`

## Dependency model (read this first)

- Core helpers (`gpu_test_backend`, `fdtest`, `dirfdtest`, `print_tf`,
  `println_slim`) are always available with `using ACETestUtils`.
- Gradient helpers (`grad_zy`, `grad_zy_ps`, `grad_fd_ps`) live in **package
  extensions** and only work once the relevant AD package is also loaded:
  - `grad_zy`, `grad_zy_ps` → also `using Zygote`
  - `grad_fd_ps` → also `using ForwardDiff`
  Calling them without the trigger package loaded throws a `MethodError`.

## What's provided, when to use it, how

### `gpu_test_backend()` — device-agnostic GPU testing

- **When:** a test suite has CPU and GPU code paths and you want it to run on
  whatever hardware is present (CPU on CI, GPU on a GPU box) without per-machine
  edits.
- **How:**
  ```julia
  using ACETestUtils: gpu_test_backend
  (; dev, gpu_backend, gpu_supports_f64) = gpu_test_backend()
  Ag = dev(A)                                  # host→device; identity on CPU
  @testset "kernel ($gpu_backend)" begin
     @test dev === identity || !(out isa Array)   # residency, backend-agnostic
     gpu_supports_f64 && @test kernel(dev(Float64.(A))) ≈ ref  # F64 only if supported
  end
  ```
- Returns `(; dev, gpu, gpu_backend, gpu_supports_f64)`. `gpu` aliases `dev`.
  `gpu_backend::String` ∈ `"CUDA"`/`"AMDGPU"`/`"Metal"`/`"oneAPI"`/`"CPU"`.
- Force a backend with the `TEST_BACKEND` env var (`"CPU"`, `"CUDA"`, …).
- **Gotcha:** it calls `Pkg.add` for the detected backend, so call it **only**
  under `Pkg.test()`'s sandbox — never against a real project env, or it
  permanently adds the GPU package there. Result is memoized (probe/install run
  once per session).

### `fdtest` / `dirfdtest` — finite-difference gradient checks

- **When:** verify that a hand-written gradient `dF` is consistent with `F`.
- **How:** returns `true`/`false`.
  ```julia
  using ACETestUtils: fdtest, dirfdtest
  @test fdtest(F, dF, x)              # x::AbstractVector, ::Number, or ::Vector{<:SVector{3}}
  @test dirfdtest(F, dF, x, u)        # directional variant along u
  ```
  Pass `verbose=false` to silence the per-step table. Ported from
  `ACEbase.Testing` (behaviour unchanged).

### `print_tf` / `println_slim` — compact `@test` result display

- **When:** loops with many `@test`s where one line per result is too noisy.
- **How:**
  ```julia
  using ACETestUtils: print_tf, println_slim
  for case in cases
     print_tf(@test check(case))     # prints "+" / "-" / "x"
  end
  println()
  ```

### `grad_zy` / `grad_zy_ps` / `grad_fd_ps` — Lux-model gradient helpers (extensions)

- **When:** testing a Lux-style `model(X, ps, st)` (returns `(y, st)`); get a
  reference gradient w.r.t. input or parameters to compare against your own.
- **How:** load the AD package alongside ACETestUtils.
  ```julia
  using ACETestUtils, Zygote, ForwardDiff
  grad_zy(X, model, ps, st)      # Zygote ∂/∂X of model(·, ps, st)[1]
  grad_zy_ps(X, model, ps, st)   # Zygote ∂/∂ps of model(X, ·, st)[1]
  grad_fd_ps(G, model, ps, st)   # ForwardDiff ∂/∂ps via Optimisers.destructure
  ```
- **Gotcha:** `grad_fd_ps` relies on `Optimisers.destructure`, which only
  flattens **array** parameter leaves — scalar parameters are ignored (you get
  `ps` back unchanged). Use array-valued params (as real Lux layers have).
- **Not here:** the ETGraph-specific `grad_fd` stays in EquivariantTensors.jl —
  moving it would invert the dependency (ACETestUtils would pull in ET + DP).

## Repo conventions

- Julia package; follow existing per-file style; keep lines ≤ 92 chars.
- Don't add dependencies to `Project.toml` without asking.
- Source layout: `gpudetect.jl` (GPU), `fdtests.jl` (FD tests), `utils.jl`
  (display + reinterpret helpers); extensions in `ext/`.
- Docstrings are the canonical API reference; this file and the README are
  overviews.
