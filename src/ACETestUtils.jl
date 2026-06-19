module ACETestUtils

using MLDataDevices
import Pkg

export gpu_test_backend

# Memoized result of `gpu_test_backend()`. The probe + on-demand `Pkg.add`
# happen at most once per session; subsequent calls return this cached
# NamedTuple. Replaces the old `Main`-global include-guard in `utils_gpu.jl`.
const _GPU_TEST_BACKEND = Ref{Any}(nothing)

# Finite-difference / display test helpers (fdtest, dirfdtest, print_tf, … )
include("testing.jl")

# Lux-model gradient-testing helpers. These are implemented in
# `ext/ACETestUtilsGradExt.jl`, which loads only when Zygote, ForwardDiff and
# Optimisers are all present, so the heavy AD stack stays out of the core
# package. Calling these without those packages loaded throws a MethodError.
export grad_zy, grad_zy_ps, grad_fd_ps

"""
    grad_zy(X, model, ps, st)

Zygote gradient of `model(·, ps, st)[1]` with respect to the input `X`.
Requires `Zygote`, `ForwardDiff` and `Optimisers` to be loaded (extension).
"""
function grad_zy end

"""
    grad_zy_ps(X, model, ps, st)

Zygote gradient of `model(X, ·, st)[1]` with respect to the parameters `ps`.
Requires `Zygote`, `ForwardDiff` and `Optimisers` to be loaded (extension).
"""
function grad_zy_ps end

"""
    grad_fd_ps(G, model, ps, st)

ForwardDiff gradient of `model(G, ·, st)[1]` with respect to the parameters
`ps`, via `Optimisers.destructure`. Requires `Zygote`, `ForwardDiff` and
`Optimisers` to be loaded (extension).
"""
function grad_fd_ps end

"""
    detect_gpu_backend() -> String

Pick a GPU backend by probing the *system* — no GPU package is loaded here, so
the default CI runner resolves to `"CPU"` and installs no GPU package. Set the
`TEST_BACKEND` env var to force a choice (`"CPU"`, `"CUDA"`, `"AMDGPU"`,
`"Metal"`, `"oneAPI"`).
"""
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

"""
    gpu_test_backend() -> (; dev, gpu, gpu_backend, gpu_supports_f64)

Discover the GPU backend available for a test suite and return the tools needed
to run device-agnostic tests. Memoized: the system probe and the on-demand
`Pkg.add` of the matching backend run at most once per session.

- `dev` / `gpu` — host→device transfer *function* (`gpu` is an alias of `dev`);
  `identity` on CPU.
- `gpu_backend::String` — `"CUDA"`, `"AMDGPU"`, `"Metal"`, `"oneAPI"`, or
  `"CPU"`.
- `gpu_supports_f64::Bool` — `false` on F32-only backends (Metal); tests run F32
  always and F64 only when this is `true`.

When a GPU is detected, the matching backend is installed *into the (sandboxed)
test env* and used; the default CI runner resolves to `"CPU"` and installs
nothing. A detected-but-unusable backend degrades to CPU with a warning so the
suite still runs.
"""
function gpu_test_backend()
   isnothing(_GPU_TEST_BACKEND[]) || return _GPU_TEST_BACKEND[]

   gpu_backend = detect_gpu_backend()
   gpu = dev = identity
   gpu_supports_f64 = true

   if gpu_backend != "CPU"
      try
         Pkg.add(gpu_backend)                  # into the sandboxed test env only
         # Load the just-installed backend and grab the module object. Unlike a
         # top-level `using`, this runs inside a function, so we resolve the
         # module via `Base.require` (no compile-time binding in this module)
         # and call into the freshly-loaded code with `Base.invokelatest` to
         # avoid a world-age error.
         Backend = Base.require(Main, Symbol(gpu_backend))
         if gpu_backend == "CUDA"
            @assert Base.invokelatest(Backend.functional)
            gpu = dev = Backend.cu
         elseif gpu_backend == "Metal"
            @assert Base.invokelatest(Backend.functional)
            gpu = dev = Backend.mtl
            gpu_supports_f64 = false                     # Metal is F32-only
         elseif gpu_backend == "AMDGPU"
            @assert Base.invokelatest(Backend.functional)
            gpu = dev = Base.invokelatest(MLDataDevices.gpu_device)
         elseif gpu_backend == "oneAPI"
            @assert Base.invokelatest(Backend.functional)
            gpu = dev = Backend.oneArray
         else
            error("unknown TEST_BACKEND = $(gpu_backend)")
         end
         @info "GPU test backend: $(gpu_backend) (F64 supported: $(gpu_supports_f64))"
      catch e
         @warn "GPU backend '$(gpu_backend)' detected but not usable; using CPU." exception=(e, catch_backtrace())
         gpu_backend = "CPU"
         gpu = dev = identity
         gpu_supports_f64 = true
      end
   end

   gpu_backend == "CPU" && @info "GPU test backend: CPU (dev = identity)."

   _GPU_TEST_BACKEND[] = (; dev, gpu, gpu_backend, gpu_supports_f64)
   return _GPU_TEST_BACKEND[]
end

end # module ACETestUtils
