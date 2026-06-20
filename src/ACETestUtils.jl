module ACETestUtils

# display + reinterpretation helpers (print_tf, println_slim, _Vec, _svecs)
include("utils.jl")

# finite-difference gradient-consistency tests (fdtest, dirfdtest)
include("fdtests.jl")

# GPU backend discovery for test suites (gpu_test_backend, detect_gpu_backend)
include("gpudetect.jl")

# Generic Lux-model gradient-testing helpers. The implementations live in
# package extensions that load only when the relevant AD package is present, so
# the heavy AD stack stays out of the core package:
#   * `grad_zy`, `grad_zy_ps`   → `ext/ACETestUtilsZygoteExt.jl`      (Zygote)
#   * `grad_fwd`, `grad_fwd_ps` → `ext/ACETestUtilsForwardDiffExt.jl` (ForwardDiff)
# `fwd` denotes ForwardDiff (not finite difference — cf. `fdtest`). Calling
# these without the relevant package loaded throws a `MethodError`.
export grad_zy, grad_zy_ps, grad_fwd, grad_fwd_ps

"""
    grad_zy(X, model, ps, st)

Zygote gradient of `model(·, ps, st)[1]` with respect to the input `X`.
Requires `Zygote` to be loaded (extension).
"""
function grad_zy end

"""
    grad_zy_ps(X, model, ps, st)

Zygote gradient of `model(X, ·, st)[1]` with respect to the parameters `ps`.
Requires `Zygote` to be loaded (extension).
"""
function grad_zy_ps end

"""
    grad_fwd(X, model, ps, st)

ForwardDiff gradient of `model(·, ps, st)[1]` with respect to the input `X`
(`X` must be a real array). Requires `ForwardDiff` to be loaded (extension).
"""
function grad_fwd end

"""
    grad_fwd_ps(G, model, ps, st)

ForwardDiff gradient of `model(G, ·, st)[1]` with respect to the parameters
`ps`, via `Optimisers.destructure`. Requires `ForwardDiff` to be loaded
(extension).
"""
function grad_fwd_ps end

end # module ACETestUtils
