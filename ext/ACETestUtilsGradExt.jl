module ACETestUtilsGradExt

# Generic Lux-model gradient-testing helpers, ported from EquivariantTensors.jl
# `test/test_utils/diffutils.jl` (see ACEsuit/EquivariantTensors.jl#132). Only
# the model-agnostic helpers live here; the ETGraph-specific `grad_fd` stays in
# EquivariantTensors to avoid an inverted dependency.

import ACETestUtils
import Zygote, ForwardDiff
using Optimisers: destructure

ACETestUtils.grad_zy(X, model, ps, st) =
      Zygote.gradient(G -> model(G, ps, st)[1], X)[1]

ACETestUtils.grad_zy_ps(X, model, ps, st) =
      Zygote.gradient(_ps -> model(X, _ps, st)[1], ps)[1]

function ACETestUtils.grad_fd_ps(G, model, ps, st)
   p_flat, rebuild = destructure(ps)
   _eval_p(p) = model(G, rebuild(p), st)[1]
   ∇p_flat = ForwardDiff.gradient(_eval_p, p_flat)
   return rebuild(∇p_flat)
end

end
