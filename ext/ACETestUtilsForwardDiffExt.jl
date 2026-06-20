module ACETestUtilsForwardDiffExt

# ForwardDiff-based gradient-testing helpers (differentiation w.r.t. the input
# and w.r.t. the parameters), ported from / analogous to EquivariantTensors.jl
# `test/test_utils/diffutils.jl` (see ACEsuit/EquivariantTensors.jl#132). The
# parameter variant uses `Optimisers.destructure`, which is a core ACETestUtils
# dependency so it is available without being an extension trigger.

import ACETestUtils
import ForwardDiff
using Optimisers: destructure

ACETestUtils.grad_fwd(X, model, ps, st) =
      ForwardDiff.gradient(x -> model(x, ps, st)[1], X)

function ACETestUtils.grad_fwd_ps(G, model, ps, st)
   p_flat, rebuild = destructure(ps)
   _eval_p(p) = model(G, rebuild(p), st)[1]
   ∇p_flat = ForwardDiff.gradient(_eval_p, p_flat)
   return rebuild(∇p_flat)
end

end
