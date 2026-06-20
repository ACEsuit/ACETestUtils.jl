module ACETestUtilsForwardDiffExt

# ForwardDiff-based gradient-testing helper (differentiation w.r.t. parameters),
# ported from EquivariantTensors.jl `test/test_utils/diffutils.jl` (see
# ACEsuit/EquivariantTensors.jl#132). Uses `Optimisers.destructure`, which is a
# core ACETestUtils dependency so it is available without being an extension
# trigger.

import ACETestUtils
import ForwardDiff
using Optimisers: destructure

function ACETestUtils.grad_fd_ps(G, model, ps, st)
   p_flat, rebuild = destructure(ps)
   _eval_p(p) = model(G, rebuild(p), st)[1]
   ∇p_flat = ForwardDiff.gradient(_eval_p, p_flat)
   return rebuild(∇p_flat)
end

end
