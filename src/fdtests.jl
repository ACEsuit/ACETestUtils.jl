# Finite-difference gradient-consistency tests, ported from `ACEbase.Testing`
# (see ACEsuit/ACEbase.jl#11). The port is faithful — behaviour is preserved.
# `test_fio` is intentionally left in ACEbase (it depends on `ACEbase.FIO`).
# `_Vec` / `_svecs` live in `utils.jl`.

using Printf
using LinearAlgebra: norm
using StaticArrays: StaticVector

export fdtest, dirfdtest

fdtest(F, dF, X::AbstractVector{<: StaticVector{3}}; kwargs...) =
      fdtest( x -> F(_svecs(x)),
              x -> _Vec(dF(_svecs(x))),
              _Vec(X); kwargs... )

fdtest(F, dF, X::Number; kwargs...) =
      fdtest( x -> F(x[1]),
              x -> [dF(x[1])],
              [X]; kwargs... )


"""
first-order finite-difference test for scalar F
```julia
fdtest(F, dF, x; h0 = 1.0, verbose=true)
```
"""
function fdtest(F, dF, x::AbstractVector; h0 = 1.0, verbose=true)
   errors = Float64[]
   E = F(x)
   dE = dF(x)
   # loop through finite-difference step-lengths
   verbose && @printf("---------|----------- \n")
   verbose && @printf("    h    | error \n")
   verbose && @printf("---------|----------- \n")
   for p = 2:11
      h = 0.1^p
      dEh = copy(dE)
      for n = 1:length(dE)
         x[n] += h
         dEh[n] = (F(x) - E) / h
         x[n] -= h
      end
      push!(errors, norm(dE - dEh, Inf))
      verbose && @printf(" %1.1e | %4.2e  \n", h, errors[end])
   end
   verbose && @printf("---------|----------- \n")
   if minimum(errors) <= 1e-3 * maximum(errors)
      verbose && println("passed")
      return true
   else
      @warn("""It seems the finite-difference test has failed, which indicates
      that there is an inconsistency between the function and gradient
      evaluation. Please double-check this manually / visually. (It is
      also possible that the function being tested is poorly scaled.)""")
      return false
   end
end

dirfdtest(F, dF, x, u; kwargs...) =
      fdtest(t -> F(x + t * u),
             t -> dF(x + t * u) .* Ref(u),
             0.0; kwargs...)
