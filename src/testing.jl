# Finite-difference and display test helpers, ported from `ACEbase.Testing`
# (see ACEsuit/ACEbase.jl#11). The port is faithful — behaviour is preserved;
# `test_fio` is intentionally left in ACEbase as it depends on `ACEbase.FIO`.

using Test, Printf
using LinearAlgebra: norm
using StaticArrays: StaticVector, SVector

export print_tf, println_slim, h0, h1, h2, h3, fdtest, dirfdtest

function h0(str)
   dashes = "≡"^(length(str)+4)
   printstyled(dashes, color=:magenta); println()
   printstyled("  "*str*"  ", bold=true, color=:magenta); println()
   printstyled(dashes, color=:magenta); println()
end

function h1(str)
   dashes = "="^(length(str)+2)
   printstyled(dashes, color=:magenta); println()
   printstyled(" " * str * " ", bold=true, color=:magenta); println()
   printstyled(dashes, color=:magenta); println()
end

function h2(str)
   dashes = "-"^length(str)
   printstyled(dashes, color=:magenta); println()
   printstyled(str, bold=true, color=:magenta); println()
   printstyled(dashes, color=:magenta); println()
end

h3(str) = (printstyled(str, bold=true, color=:magenta); println())


print_tf(::Test.Pass) = printstyled("+", bold=true, color=:green)
print_tf(::Test.Fail) = printstyled("-", bold=true, color=:red)
print_tf(::Tuple{Test.Error,Bool}) = printstyled("x", bold=true, color=:magenta)

println_slim(::Test.Pass) = printstyled("Test Passed\n", bold=true, color=:green)
println_slim(::Test.Fail) = printstyled("Test Failed\n", bold=true, color=:red)

_Vec(X::AbstractVector{<: StaticVector{3}}) =
      collect(reinterpret(Float64, X))

_svecs(x::AbstractVector{T}) where {T} =
      collect(reinterpret(SVector{3, T}, x))


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
