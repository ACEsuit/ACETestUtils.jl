# Small display + reinterpretation helpers shared across the test utilities.
# `print_tf` / `println_slim` are ported from `ACEbase.Testing`
# (see ACEsuit/ACEbase.jl#11). `_Vec` / `_svecs` are internal helpers used by
# the finite-difference tests in `fdtests.jl`.

using Test
using StaticArrays: StaticVector, SVector

export print_tf, println_slim

print_tf(::Test.Pass) = printstyled("+", bold=true, color=:green)
print_tf(::Test.Fail) = printstyled("-", bold=true, color=:red)
print_tf(::Tuple{Test.Error,Bool}) = printstyled("x", bold=true, color=:magenta)

println_slim(::Test.Pass) = printstyled("Test Passed\n", bold=true, color=:green)
println_slim(::Test.Fail) = printstyled("Test Failed\n", bold=true, color=:red)

_Vec(X::AbstractVector{<: StaticVector{3}}) =
      collect(reinterpret(Float64, X))

_svecs(x::AbstractVector{T}) where {T} =
      collect(reinterpret(SVector{3, T}, x))
