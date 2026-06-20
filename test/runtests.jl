using ACETestUtils, Test
using StaticArrays: SVector
using Zygote, ForwardDiff   # trigger ACETestUtilsZygoteExt / ACETestUtilsForwardDiffExt

@testset "ACETestUtils.jl" begin

   @testset "gpu_test_backend contract" begin
      nt = gpu_test_backend()
      @test nt.gpu_backend isa String
      @test nt.dev === identity || nt.gpu_backend != "CPU"
      @test nt.dev([1.0, 2.0]) isa AbstractVector     # transfer fn callable
      @test nt.gpu_supports_f64 isa Bool
      @test nt.gpu === nt.dev                          # `gpu` aliases `dev`
      # idempotent / memoized:
      @test gpu_test_backend().gpu_backend == nt.gpu_backend
      @test gpu_test_backend() === nt
   end

   @testset "fdtest / display helpers" begin
      F  = x -> sum(abs2, x) / 2
      dF = x -> x
      @test fdtest(F, dF, randn(5); verbose=false)              # vector
      @test fdtest(x -> x^2, x -> 2x, 1.3; verbose=false)       # Number
      X = [ SVector{3}(randn(3)) for _ in 1:4 ]                 # SVector{3}
      @test fdtest(x -> sum(sum(abs2, xi) / 2 for xi in x),
                   x -> x, X; verbose=false)
      # a deliberately wrong gradient should fail the test
      @test !fdtest(F, x -> x .+ 1, randn(5); verbose=false)
      # display helpers run without error
      res = @test 1 == 1
      @test (print_tf(res); println_slim(res); true)
   end

   @testset "gradient helpers (extension)" begin
      # array-valued parameters: `Optimisers.destructure` (used by grad_fd_ps)
      # only flattens array leaves, matching real Lux-model usage.
      model(X, ps, st) = (sum(ps.w .* X), st)
      ps = (w = randn(4),)
      st = NamedTuple()
      X  = randn(4)
      @test grad_zy(X, model, ps, st) ≈ ps.w           # ∂/∂X = w
      @test grad_zy_ps(X, model, ps, st).w ≈ X         # ∂/∂w = X
      @test grad_fd_ps(X, model, ps, st).w ≈ X         # ∂/∂w = X
   end

end
