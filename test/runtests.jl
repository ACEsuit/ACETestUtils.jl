using ACETestUtils, Test

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

end
