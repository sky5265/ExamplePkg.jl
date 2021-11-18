using Test
using ExamplePkg
using Distributions

@testset "std_normal" begin
   dist = Normal()
   @test pdf_normal(0.0, 0.0, 1.0) ≈ pdf(dist,0.0)
   @test pdf_normal(2.0, 0.0, 1.0) ≈ pdf(dist,2.0)
   @test pdf_normal(-1.0, 0.0, 1.0) ≈ pdf(dist,-1.0)
end;

@testset "normal" begin
   dist = Normal(1.0,2.0)
   @test pdf_normal(0.0, 1.0, 2.0) ≈ pdf(dist,0.0)
   @test pdf_normal(2.0, 1.0, 2.0) ≈ pdf(dist,2.0)
   @test pdf_normal(-1.0, 1.0, 2.0) ≈ pdf(dist,-1.0)
end;