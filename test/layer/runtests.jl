@testset "AffineScalingLayer" begin
    include("affineScalingLayerTest.jl")
end

@testset "DoubleSymLayer" begin
    include("doubleSymLayerTest.jl")
end

@testset "NormLayer" begin
    include("normLayerTest.jl")
end

@testset "SingleLayer" begin
    include("singleLayerTest.jl")
end
