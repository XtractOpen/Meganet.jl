using Base.Test

@testset "Layers" begin
include("affineScalingLayerTest.jl")
include("doubleSymLayerTest.jl")
include("singleLayerTest.jl")
include("normLayerTest.jl")
end
