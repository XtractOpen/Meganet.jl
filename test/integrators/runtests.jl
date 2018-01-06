@testset "connector" begin
include("connectorTest.jl")
end

@testset "NN" begin
include("NNTest.jl")
end

@testset "ResNN" begin
include("ResNNTest.jl")
end
