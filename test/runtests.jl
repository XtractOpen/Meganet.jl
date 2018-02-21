using Base.Test
using Meganet

@testset "Meganet" begin
@testset "integrators" begin
    include("integrators/runtests.jl")
end

@testset "kernels" begin
    include("kernel/runtests.jl")
end
@testset "layer" begin
    include("layer/runtests.jl")
end
@testset "optim" begin
	include("optimization/runtests.jl")
end

@testset "loss" begin
	include("loss/runtests.jl")
end

@testset "regularizer" begin
	include("regularization/runtests.jl")
end

@testset "utilities" begin
	include("test_utilities.jl")
end

end
