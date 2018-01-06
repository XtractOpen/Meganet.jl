@testset "DenseKernelType" begin
include("denseKernelTest.jl")
end

@testset "SparseKernelType" begin
include("sparseKernelTest.jl")
end

@testset "convFFTKernel" begin
include("convFFTKernelTest.jl")
end
