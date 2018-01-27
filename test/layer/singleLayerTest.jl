using Base.Test
using Meganet

for TYPE=[Float64,Float32]
    K = getDenseKernel(TYPE,[32,18])
    nex = 3
    Bin = randn(TYPE, 32,4)
    Bout = randn(TYPE, 32,3)
    nLayer = getTVNormLayer(TYPE,[8,4])

    L     = getSingleLayer(TYPE,K,nLayer)
    @testset "singleLayer (no bias) $TYPE" begin
     testAbstractMeganetElement(L)
    end

    L     = getSingleLayer(TYPE,K,nLayer,Bin=Bin,Bout=Bout)
    @testset "singleLayer (both biases) $TYPE" begin
     testAbstractMeganetElement(L)
    end


    nImg   = [8 4]
    nc     = 3
    nex    = 4
    K      = getSparseConvKernel2D(TYPE,nImg,[3,3,1,nc])
    Bin    = randn(TYPE, nFeatOut(K),4)
    nLayer = getBatchNormLayer(TYPE,[prod(nImg),nc],isTrainable=true).layers[2]
    L      = getSingleLayer(TYPE,K,nLayer,Bin=Bin)
    @testset "singleLayer (conv/Batch/not trainable) $TYPE" begin
     testAbstractMeganetElement(L,nex=nex)
end
end
