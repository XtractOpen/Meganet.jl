using Base.Test
using Meganet

for TYPE=[Float64,Float32]
    K      = getDenseKernel(TYPE,[32,18])
    nex    = 8
    Bin    = randn(TYPE,nFeatOut(K),4)
    Bout   = randn(TYPE,nFeatIn(K),3)
    nLayer = getTVNormLayer(TYPE,[8,4])
    L      = getDoubleSymLayer(TYPE,K,nLayer,Bin,Bout)
    @testset "doubleSymLayer (dense/TV) $TYPE" begin
     testAbstractMeganetElement(L)
    end

    K      = getDenseKernel(TYPE,[32,18])
    nex    = 8
    Bin    = randn(TYPE,nFeatOut(K),4)
    Bout   = randn(TYPE,nFeatIn(K),3)
    nLayer = getBatchNormLayer(TYPE,[8,4])
    L      = getDoubleSymLayer(TYPE,K,nLayer,Bin,Bout)
    @testset "doubleSymLayer (dense/Batch) $TYPE" begin
     testAbstractMeganetElement(L)
    end

    nImg   = [32 32]
    nc     = 16
    nex    = 50
    K      = getSparseConvKernel2D(TYPE,nImg,[3,3,1,nc])
    Bin    = randn(TYPE,nFeatOut(K),4)
    Bout   = randn(TYPE,nFeatIn(K),3)
    nLayer = getBatchNormLayer(TYPE,[prod(nImg),nc],isTrainable=false)
    L      = getDoubleSymLayer(TYPE,K,nLayer,Bin,Bout)
    @testset "doubleSymLayer (conv/Batch/not trainable) $TYPE" begin
     testAbstractMeganetElement(L,nex=nex)
    end


    nImg   = [8 4]
    nc     = 3
    nex    = 4
    K      = getSparseConvKernel2D(TYPE,nImg,[3,3,1,nc])
    Bin    = randn(TYPE,nFeatOut(K),4)
    Bout   = randn(TYPE,nFeatIn(K),3)
    nLayer = getBatchNormLayer(TYPE,[prod(nImg),nc])
    L      = getDoubleSymLayer(TYPE,K,nLayer,Bin,Bout)
    @testset "doubleSymLayer (conv/Batch) $TYPE" begin
     testAbstractMeganetElement(L,nex=nex)
    end

    nImg   = [16 8]
    nc     = 6
    nex    = 8
    K      = getSparseConvKernel2D(TYPE,nImg,[3,3,1,nc])
    Bin    = randn(TYPE,nFeatOut(K),4)
    Bout   = randn(TYPE,nFeatIn(K),3)
    nLayer = getTVNormLayer(TYPE,[prod(nImg),nc])
    L      = getDoubleSymLayer(TYPE,K,nLayer,Bin,Bout)
    @testset "doubleSymLayer (conv/TV) $TYPE" begin
     testAbstractMeganetElement(L)
    end
end
