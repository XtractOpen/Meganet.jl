using Base.Test
using Meganet


K = getDenseKernel(Float64,[32,18])
nex = 3
Bin = randn(32,4)
Bout = randn(32,3)
nLayer = getTVNormLayer(Float64,[8,4])

L     = getSingleLayer(Float64,K,nLayer)
@testset "singleLayer (no bias)" begin
 testAbstractMeganetElement(L)
end

L     = getSingleLayer(Float64,K,nLayer,Bin=Bin,Bout=Bout)
@testset "singleLayer (both biases)" begin
 testAbstractMeganetElement(L)
end


nImg   = [8 4]
nc     = 3
nex    = 4
K      = getSparseConvKernel2D(Float64,nImg,[3,3,1,nc])
Bin    = randn(nFeatOut(K),4)
nLayer = getBatchNormLayer(Float64,[prod(nImg),nc],isTrainable=true).layers[2]
L      = getSingleLayer(Float64,K,nLayer,Bin=Bin)
@testset "singleLayer (conv/Batch/not trainable)" begin
 testAbstractMeganetElement(L,nex=nex)
end
