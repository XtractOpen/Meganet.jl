using Base.Test
using Meganet

K      = getDenseKernel(Float64,[32,18])
nex    = 8
Bin    = randn(Float64,nFeatOut(K),4)
Bout   = randn(Float64,nFeatIn(K),3)
nLayer = getTVNormLayer(Float64,[8,4])
L      = getDoubleSymLayer(Float64,K,nLayer,Bin,Bout)
@testset "doubleSymLayer (dense/TV)" begin
 testAbstractMeganetElement(L)
end

K      = getDenseKernel(Float64,[32,18])
nex    = 8
Bin    = randn(Float64,nFeatOut(K),4)
Bout   = randn(Float64,nFeatIn(K),3)
nLayer = getBatchNormLayer(Float64,[8,4])
L      = getDoubleSymLayer(Float64,K,nLayer,Bin,Bout)
@testset "doubleSymLayer (dense/Batch)" begin
 testAbstractMeganetElement(L)
end

nImg   = [32 32]
nc     = 16
nex    = 50
K      = getSparseConvKernel2D(Float64,nImg,[3,3,1,nc])
Bin    = randn(Float64,nFeatOut(K),4)
Bout   = randn(Float64,nFeatIn(K),3)
nLayer = getBatchNormLayer(Float64,[prod(nImg),nc],isTrainable=false)
L      = getDoubleSymLayer(Float64,K,nLayer,Bin,Bout)
@testset "doubleSymLayer (conv/Batch/not trainable)" begin
 testAbstractMeganetElement(L,nex=nex)
end


nImg   = [8 4]
nc     = 3
nex    = 4
K      = getSparseConvKernel2D(Float64,nImg,[3,3,1,nc])
Bin    = randn(Float64,nFeatOut(K),4)
Bout   = randn(Float64,nFeatIn(K),3)
nLayer = getBatchNormLayer(Float64,[prod(nImg),nc])
L      = getDoubleSymLayer(Float64,K,nLayer,Bin,Bout)
@testset "doubleSymLayer (conv/Batch)" begin
 testAbstractMeganetElement(L,nex=nex)
end

nImg   = [16 8]
nc     = 6
nex    = 8
K      = getSparseConvKernel2D(Float64,nImg,[3,3,1,nc])
Bin    = randn(Float64,nFeatOut(K),4)
Bout   = randn(Float64,nFeatIn(K),3)
nLayer = getTVNormLayer(Float64,[prod(nImg),nc])
L      = getDoubleSymLayer(Float64,K,nLayer,Bin,Bout)
@testset "doubleSymLayer (conv/TV)" begin
 testAbstractMeganetElement(L)
end
