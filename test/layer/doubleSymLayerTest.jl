using Base.Test
using Meganet

K      = DenseKernel([32,18])
nex    = 8
Bin    = randn(nFeatOut(K),4)
Bout   = randn(nFeatIn(K),3)
nLayer = getTVNormLayer([8,4])
L      = DoubleSymLayer(K,nLayer,Bin,Bout)
@testset "doubleSymLayer (dense/TV)" begin
 testAbstractMeganetElement(L)
end

K      = DenseKernel([32,18])
nex    = 8
Bin    = randn(nFeatOut(K),4)
Bout   = randn(nFeatIn(K),3)
nLayer = getBatchNormLayer([8,4])
L      = DoubleSymLayer(K,nLayer,Bin,Bout)
@testset "doubleSymLayer (dense/Batch)" begin
 testAbstractMeganetElement(L)
end

nImg   = [32 32]
nc     = 16
nex    = 50
K      = getSparseConvKernel2D(nImg,[3,3,1,nc])
Bin    = randn(nFeatOut(K),4)
Bout   = randn(nFeatIn(K),3)
nLayer = getBatchNormLayer([prod(nImg),nc],isTrainable=false)
L      = DoubleSymLayer(K,nLayer,Bin,Bout)
@testset "doubleSymLayer (conv/Batch/not trainable)" begin
 testAbstractMeganetElement(L,nex=nex)
end


nImg   = [8 4]
nc     = 3
nex    = 4
K      = getSparseConvKernel2D(nImg,[3,3,1,nc])
Bin    = randn(nFeatOut(K),4)
Bout   = randn(nFeatIn(K),3)
nLayer = getBatchNormLayer([prod(nImg),nc])
L      = DoubleSymLayer(K,nLayer,Bin,Bout)
@testset "doubleSymLayer (conv/Batch)" begin
 testAbstractMeganetElement(L,nex=nex)
end

nImg   = [16 8]
nc     = 6
nex    = 8
K      = getSparseConvKernel2D(nImg,[3,3,1,nc])
Bin    = randn(nFeatOut(K),4)
Bout   = randn(nFeatIn(K),3)
nLayer = getTVNormLayer([prod(nImg),nc])
L      = DoubleSymLayer(K,nLayer,Bin,Bout)
@testset "doubleSymLayer (conv/TV)" begin
 testAbstractMeganetElement(L)
end
