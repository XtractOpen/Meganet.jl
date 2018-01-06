using Base.Test
using Meganet

K = DenseKernel([18,18])
nex = 3
Bin = randn(18,4)
Bout = randn(18,3)
nLayer = getBatchNormLayer([6,3],isTrainable=true)
L     = singleLayer(K,nLayer,Bout=Bout)

net = ResNN(L,4,.1)

@testset "ResNN" begin
 testAbstractMeganetElement(net)
end
