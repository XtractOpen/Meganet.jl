using Base.Test
using Meganet


K1 = DenseKernel([18,10])
nL = getTVNormLayer([6,3])

L1 = singleLayer(K1,nL)


K2 = DenseKernel([8,18])
nL = getTVNormLayer([2,4])
L2 = singleLayer(K2,nL)

net = NN([L1;L2])

@testset "NN" begin
 testAbstractMeganetElement(net)
end
