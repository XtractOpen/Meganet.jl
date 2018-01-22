using Base.Test
using Meganet


K1 = getDenseKernel(Float64,[18,10])
nL = getTVNormLayer(Float64,[6,3])

L1 = getSingleLayer(Float64,K1,nL)


K2 = getDenseKernel(Float64,[8,18])
nL = getTVNormLayer(Float64,[2,4])
L2 = getSingleLayer(Float64,K2,nL)

net = getNN([L1;L2])

@testset "NN" begin
 testAbstractMeganetElement(net)
end
