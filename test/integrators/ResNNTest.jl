using Base.Test
using Meganet

K = getDenseKernel(Float64,[18,18])
nex = 3
Bin = randn(18,4)
Bout = randn(18,3)
nLayer = getBatchNormLayer(Float64,[6,3],isTrainable=true)
L     = getSingleLayer(Float64,K,nLayer,Bout=Bout)

net = getResNN(Float64,L,4,.1)

@testset "ResNN" begin
 testAbstractMeganetElement(net)
end
