using Base.Test
using Meganet

for TYPE=[Float64,Float32]
    K = getDenseKernel(TYPE,[18,18])
    nex = 3
    Bin = randn(TYPE,18,4)
    Bout = randn(TYPE,18,3)
    nLayer = getBatchNormLayer(TYPE,[6,3],isTrainable=true)
    L     = getSingleLayer(TYPE,K,nLayer,Bout=Bout)

    net = getResNN(TYPE,L,4,.1)

    @testset "ResNN $TYPE" begin
     testAbstractMeganetElement(net)
    end
end
