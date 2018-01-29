using Base.Test
using Meganet

for TYPE=[Float64,Float32]
    K1 = getDenseKernel(TYPE,[18,10])
    nL = getTVNormLayer(TYPE,[6,3])

    L1 = getSingleLayer(TYPE,K1,nL)


    K2 = getDenseKernel(TYPE,[8,18])
    nL = getTVNormLayer(TYPE,[2,4])
    L2 = getSingleLayer(TYPE,K2,nL)

    blocks = Array{AbstractMeganetElement{TYPE}}(2)
    blocks[1] = L1;
    blocks[2] = L2;
    net = getNN(blocks)

    @testset "NN $TYPE" begin
     testAbstractMeganetElement(net)
    end
end
