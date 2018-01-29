using Meganet
using Base.Test

for TYPE=[Float64,Float32]
    S = getSoftMaxLoss(TYPE)
    display(S)

    @testset "softmax (with bias) $TYPE" begin
        testLossFunction(S,out=false)
    end

    S = getSoftMaxLoss(TYPE)
    S.addBias = false
    @testset "softmax (no bias) $TYPE" begin
        testLossFunction(S,out=false)
    end
end
