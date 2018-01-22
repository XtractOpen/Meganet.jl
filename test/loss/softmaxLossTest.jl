using Meganet
using Base.Test

S = getSoftMaxLoss(Float64)
display(S)

@testset "softmax (with bias)" begin
    testLossFunction(S)
end

S = getSoftMaxLoss(Float64)
S.addBias = false
@testset "softmax (no bias)" begin
    testLossFunction(S)
end
