using jInv.Utils
using Meganet
using Base.Test

S = SoftmaxLoss()
display(S)

@testset "softmax (with bias)" begin
    testLossFunction(S)
end

S = SoftmaxLoss()
S.addBias = false
@testset "softmax (no bias)" begin
    testLossFunction(S)
end
