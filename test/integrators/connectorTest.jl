using Base.Test
using Meganet

K = randn(10,5)
L = Connector(K,outTimes=1)

@testset "connector" begin
 testAbstractMeganetElement(L)
end
