using Base.Test
using Meganet

for TYPE=[Float64,Float32]
    K = randn(TYPE,10,5)
    L = getConnector(TYPE,K,outTimes=1)

    @testset "connector $TYPE" begin
     testAbstractMeganetElement(L)
    end

end
