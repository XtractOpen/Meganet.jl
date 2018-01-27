using Meganet
using Base.Test

for TYPE=[Float64,Float32]
    @testset "Tikhonov $TYPE" begin
    S = getTikhonovReg(TYPE)

    x = randn(TYPE,10)
    Sc,dS,d2S = regularizer(S,x)

    @test norm(x-dS)/norm(x) < 1e3*eps(TYPE)
    end
end
