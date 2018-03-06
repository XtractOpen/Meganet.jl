using Base.Test
using Meganet

for TYPE=[Float64,Float32]
    K = randn(TYPE,10,5)
    L = getConnector(TYPE,K,outTimes=1)

    @testset "connector $TYPE" begin
     testAbstractMeganetElement(L)
    end

end

for TYPE=[Float64,Float32]
    @testset "connector Function $TYPE" begin
        nImg = [32; 32]
        nc   = 16
        Y0 = rand(TYPE, nc*prod(nImg), 64)
        W = vcat([i*nc .+ (collect(1.0:64.0))' for i in 1:nc]...)

        # Connector
        B   = kron(speye(TYPE, nc),ones(TYPE, prod(nImg)))/prod(nImg);
        con = getConnector(TYPE, B')
        con2 = getConnector(TYPE, Y -> jumping_average(Y, prod(nImg)),
                                  W -> jumping_averageT(W, prod(nImg)),
                                  nc*prod(nImg),
                                  nc)

        Y1  = con.K * Y0
        Y2  = jumping_average(Y0, prod(nImg))
        @test isapprox(Y1, Y2)

        YT1  = con.K' * W
        YT2  = jumping_averageT(W, prod(nImg), true)
        @test isapprox(YT1, YT2)
    end
end
