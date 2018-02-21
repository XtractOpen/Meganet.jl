@testset "balance" begin
    nex = 512
    batchsize = 64
    @test Meganet.balance(nex, batchsize, 1) == [1:512]
    @test Meganet.balance(nex, batchsize, 2) == [1:256, 257:512]
    @test Meganet.balance(nex, batchsize, 3) == [1:128, 129:256, 257:384]

    nex = 10
    batchsize = 1
    @test Meganet.balance(nex, batchsize, 1) == [1:10]
    @test Meganet.balance(nex, batchsize, 5) == [1:2, 3:4, 5:6, 7:8, 9:10]
    @test Meganet.balance(nex, batchsize, 6) == [1:1, 2:2, 3:3, 4:4, 5:5, 6:6]
end

@testset "mean" begin

    n = 3
    a = zeros(n,n)
    for i in 1:n
        a[:,i] .= i
    end

    @test mean(x -> x, a, 1) == [1.0 2.0 3.0]
    @test mean(x -> x^2, a, 1) == [1.0 4.0 9.0]
    @test mean(x -> x+1, a, 2) == [3.0 3.0 3.0]'

    m1 = mean(x -> x, rand(3,3), 1)
    m2 = mean(x -> x, rand(3,3), 2)
    @test mean!(x -> x, m1, a) == [1.0 2.0 3.0]
    @test mean!(x -> x^2, m1, a) == [1.0 4.0 9.0]
    @test mean!(x -> x+1, m2, a) == [3.0 3.0 3.0]'

end
