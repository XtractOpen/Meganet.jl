using MAT, Meganet, JLD, BenchmarkTools, ProfileView

ENV["LD_LIBRARY_PATH"] = "/home/klensink/.juliapro/JuliaPro-0.6.2.1/JuliaPro/pkgs-0.6.2.1/v0.6/Meganet/src/mkl/"
BLAS.set_num_threads(1)
function test1(con, Y0)
    Y = con.K * Y0
    return Y
end

function test2(Y0, n)
    Y = jumping_average(Y0, n)
    return Y
end

function testT1(con, Y0)
    Y = con.K' * Y0
    return Y
end

function testT2(Y0, n, scale)
    Y = jumping_averageT(Y0, n, scale)
    return Y
end

# Load data
srand(123)
n = 512;

miniBatchSize = 64;
nImg = [32; 32]
nc   = 16
TYPE = Float32
Y0 = rand(TYPE, nc*prod(nImg), 64)
W = vcat([i*nc .+ (collect(1.0:64.0))' for i in 1:nc]...)

# Connector
B   = kron(speye(TYPE, nc),ones(TYPE, prod(nImg)))/prod(nImg);
con = getConnector(TYPE, B')
con2 = getConnector(TYPE, Y -> jumping_average(Y, prod(nImg)),
                          W -> jumping_averageT(W, prod(nImg)),
                          nc*prod(nImg),
                          nc)

Y1  = test1(con, Y0)
Y2  = test2(Y0, prod(nImg))

YT1 = testT1(con, W)
YT2 = testT2(W, prod(nImg), true)

t1  = @benchmark test1($con, $Y0)
t2  = @benchmark test2($Y0, $(prod(nImg)))
tT1 = @benchmark testT1($con, $W)
tT2 = @benchmark testT2($W, $(prod(nImg)), true)

j  = judge(minimum(t2), minimum(t1))
jT = judge(minimum(tT2), minimum(tT1))
