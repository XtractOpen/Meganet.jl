BLAS.set_num_threads(1)
@inline function rangesum(A::Array{T,2}, R) where {T<:Number}
    s = zero(T)
    @simd for I in R
        @inbounds s += A[I]
    end
    return s
end

function jumping_average(Y0::Array{T,2}, jump::Int) where {T<:Number}
    nf, nex = size(Y0)
    (nf%jump != 0) && error("Columns of Y0 are not divisible by jump length")
    m = div(nf, jump)
    out = Array{T,2}(m, nex)

    for j in 1:nex
        for i in 1:m
           @inbounds  out[i,j] = rangesum(Y0, CartesianRange(CartesianIndex(((i-1)*jump+1,j)), CartesianIndex(i*jump,j))) /jump
        end
    end

    return out
end

function test1(con, Y0)
    Y = con.K * Y0
    return Y
end

function test2(Y0, n)
    Y = jumping_average(Y0, n)
    return Y
end



using MAT, Meganet, JLD, BenchmarkTools, ProfileView
ENV["LD_LIBRARY_PATH"] = "/home/klensink/.juliapro/JuliaPro-0.6.2.1/JuliaPro/pkgs-0.6.2.1/v0.6/Meganet/src/mkl/"
BLAS.set_num_threads(1)

# Load data
srand(123)
n = 512;
#Y_train,C_train,Y_test,C_test = getCIFAR10(n,Pkg.dir("Meganet")*"/data/CIFAR10/");

miniBatchSize = 64;
nImg = [32; 32]
cin  = 3
nc   = [16;32;64;64]
nt   = 2*[1;1;1]
h    = [1.;1.;1.]
TYPE = Float32
Y0 = rand(TYPE, 16*prod(nImg), 64)

# Connector
B   = kron(speye(TYPE,16),ones(TYPE, prod(nImg)))/prod(nImg);
con = getConnector(TYPE, B')
#Ydata, Yout, tmp = apply(con, rand(TYPE, 10), Y0, Array{Any}(0,0))

Y1 = test1(con, Y0)
Y2 = test2(Y0, prod(nImg))

t1 = @benchmark test1($con, $Y0)
t2 = @benchmark test2($Y0, $(prod(nImg)))
