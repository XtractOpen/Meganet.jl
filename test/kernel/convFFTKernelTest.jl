using Base.Test
using Meganet
using LinearOperators


nImg = [8,10]
sK   = [3,3,4,4]
for TYPE=[Float64,Float32]
    K = getConvFFTKernel(TYPE,nImg,sK)

    @testset  "adjoint test $TYPE" begin
	nex = 2;
    theta = initTheta(K)
    A     = getOp(K,theta);
    v     = randn(TYPE,nFeatIn(K),nex)
    w     = randn(TYPE,nFeatOut(K),nex)

    t1    = vecdot(w,A*v)
    t2    = vecdot(v,A'*w)
	
    # println("adjointTest t1=$t1\t t2=$t2")
    @test norm(t1-t2)/norm(t1) < 1e3*eps(TYPE)
    end

    @testset "derivative Test" begin
    th  = initTheta(K);
    dth = initTheta(K);
    nex = 2;
    Y  = randn(TYPE,nFeatIn(K),nex)+nex;
    Z  = randn(TYPE,nFeatOut(K),nex)-nex;

    t1 = vec(Z)'*vec(Jthetamv(K,dth,th,Y));
    t2 = vec(dth)'*vec(JthetaTmv(K,Z,th,Y));
    # println("derivativeTest t1=$t1\t t2=$t2")
    @test norm(t1-t2)/norm(t2) < 1e3*eps(TYPE)
    end
end
