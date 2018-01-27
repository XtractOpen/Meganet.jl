using Base.Test
using Meganet

nK = [10;8]
nT = 5;
for TYPE=[Float64,Float32]
    A = sprandn(nK[1],nK[2],.5)
    Q = sprandn(nnz(A),nT,.7)
    K = getSparseKernel(TYPE,A,Q)

    @testset  "adjoint test ($TYPE)" begin
    theta = initTheta(K)
    A     = getOp(K,theta);

    v     = randn(TYPE,nFeatIn(K))
    w     = randn(TYPE,nFeatOut(K))

    t1    = dot(w,A*v)
    t2    = dot(v,A'*w)
    @test norm(t1-t2)/norm(t1) < 1e3*eps(TYPE)
    end

    @testset "derivative Test ($TYPE)" begin
    th  = initTheta(K);
    dth = initTheta(K);
    nex = 1;
    Y  = randn(TYPE,nFeatIn(K),nex)+nex;
    Z  = randn(TYPE,nFeatOut(K),nex)-nex;

    t1 = vec(Z)'*vec(Jthetamv(K,dth,th,Y));
    t2 = vec(dth)'*vec(JthetaTmv(K,Z,th,Y));
    # println("derivativeTest t1=$t1\t t2=$t2")
    @test norm(t1-t2)/norm(t2) < 1e3*eps(TYPE)
    end

    @testset "derivative Test ($TYPE)" begin
    th  = initTheta(K);
    dth = initTheta(K);
    nex = 1;
    Y  = randn(TYPE,nFeatIn(K),nex)+nex;
    Z  = randn(TYPE,nFeatOut(K),nex)-nex;

    t2 = vec(dth)'*vec(JthetaTmv(K,Z,th,Y));
    t1 = vec(Z)'*vec(Jthetamv(K,dth,th,Y));
    # println("derivativeTest t1=$t1\t t2=$t2")
    @test norm(t1-t2)/norm(t2) < 1e3*eps(TYPE)
    end
end
