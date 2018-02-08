using Base.Test
using Meganet
using LinearOperators


nImg = [8,10]
sK   = [3,3,4,4]
for TYPE=[Float64,Float32]
    K = getConvGEMMKernel(TYPE,nImg,sK)

    @testset  "adjoint test $TYPE" begin
    theta = initTheta(K)
    A     = getOp(K,theta);
    v     = randn(TYPE,nFeatIn(K))
    w     = randn(TYPE,nFeatOut(K))

    t1    = dot(w,A*v)
    t2    = dot(v,A'*w)
    # println("adjointTest t1=$t1\t t2=$t2")
    @test norm(t1-t2)/norm(t1) < 1e3*eps(TYPE)
    end

    @testset "derivative Test" begin
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

    @testset "new derivitive test" begin
    nImage = [16,16];
	sK = [3,3,2,4];
	K = randn(TYPE,tuple(sK...));
	Y = randn(TYPE,nImage[1],nImage[2],sK[3],2);
	Z = randn(TYPE,nImage[1],nImage[2],sK[4],2);
	Kernel2 = getConvGEMMKernel(TYPE,nImage,sK);
	AY = Amv(Kernel2,K,Y);
    ATZ = ATmv(Kernel2,K,Z);
    
	v1 = vecdot(Z,AY);
	v2 = vecdot(ATZ,Y);
	
	v3 = vecdot(Z,Jthetamv(Kernel2,K,(TYPE)[],Y));
	v4 = vecdot(K,JthetaTmv(Kernel2,Z,(TYPE)[],Y));
    @test norm(v1-v2)/norm(v2) < 1e3*eps(TYPE) && 
    norm(v2-v3)/norm(v3) < 1e3*eps(TYPE) && 
    norm(v3-v4)/norm(v4) < 1e3*eps(TYPE)
    end
end
