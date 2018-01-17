using Base.Test
using Meganet
using LinearOperators


nImg = [8,10]
sK   = [3,3,4,4]

K = getConvFFTKernel(Float64,nImg,sK)

@testset  "adjoint test" begin
theta = initTheta(K)
A     = getOp(K,theta);
v     = randn(nFeatIn(K))
w     = randn(nFeatOut(K))

t1    = dot(w,A*v)
t2    = dot(v,A'*w)
# println("adjointTest t1=$t1\t t2=$t2")
@test norm(t1-t2)/norm(t1) < 1e-10
end

@testset "derivative Test" begin
th  = initTheta(K);
dth = initTheta(K);
nex = 1;
Y  = randn(nFeatIn(K),nex)+nex;
Z  = randn(nFeatOut(K),nex)-nex;

t1 = vec(Z)'*vec(Jthetamv(K,dth,th,Y));
t2 = vec(dth)'*vec(JthetaTmv(K,Z,th,Y));
# println("derivativeTest t1=$t1\t t2=$t2")
@test norm(t1-t2)/norm(t2) < 1e-10
end
